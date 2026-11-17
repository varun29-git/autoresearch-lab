from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import load_checkpoint, restore_rng_state, rng_state, save_checkpoint
from .data import TokenManifest, TokenSpan, PrefetchingTokenLoader
from .integrity import file_sha256, fingerprint, source_hashes, validate_manifest, write_json_atomic
from .model import BitNetLlama, BitNetLlamaConfig
from .optim import OptimizerConfig, build_optimizer


class _PhaseScope:
    def __init__(self, sink, pair):
        self.sink = sink
        self.pair = pair

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.sink is not None:
            self.pair[1].record()
            self.sink.append(self.pair)
        return False


@dataclass(frozen=True)
class TrainConfig:
    device: str = "cuda"
    dtype: str = "bf16"
    development_mode: bool = False
    seed: int = 1337
    gradient_accumulation_steps: int = 1
    warmup_tokens: int = 0
    final_lr_fraction: float = 0.1
    schedule: str = "cosine"
    decay_tokens: int | None = None
    decay_style: str = "cosine"
    phase_start_tokens: int | None = None
    grad_clip: float = 1.0
    quantization_warmup_tokens: int = 0
    quantization_start_strength: float = 1.0
    prefetch_batches: int = 4
    checkpoint_interval: int = 1000
    validation_interval: int = 1000
    validation_batches_per_language: int = 8
    validation_sequences_per_language: int | None = None
    benchmark_tokens: str | None = None
    log_interval: int = 10
    stability_log_interval: int = 0
    compile: bool = False
    cuda_graphs: bool = False
    ddp_bucket_cap_mb: int = 25
    # Whole matrices only. Newton–Schulz still sees each full matrix; ranks that
    # do not own a matrix skip it and then receive the owner's weight and momentum.
    shard_muon: bool = False
    gemm_table: str | None = None
    time_limit_seconds: float | None = None
    async_checkpoint: bool = False
    checkpoint_timeout_seconds: float = 600.0

    def __post_init__(self):
        for name in ("development_mode", "compile", "cuda_graphs", "async_checkpoint", "shard_muon"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.gemm_table is not None and (not isinstance(self.gemm_table, str) or not self.gemm_table):
            raise ValueError("gemm_table must be a path when set")
        if self.device not in {"cpu", "cuda"} or self.dtype not in {"fp32", "bf16"}:
            raise ValueError("trainer supports CPU FP32 and CUDA BF16/FP32, not FP16 latent updates")
        if self.device == "cpu" and self.dtype != "fp32":
            raise ValueError("CPU training is an FP32 correctness reference")
        if self.cuda_graphs and (self.device != "cuda" or self.compile):
            raise ValueError("CUDA graphs require CUDA and cannot be combined with full-model compile")
        for name in ("gradient_accumulation_steps", "prefetch_batches", "checkpoint_interval", "validation_interval", "validation_batches_per_language", "log_interval", "ddp_bucket_cap_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.validation_sequences_per_language is not None and (type(self.validation_sequences_per_language) is not int
                or self.validation_sequences_per_language <= 0):
            raise ValueError("validation_sequences_per_language must be a positive integer when set")
        if self.benchmark_tokens is not None and (not isinstance(self.benchmark_tokens, str) or not self.benchmark_tokens):
            raise ValueError("benchmark_tokens must be a path when set")
        if (isinstance(self.stability_log_interval, bool) or not isinstance(self.stability_log_interval, int)
                or self.stability_log_interval < 0):
            raise ValueError("stability_log_interval must be a nonnegative integer")
        for name in ("warmup_tokens", "quantization_warmup_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.final_lr_fraction <= 1 or not 0 <= self.quantization_start_strength <= 1:
            raise ValueError("invalid learning-rate fraction or quantization strength")
        if self.schedule not in {"cosine", "wsd"}:
            raise ValueError("schedule must be cosine or wsd (warmup-stable-decay)")
        if self.decay_style not in {"linear", "cosine"}:
            raise ValueError("decay_style must be linear or cosine")
        if self.decay_tokens is not None:
            if isinstance(self.decay_tokens, bool) or not isinstance(self.decay_tokens, int) or self.decay_tokens <= 0:
                raise ValueError("decay_tokens must be a positive integer")
            if self.schedule != "wsd":
                raise ValueError("decay_tokens only applies to the wsd schedule")
        if self.schedule == "wsd" and self.decay_tokens is None:
            raise ValueError("the wsd schedule requires an explicit decay_tokens budget")
        if self.phase_start_tokens is not None and (isinstance(self.phase_start_tokens, bool)
                or not isinstance(self.phase_start_tokens, int) or self.phase_start_tokens < 0):
            raise ValueError("phase_start_tokens must be a nonnegative integer")
        if not math.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("grad_clip must be finite and nonnegative; zero disables clipping")
        if self.time_limit_seconds is not None and (not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0):
            raise ValueError("time limit must be finite and positive")
        if not math.isfinite(self.checkpoint_timeout_seconds) or self.checkpoint_timeout_seconds <= 0:
            raise ValueError("checkpoint timeout must be finite and positive")


def load_config(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) != {"model", "training", "optimizer"}:
        raise ValueError("configuration must contain exactly model, training and optimizer sections")
    model, training = BitNetLlamaConfig(**payload["model"]), TrainConfig(**payload["training"])
    if model.linear_backend in {"deepgemm", "rocm_fp8", "rocm_fp8_triton", "rocm_fp8_te"} and training.compile:
        raise ValueError("native DeepGEMM uses its own JIT; full-model torch.compile is not validated")
    if training.cuda_graphs and (model.gradient_checkpointing or model.dropout):
        raise ValueError("CUDA graphs require retained activations and zero dropout")
    return model, training, OptimizerConfig(**payload["optimizer"])


def schedule_scale(tokens_after_update, total_tokens, warmup_tokens, final_lr_fraction,
                   *, schedule="cosine", decay_tokens=None, decay_style="cosine"):
    if not 0 <= warmup_tokens < total_tokens:
        raise ValueError("warmup must be shorter than the training budget")
    if schedule == "wsd":
        if decay_tokens is None or decay_tokens <= 0 or warmup_tokens + decay_tokens > total_tokens:
            raise ValueError("wsd requires 0 < decay_tokens and warmup + decay within budget")
    elif schedule != "cosine":
        raise ValueError("schedule must be cosine or wsd")
    if warmup_tokens and tokens_after_update <= warmup_tokens:
        return tokens_after_update / warmup_tokens
    if schedule == "wsd":
        decay_start = total_tokens - decay_tokens
        if tokens_after_update <= decay_start:
            return 1.0
        progress = min(1.0, (tokens_after_update - decay_start) / decay_tokens)
        if decay_style == "linear":
            return final_lr_fraction + (1 - final_lr_fraction) * (1 - progress)
        return final_lr_fraction + (1 - final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
    if schedule != "cosine":
        raise ValueError("schedule must be cosine or wsd")
    progress = min(1.0, max(0.0, (tokens_after_update - warmup_tokens) / (total_tokens - warmup_tokens)))
    return final_lr_fraction + (1 - final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))


@lru_cache(maxsize=1)
def native_source_contract():
    import importlib.util

    result = {}
    suffixes = {".py", ".so", ".pyd", ".dll", ".h", ".hpp", ".cu", ".cuh", ".cpp", ".cubin", ".ptx", ".hsaco", ".co", ".csv"}
    names = ["flash_attn_interface", "flash_attn_3_cuda", "flash_attn_3", "deep_gemm", "triton", "muon", "quack"]
    if torch.version.hip:
        suffixes |= {".hip", ".s", ".S", ".json", ".cmake"}
        names += ["aiter", "aiter_meta", "flash_attn", "flash_attn_2_cuda", "transformer_engine", "transformer_engine_torch"]
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None or (spec.origin is None and not spec.submodule_search_locations):
            result[name] = None
            continue
        if spec.submodule_search_locations:
            result[name] = {str(path.relative_to(root)): file_sha256(path)
                            for location in spec.submodule_search_locations
                            for root in (Path(location),) for path in sorted(root.rglob("*"))
                            if path.is_file() and path.suffix in suffixes}
        else:
            result[name] = {Path(spec.origin).name: file_sha256(spec.origin)}
    return result


def runtime_contract(device):
    import importlib.metadata
    import platform

    packages = {}
    for name in ("numpy", "sentencepiece", "triton", "flash-attn", "muon-optimizer", "deep-gemm", "quack-kernels"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    result = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "packages": packages,
            "gpu_memory_bytes": properties.total_memory if properties else None,
            "gpu_multiprocessors": properties.multi_processor_count if properties else None,
            "python": platform.python_version(), "native_sources": native_source_contract(),
            "device": device.type, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None}
    if torch.version.hip and device.type == "cuda":
        from production.amd.runtime import hip_contract

        result["rocm"] = hip_contract(device)
    return result


def make_resume_contract(model_config, training, optimizer_config, manifest_report, world_size, device):
    training_values = asdict(training)
    for name in OPERATIONAL_TRAINING_FIELDS:
        training_values.pop(name)
    return {"model": asdict(model_config), "training": training_values,
            "optimizer": json.loads(json.dumps(asdict(optimizer_config))),
            # Node-local staging stats certify reads, not portable checkpoint identity.
            "data": {k: v for k, v in manifest_report.items() if k != "packed_files"},
            "world_size": world_size, "source": source_hashes(), "runtime": runtime_contract(device)}


def append_event(path, payload):
    payload = {"time_unix": time.time(), **payload}
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
    print(encoded, flush=True)


def append_record(path, payload):
    encoded = json.dumps({"time_unix": time.time(), **payload}, allow_nan=False, sort_keys=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()


@torch.no_grad()
def evaluate(model, manifest, root, device, amp_dtype, max_batches, rank, world_size, *, sequences_per_language=None,
             verified_files=None, progress_callback=None):
    was_training = model.training
    state = rng_state()
    model.eval()
    result = {}
    try:
        from .integrity import manifest_scope

        domains = set(manifest_scope(manifest)["domains"])
        if manifest.is_packed:
            from .packed_data import evaluate_packed

            return evaluate_packed(model, manifest, root, device, amp_dtype, max_batches,
                                   rank, world_size, sequences_per_language=sequences_per_language,
                                   verified_files=verified_files, progress_callback=progress_callback)
        val_shards = manifest.shards_for(split=manifest.config.val_split)
        key_of = lambda spec: spec.source if spec.source in domains else spec.language
        languages = sorted({key_of(spec) for spec in val_shards})
        for language in languages:
            span = TokenSpan([spec for spec in val_shards if key_of(spec) == language], root=root)
            blocks = (span.num_tokens - 1) // manifest.config.seq_len
            requested = (max_batches * manifest.config.batch_size * world_size
                         if sequences_per_language is None else sequences_per_language)
            selected = min(blocks, requested)
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            local_indices = range(rank, selected, world_size)
            for start in range(0, len(local_indices), manifest.config.batch_size):
                rows = [span.block((index * (blocks - 1)) // max(1, selected - 1), manifest.config.seq_len)
                        for index in local_indices[start:start + manifest.config.batch_size]]
                inputs, targets = (torch.stack([row[column] for row in rows]).to(device) for column in (0, 1))
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    output = model(inputs, targets, return_logits=False)
                totals[0] += output["loss_sum"].double()
                totals[1] += output["loss_tokens"].double()
                if progress_callback is not None:
                    progress_callback(dict(phase="heldout", key=language,
                        completed_sequences=start+len(rows), total_sequences=len(local_indices)))
            if world_size > 1:
                distributed.all_reduce(totals)
            loss_sum, tokens = totals.tolist()
            if tokens <= 0 or not math.isfinite(loss_sum):
                raise FloatingPointError(f"invalid validation loss or empty language split: {language}")
            mean = loss_sum / tokens
            result[language] = {"loss": mean, "tokens": int(tokens), "perplexity": math.exp(mean) if mean < 700 else None}
    finally:
        model.train(was_training)
        restore_rng_state(state)
    return result


def collective_checkpoint(directory, model, optimizer, state, contract, rank, world_size, *, writer=None, wait=False):
    local_rng = rng_state()
    states = [None] * world_size if rank == 0 else None
    if world_size > 1:
        distributed.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    error = [None]
    if rank == 0:
        try:
            if writer is None:
                save_checkpoint(directory, model, optimizer, state, contract, states)
            else:
                writer.submit(model, optimizer, state, contract, states)
                if wait:
                    writer.flush()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"checkpoint transaction failed: {error[0]}")


def poll_checkpoint_writer(writer, rank, world_size, *, wait=False):
    error = [None]
    if rank == 0 and writer is not None:
        try:
            writer.flush() if wait else writer.check()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"background checkpoint failed: {error[0]}")


OPERATIONAL_TRAINING_FIELDS = ("checkpoint_interval", "validation_interval",
                               "validation_batches_per_language", "validation_sequences_per_language", "benchmark_tokens",
                               "log_interval", "stability_log_interval", "prefetch_batches", "time_limit_seconds",
                               "async_checkpoint", "checkpoint_timeout_seconds")


class PreparedTraining:
    """A prepared run exposing the complete-update measurement boundary.

    complete_update() performs exactly one full optimizer update: scheduled
    LR/phase/quantization setup and gradient clearing; loading the next real
    batches with H2D transfer and all accumulation microsteps; forward, full
    loss, backward and gradient synchronization; nonfinite detection, global
    clipping and all optimizer group updates; FP8 cache invalidation and
    committed data/update counters. It returns the scored update token count
    (planned from descriptors for packed data); the measured valid (-100-excluded)
    target count and the data-wait time for the update are exposed as
    valid_targets and data_wait_seconds. cadence() performs the post-update
    logging/checkpoint/validation/stop work, which is production cadence
    overhead outside the measured callback.
    """

    def __init__(self):
        self.stop_after_updates = None
        self.checkpoint_writer = None
        self.choice_records = None
        self.evaluation_progress = None
        self.monitor = None
        self.captured = None
        self.muon_owners = None
        self.should_stop = [False]
        self.skip_validation = False
        self._phase_timing = os.environ.get("CI1_PHASE_TIMING") == "1"
        self._phase_events = None

    def _phase_mark(self, name):
        if not self._phase_timing:
            return _PhaseScope(None, None)
        if self._phase_events is None:
            self._phase_events = {}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scope = _PhaseScope(self._phase_events.setdefault(name, []), (start, end))
        return scope

    def _phase_report(self):
        if not self._phase_timing or not self._phase_events:
            return None
        torch.cuda.synchronize(self.device)
        totals = {name: sum(s.elapsed_time(e) for s, e in pairs) / 1e3
                  for name, pairs in self._phase_events.items()}
        self._phase_events = None
        return totals

    def complete_update(self):
        training, state = self.training, self.state
        sampled = self.monitor is not None and (state["updates"] == 0 or
                   (state["updates"] + 1) % training.stability_log_interval == 0 or
                   self.stop_after_updates == state["updates"] + 1 or
                   state["next_micro_step"] + training.gradient_accumulation_steps >= self.total_micro_steps)
        if sampled:
            self.monitor.active = True
        micro_steps = min(training.gradient_accumulation_steps, self.total_micro_steps - state["next_micro_step"])
        packed_plan = None
        if self.manifest.is_packed:
            packed_plan = torch.tensor(self.packed_batcher.plan_update(state["next_micro_step"], micro_steps),
                                       dtype=torch.int64, device=self.device)
            if self.world_size > 1:
                distributed.all_reduce(packed_plan)
            planned_targets, planned_content, planned_slots = packed_plan.tolist()
            update_tokens = planned_targets
            if update_tokens <= 0 or state["tokens_seen"] + update_tokens > self.total_tokens:
                raise ValueError("packed update has no labels or exceeds the sealed scored horizon")
        else:
            update_tokens = min(micro_steps * self.micro_tokens, self.total_tokens - state["tokens_seen"])
        scale = schedule_scale(state["tokens_seen"] + update_tokens - self.phase_offset,
                               self.total_tokens - self.phase_offset, training.warmup_tokens,
                               training.final_lr_fraction, schedule=training.schedule,
                               decay_tokens=training.decay_tokens, decay_style=training.decay_style)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        strength = self.model_config.quantization_strength
        if training.quantization_warmup_tokens:
            progress = min(1.0, (state["tokens_seen"] + update_tokens - self.phase_offset) / training.quantization_warmup_tokens)
            strength = training.quantization_start_strength + (strength - training.quantization_start_strength) * progress
        self.model.set_quantization_strength_(strength)
        self.optimizer.zero_grad(set_to_none=not training.cuda_graphs)
        local_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        local_valid = torch.zeros((), dtype=torch.float64, device=self.device)
        data_wait = 0.0
        # static_graph records its hook set on the first backward, and that
        # backward has to synchronize. The opening accumulation step would
        # otherwise run it under no_sync and the reducer aborts. The recording
        # pass is discarded; the same batch is the first real microbatch.
        replay = None
        if self.world_size > 1 and not getattr(self, "ddp_graph_ready", False) and not training.cuda_graphs:
            wait_started = time.monotonic()
            replay = next(self.iterator)
            data_wait += time.monotonic() - wait_started
            batch_index, inputs, targets = replay[:3]
            document_layout = replay[3] if len(replay) == 4 else None
            if batch_index != state["next_micro_step"]:
                raise RuntimeError("prefetch cursor drift")
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                output = self.forward_model(inputs, targets, return_logits=False, **(dict(document_layout=document_layout) if document_layout is not None else {}))
                loss = output["loss_sum"] * (self.world_size / update_tokens)
            loss.backward()
            self.optimizer.zero_grad(set_to_none=not training.cuda_graphs)
            self.model.clear_fp8_weight_cache()
            self.ddp_graph_ready = True
        for micro_index in range(micro_steps):
            if replay is not None:
                batch = replay
                replay = None
            else:
                wait_started = time.monotonic()
                batch = next(self.iterator)
                data_wait += time.monotonic() - wait_started
            batch_index, inputs, targets = batch[:3]
            document_layout = batch[3] if len(batch) == 4 else None
            local_valid += targets.ne(-100).sum()
            if batch_index != state["next_micro_step"] + micro_index:
                raise RuntimeError("prefetch cursor drift")
            if training.cuda_graphs and self.captured is None:
                from .cuda_graph import CapturedMicrostep

                self.captured = CapturedMicrostep(self.model, inputs, targets, amp_dtype=self.amp_dtype,
                                                 document_layout=document_layout)
            if self.captured is not None and self.captured.matches(inputs, targets, document_layout):
                local_loss_sum += self.captured.run(inputs, targets, self.world_size / update_tokens,
                                                   document_layout=document_layout).detach().double()
                continue
            sync_context = self.forward_model.no_sync() if self.world_size > 1 and micro_index + 1 < micro_steps else contextlib.nullcontext()
            with sync_context:
                with self._phase_mark("forward"):
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                        output = self.forward_model(inputs, targets, return_logits=False, **(dict(document_layout=document_layout) if document_layout is not None else {}))
                        loss = output["loss_sum"] * (self.world_size / update_tokens)
                local_loss_sum += output["loss_sum"].detach().double()
                with self._phase_mark("backward"):
                    loss.backward()
            if self.monitor is not None:
                self.monitor.active = False
        elapsed = time.monotonic() - self.started
        stop = self.should_stop[0] or (training.time_limit_seconds is not None and elapsed >= training.time_limit_seconds)
        summary = torch.stack((local_loss_sum, local_loss_sum.new_tensor(float(stop)), local_valid))
        if self.world_size > 1:
            with self._phase_mark("summary_reduce"):
                distributed.all_reduce(summary)
        loss_sum, stop_requests, valid_targets = summary.tolist()
        if not math.isfinite(loss_sum):
            if self.rank == 0:
                append_record(self.output_dir / "incidents.jsonl", {"event": "nonfinite_loss", "state": state})
            raise FloatingPointError("nonfinite global training loss; optimizer update was not applied")
        # Backward used the declared global token count as its denominator.
        # Reject an unexpected loss mask on every rank before it changes
        # weights/moments. Legitimate padded tails still have exactly the
        # declared number of valid targets across the complete global update.
        if valid_targets != update_tokens:
            if self.rank == 0:
                append_record(self.output_dir / "incidents.jsonl", {
                    "event": "token_count_mismatch", "state": state,
                    "valid_targets": valid_targets, "update_tokens": update_tokens})
            raise ValueError(
                f"valid targets {valid_targets:g} differ from nominal update tokens "
                f"{update_tokens}; optimizer update was not applied")
        if (state["next_micro_step"] + micro_steps == self.total_micro_steps and
                state["tokens_seen"] + update_tokens != self.total_tokens):
            raise ValueError("final scored-target total differs from sealed manifest; optimizer update was not applied")
        if sampled:
            self.monitor.gradients_before_clip()
        try:
            with self._phase_mark("clip"):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), training.grad_clip or float("inf"), error_if_nonfinite=True, foreach=True)
        except Exception as exc:
            if self.rank == 0:
                append_record(self.output_dir / "incidents.jsonl", {"event": "gradient_failure", "state": state,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        if sampled:
            self.monitor.before_step()
        with self._phase_mark("optimizer"):
            self.optimizer.step()
            if self.muon_owners is not None:
                from production.amd.sharded_muon import broadcast_owned_muon

                broadcast_owned_muon(self.optimizer, self.muon_owners, distributed)
        if sampled:
            self.monitor.after_step()
        with self._phase_mark("cache_clear"):
            self.model.clear_fp8_weight_cache()
        if self._phase_timing and self.rank == 0:
            report = self._phase_report()
            if report is not None:
                report["data_wait"] = data_wait
                print("PHASE_TIMING " + json.dumps({"update": state["updates"] + 1, **report}), flush=True)
        # Carry sealed phase decisions through every update and checkpoint.
        # Rebuilding just the counters silently discarded ending_commitment.
        self.state = {**state, "next_micro_step": state["next_micro_step"] + micro_steps,
                      "tokens_seen": state["tokens_seen"] + update_tokens, "updates": state["updates"] + 1}
        if packed_plan is not None:
            self.state.update(input_slots_seen=state.get("input_slots_seen", 0)+planned_slots,
                              content_tokens_seen=state.get("content_tokens_seen", 0)+planned_content)
        self.sampled = sampled
        self.update_tokens = update_tokens
        self.valid_targets = int(valid_targets)
        self.data_wait_seconds = data_wait
        self.loss_sum = loss_sum
        self.stop_requests = stop_requests
        self.scale = scale
        self.strength = strength
        self.grad_norm = grad_norm
        return update_tokens

    def cadence(self):
        training, state = self.training, self.state
        stopped = self.stop_requests > 0 or (self.stop_after_updates is not None and state["updates"] >= self.stop_after_updates)
        complete = state["tokens_seen"] == self.total_tokens
        if training.async_checkpoint and (state["updates"] % training.log_interval == 0 or stopped or complete):
            poll_checkpoint_writer(self.checkpoint_writer, self.rank, self.world_size)
        if self.sampled:
            append_record(self.output_dir / "stability.jsonl", {"event": "stability", **state, **self.monitor.finish()})
        if self.rank == 0 and (state["updates"] % training.log_interval == 0 or stopped or complete):
            now = time.monotonic()
            append_event(self.output_dir / "events.jsonl", {"event": "train", **state, "loss": self.loss_sum / self.update_tokens,
                         "loss_tokens": self.update_tokens, "valid_targets": self.valid_targets,
                         "data_wait_seconds": self.data_wait_seconds,
                         "grad_norm_before_clip": float(self.grad_norm), "lr_scale": self.scale,
                         "grad_clip_coefficient": min(1.0, training.grad_clip / (float(self.grad_norm) + 1e-6)) if training.grad_clip else 1.0,
                         "optimizer_group_lr_min": min(group["lr"] for group in self.optimizer.param_groups),
                         "optimizer_group_lr_max": max(group["lr"] for group in self.optimizer.param_groups),
                         "quantization_strength": self.strength, "tokens_per_second": (state["tokens_seen"] - self.last_log_tokens) / (now - self.last_log),
                         "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                         "allocated_memory_bytes": torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else None,
                         "reserved_memory_bytes": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else None})
            self.last_log, self.last_log_tokens = now, state["tokens_seen"]
        if state["updates"] % training.checkpoint_interval == 0 or stopped or complete:
            collective_checkpoint(self.output_dir / "checkpoints", self.model, self.optimizer, state, self.contract, self.rank, self.world_size,
                                  writer=self.checkpoint_writer, wait=stopped or complete)
        if (not self.skip_validation) and (state["updates"] % training.validation_interval == 0 or stopped or complete):
            results = evaluate(self.model, self.manifest, self.data_root, self.device, self.amp_dtype,
                               training.validation_batches_per_language, self.rank, self.world_size,
                               sequences_per_language=training.validation_sequences_per_language,
                               verified_files=self.verified_files, progress_callback=self.evaluation_progress)
            mixture = results.pop("mixture", None)
            benchmarks = None
            if self.choice_records:
                from .task_eval import score_frozen_choices

                benchmarks = score_frozen_choices(self.model, self.choice_records, device=self.device,
                                                  max_seq_len=self.model_config.max_seq_len, rank=self.rank,
                                                  world_size=self.world_size, amp_dtype=self.amp_dtype,
                                                  progress_callback=self.evaluation_progress)
            if self.rank == 0:
                payload = {"event": "validation", **state, "languages": results}
                if mixture is not None:
                    payload["mixture"] = mixture
                if benchmarks is not None:
                    payload["benchmarks"] = benchmarks
                append_event(self.output_dir / "events.jsonl", payload)
        if stopped or complete:
            if self.rank == 0:
                append_event(self.output_dir / "events.jsonl", {"event": "complete" if complete else "stopped", **state,
                             "tokens_this_invocation": state["tokens_seen"] - self.tokens_at_start})
            return True
        return False


@contextlib.contextmanager
def prepare_training(config_path, manifest_path, data_root, output_dir, *, resume=None,
                     preflight_report=None, operational_overrides=None,
                     research_timing_only=False, staging_receipt=None):
    model_config, training, optimizer_config = load_config(config_path)
    if operational_overrides is not None:
        unknown = set(operational_overrides) - set(OPERATIONAL_TRAINING_FIELDS)
        if unknown:
            raise ValueError("not operational training fields: " + ", ".join(sorted(unknown)))
        training = replace(training, **operational_overrides)
    world_size, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world size")
    if training.cuda_graphs and world_size != 1:
        raise ValueError("captured microsteps currently require one GPU; NCCL graph capture needs a separate implementation")
    if training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(local_rank)
        if training.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on the selected GPU")
    device = torch.device("cuda", local_rank) if training.device == "cuda" else torch.device("cpu")
    if not training.development_mode and model_config.gradient_checkpointing:
        raise ValueError("production retains activations; gradient checkpointing is off")
    if not training.development_mode and (device.type != "cuda" or training.dtype != "bf16" or model_config.attention_backend == "sdpa_reference"):
        raise ValueError("production requires BF16 autocast and an explicit CUDA attention backend")
    if research_timing_only and (world_size != 1 or device.type != "cuda"):
        raise ValueError("research timing without native preflight requires one CUDA GPU")
    if not training.development_mode and not research_timing_only:
        from .preflight import validate_preflight_report

        validate_preflight_report(preflight_report, model_config, device, cuda_graphs=training.cuda_graphs)
        if training.gemm_table:
            from production.amd.gemm_dispatch import install_table

            install_table(json.loads(Path(training.gemm_table).read_text()))
    if world_size > 1:
        distributed.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=5))
    workload = PreparedTraining()
    lock_handle = None
    iterator = None
    monitor = None
    old_handlers = {}
    try:
        manifest = TokenManifest.load(manifest_path)
        if staging_receipt is None:
            manifest_report = validate_manifest(manifest, data_root, model_config, development_mode=training.development_mode)
        else:
            from production.amd.staging import verify as verify_staging

            manifest_report = verify_staging(json.loads(Path(staging_receipt).read_text()), manifest,
                                             data_root, model_config, development_mode=training.development_mode)
        total_tokens = manifest.scored_train_tokens
        if training.warmup_tokens >= total_tokens:
            raise ValueError("warmup token budget must be shorter than the total budget")
        if training.quantization_warmup_tokens > total_tokens:
            raise ValueError("quantization warmup exceeds the total token budget")
        if not training.development_mode and model_config.weight_mode == "ternary" and model_config.quantization_strength != 1:
            raise ValueError("production must finish with fully ternary projections")
        if model_config.weight_mode == "dense" and (training.quantization_warmup_tokens or training.quantization_start_strength != 1):
            raise ValueError("the dense-weight stage has no quantization curriculum")
        if model_config.linear_backend == "deepgemm" and (training.quantization_warmup_tokens or training.quantization_start_strength != 1):
            raise ValueError("native DeepGEMM is not validated for a quantization curriculum")
        contract = make_resume_contract(model_config, training, optimizer_config, manifest_report, world_size, device)
        if world_size > 1:
            contracts = [None] * world_size
            distributed.all_gather_object(contracts, fingerprint(contract))
            if len(set(contracts)) != 1:
                raise ValueError("ranks disagree on data, source, configuration, or runtime identity")
        output_dir = Path(output_dir)
        setup_error = [None]
        if rank == 0:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                lock_handle = (output_dir / ".run.lock").open("a")
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_file = output_dir / "run.json"
                if resume:
                    if run_file.exists() and json.loads(run_file.read_text())["contract_sha256"] != fingerprint(contract):
                        raise ValueError("output directory belongs to an incompatible run")
                elif run_file.exists() or (output_dir / "events.jsonl").exists() or (output_dir / "checkpoints").exists():
                    raise FileExistsError("choose a fresh output directory or resume explicitly")
                write_json_atomic(run_file, {"contract_sha256": fingerprint(contract), "contract": contract})
            except Exception as exc:
                setup_error[0] = f"{type(exc).__name__}: {exc}"
        if world_size > 1:
            distributed.broadcast_object_list(setup_error, src=0)
        if setup_error[0]:
            raise RuntimeError(setup_error[0])
        random.seed(training.seed)
        np.random.seed(training.seed)
        torch.manual_seed(training.seed)
        model = BitNetLlama(model_config).to(device)
        if training.stability_log_interval and rank == 0:
            if training.cuda_graphs or training.compile:
                raise ValueError("stability probes require eager microsteps")
            from .stability import StabilityMonitor

            monitor = StabilityMonitor(model)
        optimizer = build_optimizer(model, optimizer_config)
        state = {"next_micro_step": 0, "tokens_seen": 0, "updates": 0}
        payload = load_checkpoint(resume, contract) if resume else None
        if payload is not None:
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            state = payload["state"]
        amp_dtype = torch.bfloat16 if training.dtype == "bf16" else torch.float32
        forward_model = torch.compile(model) if training.compile else model
        if world_size > 1:
            forward_model = DistributedDataParallel(forward_model, device_ids=[local_rank] if device.type == "cuda" else None,
                                                    broadcast_buffers=False, gradient_as_bucket_view=True,
                                                    static_graph=True, bucket_cap_mb=training.ddp_bucket_cap_mb)
            if os.environ.get("CI1_GRAD_BF16") == "1":
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook

                # Candidate-only wire compression: one BF16 rounding per bucket
                # per update; Muon NS is scale-invariant and AdamW is relative.
                forward_model.register_comm_hook(state=None, hook=bf16_compress_hook)
        if payload is not None:
            if len(payload["rank_rng"]) != world_size:
                raise ValueError("checkpoint rank RNG states do not match world size")
            restore_rng_state(payload["rank_rng"][rank])
            del payload
        else:
            torch.manual_seed(training.seed + rank)
        micro_tokens = manifest.config.tokens_per_step * world_size
        total_micro_steps = math.ceil(manifest.config.total_examples / (manifest.config.batch_size * world_size))
        packed_batcher = None
        if manifest.is_packed:
            from .packed_data import PackedTokenBatcher

            if training.compile:
                raise ValueError("packed document metadata requires eager training; compile qualification is separate")
            packed_batcher = PackedTokenBatcher(manifest, root=data_root, rank=rank, world_size=world_size,
                                               verified_files=manifest_report["packed_files"])
            tokens_before = packed_batcher.spec.get("tokens_before", 0)
            if tokens_before and (not resume or training.phase_start_tokens != tokens_before):
                raise ValueError("packed lifetime offset requires the matching transition checkpoint/schedule")
            expected_slots = min(state["next_micro_step"]*micro_tokens, manifest.config.usable_train_tokens)
            if state["next_micro_step"] == 0:
                state.setdefault("input_slots_seen", 0)
                state.setdefault("content_tokens_seen", 0)
            if (state.get("input_slots_seen") != expected_slots or
                    not 0 <= state["tokens_seen"] - tokens_before <= expected_slots or
                    not 0 <= state.get("content_tokens_seen", -1) <= expected_slots):
                raise ValueError("packed checkpoint slot/label/content counters disagree")
        elif state["tokens_seen"] != min(state["next_micro_step"] * micro_tokens, total_tokens):
            raise ValueError("checkpoint token and data-cursor counters disagree")
        if state["next_micro_step"] != min(state["updates"] * training.gradient_accumulation_steps, total_micro_steps):
            raise ValueError("checkpoint update and accumulation counters disagree")
        phase_offset = training.phase_start_tokens or 0
        if phase_offset and not resume:
            raise ValueError("a nonzero phase_start_tokens requires resuming from a transition checkpoint")
        if phase_offset >= total_tokens:
            raise ValueError("phase_start_tokens exceeds the manifest token budget")
        if state["tokens_seen"] < phase_offset:
            raise ValueError("checkpoint cursor predates the phase start")
        if state["tokens_seen"] >= total_tokens:
            raise ValueError("training budget is already complete")
        iterator = iter(PrefetchingTokenLoader(manifest, root=data_root, rank=rank, world_size=world_size,
                        start_step=state["next_micro_step"], max_steps=total_micro_steps - state["next_micro_step"],
                        prefetch_batches=training.prefetch_batches, device=device, return_step=True, batcher=packed_batcher))
        should_stop = [False]

        def request_stop(signum, frame):
            should_stop[0] = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        started = last_log = time.monotonic()
        tokens_at_start = last_log_tokens = state["tokens_seen"]
        if rank == 0:
            append_event(output_dir / "events.jsonl", {"event": "startup", "counts": model.count_parameters(),
                         "world_size": world_size, "state": state, "unused_tail_tokens": manifest.config.total_train_tokens - manifest.config.usable_train_tokens,
                         "token_clock": "scored_targets", "input_slot_horizon": manifest.config.usable_train_tokens})
        model.train()
        workload.model_config = model_config
        workload.training = training
        workload.optimizer_config = optimizer_config
        workload.world_size = world_size
        workload.rank = rank
        workload.device = device
        workload.manifest = manifest
        workload.data_root = data_root
        workload.total_tokens = total_tokens
        workload.contract = contract
        workload.output_dir = output_dir
        workload.model = model
        workload.monitor = monitor
        workload.optimizer = optimizer
        if training.shard_muon and world_size > 1:
            from production.amd.sharded_muon import install_owned_muon, muon_parameter_owners

            workload.muon_owners = install_owned_muon(optimizer, muon_parameter_owners(optimizer, world_size), rank)
        workload.state = state
        workload.amp_dtype = amp_dtype
        workload.forward_model = forward_model
        workload.micro_tokens = micro_tokens
        workload.total_micro_steps = total_micro_steps
        workload.phase_offset = phase_offset
        workload.iterator = iterator
        workload.packed_batcher = packed_batcher
        workload.verified_files = manifest_report.get("packed_files")
        if training.benchmark_tokens:
            from .task_eval import load_development_choices

            workload.choice_records = load_development_choices(
                training.benchmark_tokens, expected_tokenizer_sha256=manifest.tokenizer["sha256"])
        workload.should_stop = should_stop
        workload.started = started
        workload.last_log = last_log
        workload.tokens_at_start = tokens_at_start
        workload.last_log_tokens = last_log_tokens
        if training.async_checkpoint and rank == 0:
            from .async_checkpoint import AsyncCheckpointWriter
            workload.checkpoint_writer = AsyncCheckpointWriter(output_dir / "checkpoints",
                                                              timeout=training.checkpoint_timeout_seconds)
        yield workload
        if training.async_checkpoint:
            poll_checkpoint_writer(workload.checkpoint_writer, rank, world_size, wait=True)
    except Exception as exc:
        candidate = Path(output_dir)
        if rank == 0 and (candidate / "run.json").exists():
            try:
                append_record(candidate / "incidents.jsonl", {
                    "event": "run_failure", "state": getattr(workload, "state", None),
                    "error_type": type(exc).__name__, "error": str(exc),
                    "cuda_oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                })
            except Exception:
                pass  # Preserve the original failure if the output disk is also unhealthy.
        raise
    finally:
        original_error = sys.exc_info()[0]
        close_error = None
        try:
            if workload.checkpoint_writer is not None:
                workload.checkpoint_writer.close()
        except Exception as exc:
            close_error = exc
        if monitor is not None:
            monitor.close()
        if iterator is not None:
            iterator.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if lock_handle is not None:
            lock_handle.close()
        if world_size > 1 and distributed.is_initialized():
            distributed.destroy_process_group()
        if close_error is not None and original_error is None:
            raise close_error


def run_training(config_path, manifest_path, data_root, output_dir, *, resume=None, stop_after_updates=None, preflight_report=None):
    if stop_after_updates is not None and stop_after_updates <= 0:
        raise ValueError("stop_after_updates must be positive")
    with prepare_training(config_path, manifest_path, data_root, output_dir,
                          resume=resume, preflight_report=preflight_report) as workload:
        workload.stop_after_updates = stop_after_updates
        while workload.state["next_micro_step"] < workload.total_micro_steps:
            workload.complete_update()
            if workload.cadence():
                break
        return workload.state


def main():
    parser = argparse.ArgumentParser(description="Token-correct single-device/DDP BitNet pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--preflight-report")
    args = parser.parse_args()
    run_training(args.config, args.manifest, args.data_root, args.output_dir,
                 resume=args.resume, stop_after_updates=args.stop_after_updates, preflight_report=args.preflight_report)


if __name__ == "__main__":
    main()
