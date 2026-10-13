"""Inspect a selected dense run on CPU; optionally verify all local data bytes.

This never starts training and cannot certify native kernels, cluster throughput,
memory fit, recovery, or quality. JSON output must live outside the repository.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from production.recipe import SELECTED_MODEL, SELECTED_OPTIMIZER, SELECTED_TRAINING
from terminus.data import TokenManifest
from terminus.integrity import fingerprint, validate_manifest
from terminus.model import BitNetLlamaConfig
from terminus.optim import OptimizerConfig
from terminus.preflight import count_model
from terminus.train import load_config, schedule_scale


def inspect_run(model, training, optimizer, manifest, *, world_size,
                expected_global_batch_tokens, data_root=None):
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if model != BitNetLlamaConfig(**SELECTED_MODEL):
        raise ValueError("model differs from the selected dense baseline; review the change first")
    # JSON loads optimizer betas as a list, while dataclass defaults use a tuple.
    if fingerprint(asdict(optimizer)) != fingerprint(asdict(OptimizerConfig(**SELECTED_OPTIMIZER))):
        raise ValueError("optimizer differs from the selected dense baseline; review the change first")
    if any(getattr(training, key) != value for key, value in SELECTED_TRAINING.items()):
        raise ValueError("training semantics differ from the selected dense baseline")
    if (training.cuda_graphs or training.quantization_warmup_tokens
            or training.quantization_start_strength != 1 or training.phase_start_tokens):
        raise ValueError("selected dense run requires eager execution without a QAT/phase transition")
    if manifest.config.seq_len != model.max_seq_len:
        raise ValueError("manifest sequence length differs from model context")
    microstep_tokens = world_size * manifest.config.tokens_per_step
    batch_tokens = microstep_tokens * training.gradient_accumulation_steps
    if type(expected_global_batch_tokens) is not int or batch_tokens != expected_global_batch_tokens:
        raise ValueError("global batch does not match the explicitly expected batch at this world size")
    total_tokens = manifest.config.usable_train_tokens
    schedule_scale(0, total_tokens, training.warmup_tokens, training.final_lr_fraction,
                   schedule=training.schedule, decay_tokens=training.decay_tokens,
                   decay_style=training.decay_style)
    counts = count_model(model)
    full_updates, tail_tokens = divmod(total_tokens, batch_tokens)
    parameters = counts["total"]
    data_report = (validate_manifest(manifest, data_root, model, development_mode=False)
                   if data_root is not None else None)
    return {
        "status": "offline_checks_passed",
        "scope": "CPU configuration/arithmetic checks only; not a production launch gate",
        "identity": {"model_sha256": fingerprint(asdict(model)),
                     "training_sha256": fingerprint(asdict(training)),
                     "optimizer_sha256": fingerprint(asdict(optimizer)),
                     "manifest_sha256": fingerprint(asdict(manifest))},
        "parameters": parameters,
        "parallelism": {"strategy": "replicated_ddp", "world_size": world_size,
                        "microbatch_sequences_per_rank": manifest.config.batch_size,
                        "accumulation_steps": training.gradient_accumulation_steps,
                        "global_batch_tokens": batch_tokens,
                        "tokens_per_rank_per_full_update": batch_tokens // world_size,
                        "fp32_gradient_bytes_per_rank": 4 * parameters,
                        "ring_allreduce_bytes_per_rank_per_full_update":
                            2 * (world_size - 1) / world_size * 4 * parameters},
        "schedule": {"usable_train_tokens": total_tokens,
                     "unused_tail_tokens": manifest.config.total_train_tokens - total_tokens,
                     "full_updates": full_updates,
                     "partial_final_update_tokens": tail_tokens,
                     "total_updates": full_updates + bool(tail_tokens),
                     "warmup_tokens": training.warmup_tokens,
                     "decay_start_tokens": total_tokens - training.decay_tokens,
                     "checkpoint_interval_updates": training.checkpoint_interval,
                     "checkpoint_interval_full_update_tokens": training.checkpoint_interval * batch_tokens,
                     "validation_interval_updates": training.validation_interval},
        "data": {"status": "verified" if data_report is not None else "not_checked",
                 "verification": data_report},
        "unverified": [*(["data bytes, tokenizer, provenance and source budgets"] if data_report is None else []),
                       "native H100 forward/backward and optimizer parity",
                       "cluster topology, memory and all-rank throughput",
                       "checkpoint durability and same-world-size restart on the cluster",
                       "held-out quality, dense-to-QAT transfer and aggregate GPU-hour budget"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--expected-global-batch-tokens", type=int, required=True)
    parser.add_argument("--data-root", type=Path, help="Read and validate every shard; may take hours on a full corpus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        parser.error("put production reports outside this repository")
    if output.exists():
        parser.error(f"output already exists: {output}")
    result = inspect_run(*load_config(args.config), TokenManifest.load(args.manifest),
                         world_size=args.world_size,
                         expected_global_batch_tokens=args.expected_global_batch_tokens,
                         data_root=args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"], "scope": result["scope"]}))


if __name__ == "__main__":
    main()
