"""Human-invoked AMD training/measurement entry point using the shared trainer."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from production.artifacts import external_path
from terminus.data import TokenManifest
from terminus.integrity import fingerprint, write_json_atomic
from terminus.train import load_config, prepare_training
from .recipe import validate_production
from .runtime import capture_runtime, verify, require_gfx942
from .staging import verify as verify_staging
from .measurement import summarize, validate_window


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "data-root", "output-dir", "runtime-report", "preflight-report", "staging-receipt"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--expected-world-size", type=int, required=True)
    p.add_argument("--expected-global-batch-tokens", type=int, required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after-updates", type=int)
    p.add_argument("--benchmark-updates", type=int, default=0)
    p.add_argument("--warmup-updates", type=int, default=5)
    args = p.parse_args()
    if args.benchmark_updates < 0 or args.warmup_updates < 0 or (args.stop_after_updates is not None and args.stop_after_updates <= 0):
        p.error("invalid update counts")
    if args.benchmark_updates and args.stop_after_updates:
        p.error("benchmark uses its own update bound")
    for key in ("config", "manifest", "data_root", "output_dir", "runtime_report", "preflight_report", "staging_receipt"):
        setattr(args, key, external_path(getattr(args, key)))
    model, training, optimizer = load_config(args.config)
    if model.weight_mode == "ternary" and not args.resume:
        p.error("QAT requires --resume from a compatible transition checkpoint")
    if args.benchmark_updates and args.resume and model.weight_mode != "ternary":
        p.error("a resumed benchmark is only supported for the QAT transition phase")
    candidate = validate_production(model, training, optimizer)
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world != args.expected_world_size:
        raise ValueError("launcher world size differs from the inspected run")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda", torch.cuda.current_device())
    require_gfx942(device)
    manifest = TokenManifest.load(args.manifest)
    batch = world * manifest.config.tokens_per_step * training.gradient_accumulation_steps
    if batch != args.expected_global_batch_tokens:
        raise ValueError("global batch changed; retune accumulation or regenerate the config")
    # Do bounded snapshot checks before opening collectives; torchrun propagates
    # a worker failure to the other workers instead of launching a partial run.
    verify_staging(json.loads(args.staging_receipt.read_text()), manifest, args.data_root, model)
    verify(json.loads(args.runtime_report.read_text()), capture_runtime(str(device), include_inventory=False))
    with prepare_training(args.config, args.manifest, args.data_root, args.output_dir,
                          resume=args.resume, preflight_report=args.preflight_report,
                          staging_receipt=args.staging_receipt) as workload:
        if args.benchmark_updates:
            validate_window(total_tokens=workload.total_tokens, tokens_seen=workload.state["tokens_seen"],
                            batch_tokens=batch, warmup_updates=args.warmup_updates,
                            measured_updates=args.benchmark_updates)
        workload.stop_after_updates = args.stop_after_updates
        if args.stop_after_updates is not None and args.stop_after_updates <= workload.state["updates"]:
            raise ValueError("stop-after-updates must exceed the resumed update count")
        if not args.benchmark_updates:
            while workload.state["next_micro_step"] < workload.total_micro_steps:
                workload.complete_update()
                if workload.cadence():
                    break
            return
        warm, records = args.warmup_updates, []
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(warm + args.benchmark_updates):
            if workload.should_stop[0]:
                raise RuntimeError("benchmark interrupted; no complete measurement receipt")
            if index == warm and world > 1:
                dist.barrier()
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            tokens = workload.complete_update()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - begin
            if workload.stop_requests:
                raise RuntimeError("benchmark reached a stop/time-limit request; no complete receipt")
            if index >= warm:
                records.append({"seconds": elapsed, "global_tokens": tokens, "loss": workload.loss_sum / tokens})
        local = {"rank": rank, "records": records, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        gathered = [None] * world if world > 1 else [local]
        if world > 1:
            dist.all_gather_object(gathered, local)
        if rank == 0:
            result = summarize(gathered, world_size=world)
            result.update(candidate=candidate, weight_mode=model.weight_mode,
                          warmup_updates=warm, contract=workload.contract,
                          training=asdict(training),
                          global_batch_tokens=batch, microbatch=manifest.config.batch_size,
                          accumulation=training.gradient_accumulation_steps,
                          scope="Real data + complete optimizer updates. Timed window excludes scheduled checkpoint/evaluation cadence.")
            result["report_sha256"] = fingerprint(result)
            write_json_atomic(Path(args.output_dir) / "benchmark.json", result)
            print(json.dumps({k: v for k, v in result.items() if k not in {"ranks", "contract"}}, indent=2))
        # Exercise final checkpoint + validation separately and record their cost.
        workload.stop_after_updates = workload.state["updates"]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        workload.cadence()
        torch.cuda.synchronize(device)
        timing = torch.tensor(time.perf_counter() - start, device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        if rank == 0:
            write_json_atomic(Path(args.output_dir) / "cadence.json",
                              {"checkpoint_plus_validation_seconds": timing.item(), "updates": workload.state["updates"]})


if __name__ == "__main__":
    main()
