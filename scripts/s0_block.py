#!/usr/bin/env python3
"""S0 measured full-update block driver for the stack-004 speed campaign.

Runs a predeclared warmup and then wraps
speed_research.measurement.measure_block around complete trainer updates
(see docs/speed/MEASUREMENT.md section 2) and exports the raw block record
outside Git. Measured-block tokens are a throughput measurement, never
production corpus exposure. Billed GPU/VM-on time is recorded by the
external supervisor, not this script. Device-wide peak memory (including
non-torch allocations) must accompany native runs from external telemetry;
the torch allocator peak recorded here is the lower bound.

Usage: .venv/bin/python scripts/s0_block.py --config CONFIG --manifest MANIFEST \
  --data-root ROOT --output-dir RUN --warmup-updates N --measured-updates N \
  --out /external/path/block.json [--resume CHECKPOINTS] [--label LABEL]
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from speed_research.measurement import measure_block
from terminus.integrity import file_sha256, fingerprint, source_hashes, write_json_atomic
from terminus.train import collective_checkpoint, load_config, prepare_training, runtime_contract

CADENCE_FREE = {"log_interval": 10**9, "checkpoint_interval": 10**9,
                "validation_interval": 10**9, "stability_log_interval": 0}


def device_synchronize(workload):
    if workload.device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def device_peak_reader(workload):
    if workload.device.type == "cuda":
        return lambda: torch.cuda.max_memory_allocated(workload.device), "torch.cuda.max_memory_allocated"
    return lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "ru_maxrss (CPU development reference only)"


def incident_count(output_dir):
    path = Path(output_dir) / "incidents.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--preflight-report")
    parser.add_argument("--warmup-updates", type=int, required=True)
    parser.add_argument("--measured-updates", type=int, required=True)
    parser.add_argument("--profile-updates", type=int, default=0,
                        help="extra predeclared updates under torch.profiler between warmup and the measured block")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="opt in to a full model/optimizer checkpoint after the block")
    parser.add_argument("--label", default="full_update")
    parser.add_argument("--out", required=True, help="raw block record path; must be outside Git")
    args = parser.parse_args()
    if args.warmup_updates < 0 or args.profile_updates < 0 or args.measured_updates <= 0:
        raise ValueError("warmup/profile must be nonnegative and measured updates positive")
    out_path = Path(args.out)
    repo = Path(__file__).resolve().parent.parent
    if out_path.resolve().is_relative_to(repo):
        raise ValueError("the raw block record must be written outside the Git checkout")

    _, training_config, _ = load_config(args.config)
    quick_native_timing = (training_config.device == "cuda" and not training_config.development_mode
                           and args.preflight_report is None)
    with prepare_training(args.config, args.manifest, args.data_root, args.output_dir,
                          resume=args.resume, preflight_report=args.preflight_report,
                          operational_overrides=CADENCE_FREE,
                          research_timing_only=quick_native_timing) as workload:
        state = workload.state
        needed = (args.warmup_updates + args.profile_updates + args.measured_updates) * workload.training.gradient_accumulation_steps
        remaining = workload.total_micro_steps - state["next_micro_step"]
        if remaining < needed:
            raise ValueError(f"remaining budget {remaining} micro steps cannot cover "
                             f"{args.warmup_updates}+{args.profile_updates}+{args.measured_updates} updates ({needed})")
        for _ in range(args.warmup_updates):
            workload.complete_update()
            if workload.cadence():
                raise RuntimeError("training budget exhausted or stopped during warmup")
        trace_path = None
        if args.profile_updates:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if workload.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            trace_path = workload.output_dir / "s0_trace.json.gz"
            with torch.profiler.profile(activities=activities) as profiler:
                for _ in range(args.profile_updates):
                    workload.complete_update()
                if workload.cadence():
                    raise RuntimeError("training budget exhausted or stopped during profiling")
            profiler.export_chrome_trace(str(trace_path))
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        incidents_before = incident_count(workload.output_dir)
        updates_before, tokens_before = workload.state["updates"], workload.state["tokens_seen"]
        if workload.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(workload.device)
        read_memory, memory_source = device_peak_reader(workload)
        raw = measure_block(workload.complete_update, updates=args.measured_updates,
                            synchronize=device_synchronize(workload), read_memory=read_memory)
        finished = time.time()
        state = workload.state
        if state["updates"] != updates_before + args.measured_updates:
            raise RuntimeError("update counters did not advance by exactly the measured block length")
        if state["tokens_seen"] - tokens_before != raw["valid_tokens"]:
            raise RuntimeError("valid target count disagrees with committed trainer counters")
        if incident_count(workload.output_dir) != incidents_before:
            raise RuntimeError("incidents were recorded during the measured block; refusing a finite attestation")
        checkpoint = None
        if args.save_checkpoint:
            collective_checkpoint(workload.output_dir / "checkpoints", workload.model, workload.optimizer,
                                  state, workload.contract, workload.rank, workload.world_size)
            receipt = json.loads((workload.output_dir / "checkpoints/latest.json").read_text())
            checkpoint = {"file": receipt["file"], "sha256": receipt["sha256"]}
        record = {
            "schema": 1,
            "kind": "s0_raw_block",
            "run_id": uuid.uuid4().hex,
            "label": args.label,
            "commit": commit,
            "config_sha256": file_sha256(args.config),
            "manifest_sha256": file_sha256(args.manifest),
            "contract_sha256": fingerprint(workload.contract),
            "source_hashes": source_hashes(),
            "runtime": runtime_contract(workload.device),
            "resume_from": str(args.resume) if args.resume else None,
            "profile_trace": str(trace_path) if trace_path else None,
            "profile_updates": args.profile_updates,
            "warmup_updates": args.warmup_updates,
            "block": {**raw, "finite": True},
            "counters_before": {"updates": updates_before, "tokens_seen": tokens_before},
            "counters_after": dict(state),
            "final_checkpoint": checkpoint,
            "memory_source": memory_source,
            "native_preflight": ("validated" if args.preflight_report else
                                 "omitted_for_research_timing" if quick_native_timing else
                                 "development_reference"),
            "finished_unix": finished,
            "note": "throughput measurement only; billed GPU/VM time is recorded by the external supervisor",
        }
        write_json_atomic(out_path, record)
        print(json.dumps({"run_id": record["run_id"], "out": str(out_path),
                          "seconds": raw["seconds"], "valid_tokens": raw["valid_tokens"],
                          "tokens_per_second": raw["valid_tokens"] / raw["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
