"""Bounded qualification runner for the human-launched H100 campaign.

Reuses `terminus.train` update semantics through `prepare_training`, wrapped
in the campaign gates: frozen config/matrix validation, token accounting,
source and receipt freshness, allocation-ledger stage limits, per-update
full-state observations, boundary checkpoints/validations, review alarms
that stop at valid boundaries, and durable backup enqueueing. This runner
never provisions a VM, never resets the campaign clock, never silently
repairs a failed arm, and never counts probes as candidate progress.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import time

from production.artifacts import external_path

from . import branches, budget, campaign, horizons, telemetry
from .builder import validate_qualification_config
from .recovery import BackupQueue, backup_health


def expected_group_structure(model_config) -> dict:
    kv = model_config.n_kv_heads * model_config.head_dim
    return {
        "muon_groups": 4 * model_config.n_layers,
        "qkv_rows": (model_config.dim, kv, kv),
        "gate_up_rows": (model_config.ffn_dim, model_config.ffn_dim),
        "adam_labels": {"tied_embedding", "input_norm", "peri_norm",
                        "final_norm", "value_mix", "qk_norm"},
    }


def assert_effective_groups(model_config, optimizer, *, expected_optimizer: dict) -> dict:
    """Assert the constructed optimizer groups, not just the input JSON.

    Defaults and packed logical shapes are part of the contract: packed QKV
    must split into Q/K/V and packed gate/up into two logical matrices, all
    Muon groups must use spectral_unclamped, and the AdamW-side labels and
    rates must match the frozen settings.
    """
    structure = expected_group_structure(model_config)
    muon = [group for group in optimizer.param_groups if group.get("use_muon")]
    adam = [group for group in optimizer.param_groups if not group.get("use_muon")]
    if len(muon) != structure["muon_groups"]:
        raise ValueError("effective Muon group count differs from the packed contract")
    counts = {"qkv": 0, "o": 0, "gate_up": 0, "down": 0}
    for group in muon:
        name = group["name"]
        if name.endswith(".attn.qkv_proj.weight"):
            if tuple(group.get("logical_rows") or ()) != structure["qkv_rows"]:
                raise ValueError("packed QKV logical rows differ from the contract")
            counts["qkv"] += 1
        elif name.endswith(".ffn.gate_up_proj.weight"):
            if tuple(group.get("logical_rows") or ()) != structure["gate_up_rows"]:
                raise ValueError("packed gate/up logical rows differ from the contract")
            counts["gate_up"] += 1
        elif name.endswith(".attn.o_proj.weight"):
            if group.get("logical_rows"):
                raise ValueError("O projection must be a single logical matrix")
            counts["o"] += 1
        elif name.endswith(".ffn.down_proj.weight"):
            if group.get("logical_rows"):
                raise ValueError("down projection must be a single logical matrix")
            counts["down"] += 1
        else:
            raise ValueError(f"unexpected Muon matrix group: {name}")
        if group.get("adjust_lr_fn", "spectral_unclamped") != "spectral_unclamped":
            raise ValueError("effective Muon scaling is not spectral_unclamped")
        if group["base_lr"] != expected_optimizer["muon_lr"]:
            raise ValueError("effective Muon base LR differs from the trial matrix")
        if group["weight_decay"] != expected_optimizer["weight_decay"]:
            raise ValueError("effective matrix decay differs from the fixed 0.02")
    if counts != {"qkv": model_config.n_layers, "o": model_config.n_layers,
                  "gate_up": model_config.n_layers, "down": model_config.n_layers}:
        raise ValueError(f"effective Muon role counts differ from the contract: {counts}")
    labels = {}
    for group in adam:
        labels.setdefault(group["name"], []).append(group)
    if set(labels) != structure["adam_labels"]:
        raise ValueError(f"effective AdamW group labels differ from the contract: {sorted(labels)}")
    expected_rates = {"tied_embedding": expected_optimizer["adamw_lr"],
                      "input_norm": expected_optimizer["norm_lr"],
                      "peri_norm": expected_optimizer["norm_lr"],
                      "final_norm": expected_optimizer["norm_lr"],
                      "value_mix": expected_optimizer["norm_lr"],
                      "qk_norm": expected_optimizer["qk_norm_lr"]}
    for label, groups in labels.items():
        for group in groups:
            if group["base_lr"] != expected_rates[label]:
                raise ValueError(f"effective {label} base LR differs from the frozen settings")
            expected_wd = (expected_optimizer["embedding_weight_decay"]
                           if label == "tied_embedding" else 0.0)
            if group["weight_decay"] != expected_wd:
                raise ValueError(f"effective {label} decay differs from the frozen settings")
    return {"muon_groups": len(muon), "role_counts": counts, "adam_labels": sorted(labels)}


def gate_offline(config_path, manifest_path, *, trial_id: str, confirmation: bool,
                 ledger_path=None, stage_forecast_hours: float | None = None,
                 source_receipt_path=None, now=None) -> dict:
    """Fail-closed offline gates; native runtime/data/preflight gates follow on the H100."""
    receipt = validate_qualification_config(config_path, manifest_path, trial_id=trial_id,
                                            confirmation=confirmation)
    from terminus.data import TokenManifest

    receipt["token_accounting"] = horizons.audit_token_accounting(
        TokenManifest.load(external_path(manifest_path)))
    if source_receipt_path is not None:
        recorded = json.loads(Path(external_path(source_receipt_path)).read_text())
        current = campaign.qualification_source_hashes()
        if recorded.get("source") != current:
            changed = sorted(k for k in set(recorded.get("source", {})) | set(current)
                             if recorded.get("source", {}).get(k) != current.get(k))
            raise ValueError("trainer/qualification source changed since the packet froze: "
                             + ", ".join(changed[:8]))
        receipt["source_receipt"] = "verified"
    if ledger_path is not None and stage_forecast_hours is not None:
        ledger = budget.Ledger(external_path(ledger_path))
        label = f"{'confirmation' if confirmation else 'trial'} {trial_id}"
        receipt["budget"] = ledger.require(stage_forecast_hours, label=label, now=now)
    return receipt


def _checkpoints_pending_upload(checkpoint_dir) -> list:
    pending = []
    for payload in sorted(Path(checkpoint_dir).glob("step_*.pt")):
        marker = payload.with_suffix(".uploaded.json")
        if not marker.exists():
            pending.append(payload.name)
    return pending


def drive(workload, *, boundaries: tuple, ledger_path, final_update: int,
          output_dir: Path, reference_tolerances: dict | None = None,
          backup: BackupQueue | None = None, backup_bounds: dict | None = None,
          emergency_reserve_hours: float = campaign.EMERGENCY_RESERVE_HOURS,
          clock=time.monotonic) -> dict:
    """The bounded trial loop around complete_update()/cadence()."""
    from terminus.train import append_event, append_record, collective_checkpoint, evaluate

    tracker = telemetry.RoleDeltaTracker(workload.optimizer)
    qualification_log = Path(output_dir) / "qualification.jsonl"
    warmup_updates = workload.training.warmup_tokens // (
        workload.training.gradient_accumulation_steps * workload.micro_tokens)
    probe_updates = ({1, warmup_updates + 1, sorted(boundaries)[-2] + 1}
                     if reference_tolerances else set())
    clip_window = deque(maxlen=100)
    alarm = None
    ewma_seconds = None
    summary = {"updates_completed": 0, "alarms": [], "reference_replays": []}

    while workload.state["updates"] < final_update:
        next_update = workload.state["updates"] + 1
        diagnostic = (next_update % campaign.ROLE_DIAGNOSTIC_INTERVAL == 0
                      or next_update in boundaries)
        if diagnostic:
            tracker.begin()
        started = clock()
        if next_update in probe_updates:
            replay = branches.reference_replay(workload)
            branches.require_within_tolerance(
                replay, max_weight_delta=reference_tolerances["max_weight_delta"],
                max_moment_delta=reference_tolerances["max_moment_delta"])
            summary["reference_replays"].append(replay)
        else:
            workload.complete_update()
        update_seconds = clock() - started
        if workload.valid_targets != workload.update_tokens:
            raise telemetry.QualificationFailure(
                f"valid targets {workload.valid_targets} differ from nominal update tokens "
                f"{workload.update_tokens} at update {next_update}")
        tracker.record_decay(workload.optimizer)
        health = telemetry.require_healthy_state(workload.model, workload.optimizer)
        record = {"event": "qualification_update", **workload.state,
                  "update_seconds": update_seconds,
                  "data_wait_seconds": workload.data_wait_seconds,
                  "update_tokens": workload.update_tokens,
                  "valid_targets": workload.valid_targets,
                  "loss": workload.loss_sum / workload.update_tokens,
                  "lr_scale": workload.scale,
                  "grad_norm_before_clip": float(workload.grad_norm),
                  "group_lrs": telemetry.group_lr_records(workload.optimizer),
                  "finite": {"finite": health["finite"]["finite"],
                             "tensors_checked": health["finite"]["tensors_checked"],
                             "elements_checked": health["finite"]["elements_checked"],
                             "seconds": health["finite"]["seconds"]},
                  "required_gradients": {"checked": health["gradients"]["checked"],
                                         "seconds": health["gradients"]["seconds"]}}
        if diagnostic:
            record["role_diagnostics"] = tracker.finish(next_update)
        append_record(qualification_log, record)

        coefficient = min(1.0, workload.training.grad_clip / (float(workload.grad_norm) + 1e-6))
        clip_window.append(coefficient)
        if (alarm is None and next_update > warmup_updates and len(clip_window) == 100
                and sum(1 for value in clip_window if value < 0.1) > 20):
            alarm = "severe_sustained_clipping"
            summary["alarms"].append({"alarm": alarm, "update": next_update})
            append_record(qualification_log, {"event": "review_alarm", "alarm": alarm,
                                              "updates": next_update})

        stopped = workload.stop_requests > 0
        complete = workload.state["tokens_seen"] == workload.total_tokens
        if next_update in boundaries and not (stopped or complete):
            if next_update % workload.training.checkpoint_interval != 0:
                collective_checkpoint(Path(output_dir) / "checkpoints", workload.model,
                                      workload.optimizer, workload.state, workload.contract,
                                      workload.rank, workload.world_size)
            if next_update % workload.training.validation_interval != 0:
                results = evaluate(workload.model, workload.manifest, workload.data_root,
                                   workload.device, workload.amp_dtype,
                                   workload.training.validation_batches_per_language,
                                   workload.rank, workload.world_size)
                if workload.rank == 0:
                    append_event(Path(output_dir) / "events.jsonl",
                                 {"event": "validation", **workload.state, "languages": results})

        if ledger_path is not None and next_update % campaign.ROLE_DIAGNOSTIC_INTERVAL == 0:
            ewma_seconds = update_seconds if ewma_seconds is None else 0.9 * ewma_seconds + 0.1 * update_seconds
            ledger = budget.Ledger(ledger_path)
            if ledger.remaining_hours() <= 0:
                raise telemetry.QualificationFailure("allocation cap breached")
            try:
                ledger.require(ewma_seconds * (final_update - next_update) / 3600,
                               label=f"updates {next_update + 1}..{final_update}",
                               reserve_hours=emergency_reserve_hours)
            except budget.BudgetExhausted:
                if alarm is None:
                    alarm = "budget_projection"
                    summary["alarms"].append({"alarm": alarm, "update": next_update})
                    append_record(qualification_log, {"event": "review_alarm", "alarm": alarm,
                                                      "updates": next_update})
        if alarm is not None and next_update % workload.training.checkpoint_interval == 0:
            workload.should_stop[0] = True  # stop at a valid checkpointed boundary for review
        done = workload.cadence()
        if backup is not None:
            for name in _checkpoints_pending_upload(Path(output_dir) / "checkpoints"):
                backup.enqueue(name)
            health_report = backup_health(Path(output_dir) / "checkpoints", backup.dest,
                                          **backup_bounds)
            if health_report["alarms"] and alarm is None:
                alarm = "backup_health"
                summary["alarms"].append({"alarm": alarm, "update": next_update,
                                          "detail": health_report["alarms"]})
                append_record(qualification_log, {"event": "review_alarm",
                                                  "alarm": "backup_health",
                                                  "detail": health_report["alarms"],
                                                  "updates": next_update})
        summary["updates_completed"] = next_update
        if done:
            break
    return summary


def run(args) -> int:
    from terminus.train import prepare_training

    confirmation = bool(args.confirmation_from)
    trial_id = args.trial or args.confirmation_from
    probe = bool(args.probe)
    output_dir = Path(external_path(args.output_dir))
    basename = output_dir.name
    if probe:
        if args.stop_after_updates is None:
            raise ValueError("a probe requires --stop-after-updates")
        if basename in campaign.TRIAL_ORDER or basename == "confirmation":
            raise ValueError("probes must not use candidate output directories")
    else:
        expected = "confirmation" if confirmation else trial_id
        if basename != expected:
            raise ValueError(f"candidate output directory must be named {expected!r}")
        if args.stop_after_updates is not None:
            raise ValueError("candidate runs never truncate the complete schedule")

    tolerances = None
    if args.reference_probes:
        if args.replay_max_weight_delta is None or args.replay_max_moment_delta is None:
            raise ValueError("reference probes require frozen replay tolerances")
        tolerances = {"max_weight_delta": args.replay_max_weight_delta,
                      "max_moment_delta": args.replay_max_moment_delta}

    receipt = gate_offline(args.config, args.manifest, trial_id=trial_id,
                           confirmation=confirmation, ledger_path=args.ledger,
                           stage_forecast_hours=args.stage_forecast_hours,
                           source_receipt_path=args.source_receipt)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != campaign.WORLD_SIZE:
        raise ValueError("qualification requires exactly one H100 (world size 1)")

    from production.runtime import capture_runtime, verify_runtime

    report = json.loads(Path(external_path(args.runtime_report)).read_text())
    verify_runtime(report, capture_runtime())
    receipt["runtime_receipt"] = "verified"

    schedule = campaign.CONFIRMATION_SCHEDULE if confirmation else campaign.PROXY_SCHEDULE
    boundaries = tuple(schedule["boundary_updates"])
    final_update = schedule["total_updates"]

    backup = None
    backup_bounds = None
    if args.backup_destination:
        from .recovery import destination as make_destination

        output_dir.mkdir(parents=True, exist_ok=True)
        backup = BackupQueue(output_dir / "checkpoints", make_destination(args.backup_destination))
        backup_bounds = {"max_pending_bytes": args.backup_max_pending_bytes,
                         "max_pending_seconds": args.backup_max_pending_seconds,
                         "min_free_bytes": args.min_free_bytes}

    ledger = budget.Ledger(external_path(args.ledger))
    ledger.log_activity("run_start", {"trial": trial_id, "confirmation": confirmation,
                                      "probe": args.probe or None})
    output_dir.mkdir(parents=True, exist_ok=True)
    from terminus.integrity import write_json_atomic

    write_json_atomic(output_dir / "qualification_gate.json",
                      {**receipt, "kind": "probe" if probe else
                          ("confirmation" if confirmation else "trial")})
    try:
        with prepare_training(args.config, args.manifest, args.data_root, output_dir,
                              resume=args.resume, preflight_report=args.preflight_report,
                              staging_receipt=args.staging_receipt) as workload:
            model_config = workload.model_config
            expected_optimizer = (campaign.confirmation_optimizer_dict(trial_id) if confirmation
                                  else campaign.trial_optimizer_dict(trial_id))
            receipt["effective_groups"] = assert_effective_groups(
                model_config, workload.optimizer, expected_optimizer=expected_optimizer)
            write_json_atomic(output_dir / "qualification_gate.json",
                              {**receipt, "kind": "probe" if probe else
                                  ("confirmation" if confirmation else "trial")})
            if args.stop_after_updates is not None:
                if args.stop_after_updates <= workload.state["updates"]:
                    raise ValueError("stop-after-updates must exceed the resumed update count")
                workload.stop_after_updates = args.stop_after_updates
            summary = drive(workload, boundaries=boundaries, ledger_path=args.ledger,
                            final_update=(args.stop_after_updates or final_update),
                            output_dir=output_dir, reference_tolerances=tolerances,
                            backup=backup, backup_bounds=backup_bounds,
                            emergency_reserve_hours=args.emergency_reserve_hours)
    finally:
        if backup is not None:
            backup.close()
        budget.Ledger(external_path(args.ledger)).log_activity(
            "run_end", {"trial": trial_id, "confirmation": confirmation,
                        "probe": args.probe or None})
    print(json.dumps({"status": "run_finished", "trial": trial_id,
                      "confirmation": confirmation, "probe": args.probe or None,
                      "updates_completed": summary["updates_completed"],
                      "alarms": summary["alarms"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--trial", help="Candidate trial id (H, R0, R1, R2, R3)")
    who.add_argument("--confirmation-from", metavar="TRIAL",
                     help="Run the single transferred full-size confirmation")
    parser.add_argument("--probe", help="Mark a bounded non-candidate probe run")
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--staging-receipt", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--stage-forecast-hours", type=float, required=True)
    parser.add_argument("--emergency-reserve-hours", type=float,
                        default=campaign.EMERGENCY_RESERVE_HOURS)
    parser.add_argument("--backup-destination")
    parser.add_argument("--backup-max-pending-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--backup-max-pending-seconds", type=float, default=7200.0)
    parser.add_argument("--min-free-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--reference-probes", action="store_true")
    parser.add_argument("--replay-max-weight-delta", type=float)
    parser.add_argument("--replay-max-moment-delta", type=float)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    if args.probe and (args.trial is None):
        parser.error("--probe still needs --trial to identify the config contract")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
