"""Human-launched, fail-closed 24-hour trial handoff on one allocated H100.

This process never provisions, starts, stops or deallocates a VM. The human
opens the allocation ledger at launch and closes it at deallocation. The
controller verifies the frozen packet, forks the reviewed H checkpoint,
advances only completed healthy arms, and reforecasts all remaining work
before every stage. Any failed/stopped arm needs human review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from production.artifacts import external_path
from terminus.integrity import write_json_atomic

from . import budget, campaign
from .selection import load_trial_summary


PROXY_RATE_FLOOR_INPUT = 163_248.0  # recent cadence-inclusive H observation
FULL_RATE_INPUT = 35_000.0           # planning haircut, not sustained evidence
SETUP_RESERVE_HOURS = 3.0            # includes emergency 0.5 h


def observed_rate(trial_dir: Path, *, min_update: int = 1, window: int = 100) -> float | None:
    path = trial_dir / "events.jsonl"
    if not path.is_file():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in events if row.get("event") == "train"
            and row.get("updates", 0) >= min_update]
    if len(rows) < 21:
        return None
    rows = rows[-max(window + 1, 21):]
    seconds = rows[-1]["time_unix"] - rows[0]["time_unix"]
    tokens = rows[-1]["tokens_seen"] - rows[0]["tokens_seen"]
    return tokens / seconds if seconds > 0 and tokens > 0 else None


def remaining_work(root: Path, *, h_fork_credit: bool) -> tuple[dict, int]:
    """Fail if an existing arm is partial; never silently relabel an attempt."""
    pending = {}
    for trial in campaign.TRIAL_ORDER:
        trial_dir = root / "trials" / trial
        summary = load_trial_summary(trial_dir)
        if summary["completed"] and not summary["failed"] and not summary["missing_boundaries"]:
            pending[trial] = 0
        elif (trial_dir / "run.json").exists():
            if trial == "H" and h_fork_credit and (trial_dir / "fork_receipt.json").exists():
                state = json.loads((trial_dir / "checkpoints" / "latest.json").read_text())["state"]
                if state["updates"] == campaign.CREDIT_H_UPDATES:
                    pending[trial] = campaign.PROXY_SCHEDULE["total_updates"] - state["updates"]
                    continue
            raise RuntimeError(f"{trial} is partial/failed; review and resume it explicitly")
        else:
            pending[trial] = (campaign.PROXY_SCHEDULE["total_updates"]
                              - (campaign.CREDIT_H_UPDATES if trial == "H" and h_fork_credit else 0))
    confirmation = root / "confirmation"
    if (confirmation / "run.json").exists():
        summary = load_trial_summary(confirmation,
                                     boundary_updates=campaign.CONFIRMATION_BOUNDARY_UPDATES)
        if summary["completed"] and not summary["failed"] and not summary["missing_boundaries"]:
            return pending, 0
        raise RuntimeError("confirmation is partial/failed; review and resume it explicitly")
    return pending, campaign.CONFIRMATION_SCHEDULE["total_updates"]


def require_campaign_budget(root: Path, ledger: budget.Ledger, *, h_fork_credit: bool,
                            proxy_tps: float, full_tps: float) -> dict:
    if not ledger.allocation_open:
        raise ValueError("human must open the 24-hour allocation ledger at VM launch")
    pending, full_updates = remaining_work(root, h_fork_credit=h_fork_credit)
    proxy_tokens = sum(pending.values()) * campaign.GLOBAL_BATCH_TOKENS
    full_tokens = full_updates * campaign.GLOBAL_BATCH_TOKENS
    training_hours = (proxy_tokens / proxy_tps + full_tokens / full_tps) / 3600
    consumed = ledger.consumed_hours()
    remaining_overhead = max(campaign.EMERGENCY_RESERVE_HOURS,
                             SETUP_RESERVE_HOURS - consumed)
    receipt = ledger.require(training_hours, label="all remaining proxy and confirmation work",
                             reserve_hours=remaining_overhead)
    receipt.update({"proxy_tps": proxy_tps, "full_tps": full_tps,
                    "proxy_updates_remaining": pending,
                    "full_updates_remaining": full_updates,
                    "overhead_remaining_hours": remaining_overhead})
    return receipt


def _call(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def _runner_args(root: Path, args, *, trial: str, confirmation: bool,
                 rate: float, future_work_hours: float,
                 resume: Path | None = None) -> list[str]:
    config = root / ("confirmation/config.json" if confirmation else f"configs/{trial}.json")
    manifest = root / ("manifests/confirmation_600m_m2.json" if confirmation
                       else "manifests/proxy_2b_m8.json")
    output = root / ("confirmation" if confirmation else f"trials/{trial}")
    tokens = (campaign.CONFIRMATION_SCHEDULE if confirmation else campaign.PROXY_SCHEDULE)["total_tokens"]
    if resume is not None:
        tokens -= campaign.CREDIT_H_UPDATES * campaign.GLOBAL_BATCH_TOKENS
    stage_hours = tokens / rate / 3600
    command = ["--config", str(config), "--manifest", str(manifest),
               "--data-root", args.data_root, "--output-dir", str(output),
               "--confirmation-from" if confirmation else "--trial", trial,
               "--runtime-report", args.runtime_report,
               "--staging-receipt", args.staging_full if confirmation else args.staging_proxy,
               "--preflight-report", args.preflight_full if confirmation else args.preflight_proxy,
               "--source-receipt", str(root / "identity/source.json"),
               "--ledger", str(root / "budget.jsonl"),
               "--stage-forecast-hours", str(stage_hours),
               "--future-work-hours", str(max(0.0, future_work_hours)),
               "--backup-destination", args.backup_destination]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def run(args) -> int:
    root = Path(external_path(args.qual_root))
    packet = json.loads((root / "campaign.json").read_text())
    if (packet.get("cap_hours") != campaign.MAX_ALLOCATED_HOURS
            or packet.get("proxy_schedule") != campaign.PROXY_SCHEDULE
            or packet.get("confirmation_schedule") != campaign.CONFIRMATION_SCHEDULE):
        raise ValueError("external campaign packet differs from the 24-hour source contract")
    source = json.loads((root / "identity/source.json").read_text())
    if source.get("source") != campaign.qualification_source_hashes():
        raise ValueError("qualification source differs from the frozen packet")
    git = subprocess.run(["git", "-C", str(campaign.REPOSITORY), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(campaign.REPOSITORY), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if git != packet.get("git_commit") or git != source.get("git_commit") or dirty:
        raise ValueError("controller requires the clean packet-pinned Git commit")
    ledger = budget.Ledger(root / "budget.jsonl")
    if not ledger.allocation_open:
        raise ValueError("human must open the allocation ledger after starting the VM")
    if (not math.isfinite(args.proxy_tps) or not math.isfinite(args.full_tps)
            or args.proxy_tps <= 0 or args.full_tps <= 0):
        raise ValueError("throughput inputs must be positive")

    h_dir = root / "trials/H"
    if not (h_dir / "run.json").exists():
        require_campaign_budget(root, ledger, h_fork_credit=True,
                                proxy_tps=args.proxy_tps, full_tps=args.full_tps)
        _call("production.h100_24h.fork",
              "--parent-run-dir", args.parent_run_dir,
              "--parent-manifest", args.parent_manifest,
              "--parent-checkpoint", args.parent_checkpoint,
              "--child-config", root / "configs/H.json",
              "--child-manifest", root / "manifests/proxy_2b_m8.json",
              "--data-root", args.data_root, "--output-dir", h_dir)
    if not (h_dir / "fork_receipt.json").exists():
        raise ValueError("the 24-hour campaign requires the verified H checkpoint fork")
    fork = json.loads((h_dir / "fork_receipt.json").read_text())
    if fork.get("past_lr_identical_updates") != campaign.CREDIT_H_UPDATES:
        raise ValueError("H fork did not prove the credited update prefix")
    proxy_rate = args.proxy_tps
    for trial in campaign.TRIAL_ORDER:
        receipt = require_campaign_budget(root, ledger, h_fork_credit=True,
                                          proxy_tps=proxy_rate, full_tps=args.full_tps)
        write_json_atomic(root / "reports" / "live_forecast.json", receipt)
        if receipt["proxy_updates_remaining"][trial] == 0:
            continue
        output = root / "trials" / trial
        if trial != "H" and (output / "run.json").exists():
            raise RuntimeError(f"{trial} has an existing partial attempt")
        resume = h_dir / "checkpoints" / "step_00000725.pt" if trial == "H" else None
        _call("production.h100_24h.runner",
              *_runner_args(root, args, trial=trial, confirmation=False,
                            rate=proxy_rate,
                            future_work_hours=receipt["needed_hours"]
                            - (receipt["proxy_updates_remaining"][trial]
                               * campaign.GLOBAL_BATCH_TOKENS / proxy_rate / 3600),
                            resume=resume))
        summary = load_trial_summary(output)
        if summary["failed"] or not summary["completed"] or summary["missing_boundaries"]:
            raise RuntimeError(f"{trial} did not finish cleanly; stop automatic handoff")
        measured = observed_rate(output, min_update=campaign.CREDIT_H_UPDATES + 1 if trial == "H" else 1)
        if measured is not None:
            proxy_rate = measured * 0.95  # 5% planning haircut to observed cadence-inclusive rate

    require_campaign_budget(root, ledger, h_fork_credit=True,
                            proxy_tps=proxy_rate, full_tps=args.full_tps)
    selection = root / "reports" / "selection.json"
    if not selection.exists():
        _call("production.h100_24h.selection", "compare", "--qual-root", root,
              "--margins", root / "evaluation/margins.json", "--output", selection)
    selected = json.loads(selection.read_text())
    if selected.get("status") not in ("selected", "tie"):
        raise RuntimeError("comparison is incomplete; no full-size confirmation is authorized")
    winner = selected["selection"]
    transfer = root / "reports" / "transfer.json"
    if not transfer.exists():
        _call("production.h100_24h.selection", "transfer", "--selection", selection,
              "--output", transfer)
    full_config = root / "confirmation" / "config.json"
    if not full_config.exists():
        _call("production.h100_24h.builder", "--confirm-from", winner,
              "--microbatch", "2", "--manifest", root / "manifests/confirmation_600m_m2.json",
              "--output", full_config)
    receipt = require_campaign_budget(root, ledger, h_fork_credit=True,
                                      proxy_tps=proxy_rate, full_tps=args.full_tps)
    write_json_atomic(root / "reports" / "live_forecast.json", receipt)
    if receipt["full_updates_remaining"]:
        _call("production.h100_24h.runner",
              *_runner_args(root, args, trial=winner, confirmation=True,
                            rate=args.full_tps, future_work_hours=0.0))
    summary = load_trial_summary(root / "confirmation",
                                 boundary_updates=campaign.CONFIRMATION_BOUNDARY_UPDATES)
    if summary["failed"] or not summary["completed"] or summary["missing_boundaries"]:
        raise RuntimeError("full-size confirmation did not finish cleanly")
    print(json.dumps({"status": "campaign_complete", "winner": winner,
                      "allocation_hours_used": ledger.consumed_hours()}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qual-root", "data-root", "parent-run-dir", "parent-manifest",
                 "parent-checkpoint", "runtime-report", "staging-proxy", "staging-full",
                 "preflight-proxy", "preflight-full", "backup-destination"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--proxy-tps", type=float, default=PROXY_RATE_FLOOR_INPUT)
    parser.add_argument("--full-tps", type=float, default=FULL_RATE_INPUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
