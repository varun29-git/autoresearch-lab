"""Build the immutable external 24-hour packet from the stopped H campaign.

This is CPU-only and does not copy training data or checkpoints into Git.
The restored H payload, shard bytes, runtime and native gates remain launch
checks on the human-allocated H100.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from production.artifacts import external_path
from terminus.data import TokenManifest
from terminus.integrity import fingerprint, write_json_atomic

from . import campaign, horizons, plan
from .builder import (build_confirmation_config, build_trial_config,
                      validate_qualification_config)


def prepare(parent_root, output_root) -> dict:
    parent = Path(external_path(parent_root))
    root = Path(external_path(output_root, fresh=True))
    if root.exists():
        raise FileExistsError("new campaign root must not exist")
    git = subprocess.run(["git", "-C", str(campaign.REPOSITORY), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(campaign.REPOSITORY), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("freeze the 24-hour packet only from a clean source checkout")
    parent_run = json.loads((parent / "trials/H/run.json").read_text())
    parent_receipt = json.loads((parent / "trials/H/checkpoints/step_00000725.json").read_text())
    durable = json.loads((parent / "trials/H/checkpoints/step_00000725.uploaded.json").read_text())
    if (fingerprint(parent_run["contract"]) != parent_run["contract_sha256"]
            or parent_receipt["contract_sha256"] != parent_run["contract_sha256"]
            or parent_receipt["state"]["updates"] != campaign.CREDIT_H_UPDATES
            or parent_receipt["state"]["tokens_seen"] != campaign.CREDIT_H_UPDATES * campaign.GLOBAL_BATCH_TOKENS
            or durable["payload_sha256"] != parent_receipt["sha256"]
            or durable["contract_sha256"] != parent_run["contract_sha256"]):
        raise ValueError("parent H checkpoint identity or durable marker differs from the claimed prefix")
    proxy_parent_path = parent / "manifests/proxy_5b_m8.json"
    proxy_parent = TokenManifest.load(proxy_parent_path)
    if fingerprint(asdict(proxy_parent)) != parent_run["contract"]["data"]["manifest_sha256"]:
        raise ValueError("parent H manifest differs from the checkpoint contract")
    full_parent = TokenManifest.load(parent / "manifests/confirmation_1b_m2.json")
    proxy = horizons.build_horizon_view(
        proxy_parent, total_train_tokens=campaign.PROXY_SCHEDULE["total_tokens"], microbatch=8)
    full = horizons.build_horizon_view(
        full_parent, total_train_tokens=campaign.CONFIRMATION_SCHEDULE["total_tokens"], microbatch=2)
    if ([asdict(s) for s in proxy.shards] != [asdict(s) for s in proxy_parent.shards]
            or [asdict(s) for s in full.shards] != [asdict(s) for s in full_parent.shards]):
        raise ValueError("horizon views unexpectedly changed immutable shard identities")
    horizons.audit_token_accounting(proxy)
    horizons.audit_token_accounting(full)
    for directory in ("identity", "manifests", "configs", "evaluation", "gates", "trials",
                      "confirmation", "reports", "launch"):
        (root / directory).mkdir(parents=True, exist_ok=False)
    for trial in campaign.TRIAL_ORDER:
        (root / "trials" / trial).mkdir()
    proxy_path = root / "manifests/proxy_2b_m8.json"
    full_path = root / "manifests/confirmation_600m_m2.json"
    proxy.save(proxy_path)
    full.save(full_path)
    receipts = {}
    for trial in campaign.TRIAL_ORDER:
        path = root / "configs" / f"{trial}.json"
        write_json_atomic(path, build_trial_config(trial, microbatch=8))
        receipts[trial] = validate_qualification_config(path, proxy_path, trial_id=trial)
    fit_full = root / "configs/fit_full_R0.json"
    write_json_atomic(fit_full, build_confirmation_config("R0", microbatch=2))
    receipts["fit_full_R0"] = validate_qualification_config(
        fit_full, full_path, trial_id="R0", confirmation=True)
    source = campaign.qualification_source_hashes()
    write_json_atomic(root / "identity/source.json", {"schema": 1, "git_commit": git,
                                                     "source": source})
    write_json_atomic(root / "evaluation/margins.json", campaign.SELECTION_MARGINS)
    forecast = plan.make_plan()
    if not forecast["forecast_hours"]["fits_arithmetic_cap"]:
        raise ValueError("24-hour arithmetic forecast does not fit")
    write_json_atomic(root / "forecast.json", forecast)
    write_json_atomic(root / "campaign.json", {
        "schema": 2, "status": "staged_not_launched", "cap_hours": campaign.MAX_ALLOCATED_HOURS,
        "trials_in_order": list(campaign.TRIAL_ORDER),
        "global_batch_tokens": campaign.GLOBAL_BATCH_TOKENS,
        "proxy_schedule": campaign.PROXY_SCHEDULE,
        "confirmation_schedule": campaign.CONFIRMATION_SCHEDULE,
        "git_commit": git, "parent_campaign": str(parent),
        "parent_H_checkpoint_sha256": parent_receipt["sha256"],
        "parent_H_contract_sha256": parent_run["contract_sha256"],
        "H_credit_requires_native_fork_receipt": True,
    })
    write_json_atomic(root / "gates/offline_contract.json", {
        "status": "passed_cpu_only", "receipts": receipts,
        "parent_checkpoint_payload_verified_on_native": False})
    for gate, needed in {
        "runtime_native": "capture same native runtime as parent; verify source and FP8/preflight",
        "data_staging": "verify all shard/tokenizer bytes against both new manifests",
        "checkpoint_fork": "restore H payload from S3; validate payload, LR history and data continuation",
        "full_size_fit": "confirm full model m2 complete-update cadence-inclusive throughput and HBM headroom",
        "backup": "verify S3 upload and recovery against the new output prefix",
        "human_launch": "human opens fresh ledger at next VM allocation and launches auto.py",
    }.items():
        write_json_atomic(root / "gates" / f"{gate}.json",
                          {"status": "pending_native", "requirement": needed})
    (root / "launch/README.md").write_text(
        "See production/h100_24h/RUNBOOK.md in the pinned source checkout.\n"
        "This packet has no authority to start or stop a paid VM.\n", encoding="utf-8")
    return {"root": str(root), "status": "staged_not_launched",
            "proxy_tokens_each": proxy.config.usable_train_tokens,
            "full_tokens": full.config.usable_train_tokens,
            "forecast_hours": forecast["forecast_hours"]["total"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
