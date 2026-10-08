"""Short native full-update timing session on the research H100.

The broad production preflight is intentionally outside the quick timing path.
Failures still leave a trial artifact for the worker to archive and shut down.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from speed_research.protocol import file_digest, read_json, validate_lock
from terminus.data import TokenManifest

PHASES = ("dense", "qat")
REPO = Path(__file__).resolve().parent.parent


def load_lock(lock_path, expected_sha256):
    path = Path(lock_path)
    if file_digest(path) != expected_sha256:
        raise ValueError("staged campaign lock does not match the controller-declared SHA-256")
    return validate_lock(read_json(path))


def load_hashed_json(path, expected_sha256, name):
    path = Path(path)
    if file_digest(path) != expected_sha256:
        raise ValueError(f"{name} does not match the campaign lock SHA-256")
    return read_json(path)


def verify_manifest_identity(manifest_path, expected_sha256):
    path = Path(manifest_path)
    if file_digest(path) != expected_sha256:
        raise ValueError("staged data manifest does not match the campaign lock SHA-256")
    return TokenManifest.load(path)


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    plan = {k: v for k, v in plan.items() if not k.startswith("_")}
    if set(plan) - {"warmup_updates", "measured_updates", "profile_updates", "phases"}:
        raise ValueError("plan may only declare warmup_updates, measured_updates, profile_updates and phases")
    warmup, measured = plan.get("warmup_updates"), plan.get("measured_updates")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_updates must be a nonnegative integer")
    if isinstance(measured, bool) or not isinstance(measured, int) or measured <= 0:
        raise ValueError("measured_updates must be a positive integer")
    profile = plan.get("profile_updates", 0)
    if isinstance(profile, bool) or not isinstance(profile, int) or profile < 0:
        raise ValueError("profile_updates must be a nonnegative integer")
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases or len(set(phases)) != len(phases) \
            or set(phases) - set(PHASES):
        raise ValueError("phases must be a non-empty subset of dense/qat without repeats")
    return {"warmup_updates": warmup, "measured_updates": measured,
            "profile_updates": profile, "phases": list(phases)}


def phase_training_config(campaign, schedule, manifest, phase):
    """Training section for the locked measurement profile of one phase."""
    micro_tokens = manifest.config.tokens_per_step
    update = campaign["global_tokens_per_update"]
    if update % micro_tokens:
        raise ValueError(f"locked update {update} is not a whole number of {micro_tokens}-token microsteps")
    return {"device": "cuda", "dtype": "bf16", "development_mode": False,
            "seed": schedule["seed"], "gradient_accumulation_steps": update // micro_tokens,
            "warmup_tokens": schedule["warmup_tokens"], "final_lr_fraction": schedule["final_lr_fraction"],
            "schedule": schedule["schedule"], "decay_tokens": schedule["decay_tokens"],
            "decay_style": schedule["decay_style"], "phase_start_tokens": None,
            "grad_clip": schedule["grad_clip"],
            "quantization_warmup_tokens": 0, "quantization_start_strength": 1.0,
            "prefetch_batches": 4, "checkpoint_interval": 10**9, "validation_interval": 10**9,
            "validation_batches_per_language": 8, "log_interval": 10**9, "stability_log_interval": 0,
            "compile": False, "cuda_graphs": False, "time_limit_seconds": None}


def phase_model_config(campaign, phase):
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase}")
    return {**campaign["model"], "weight_mode": "dense" if phase == "dense" else "ternary"}


def run_plan(*, speed_root, data_root, results_root, cycle_id, plan_path,
             budget_seconds, command=subprocess.run):
    plan = validate_plan(read_json(plan_path))
    speed_root, data_root = Path(speed_root), Path(data_root)
    trial_dir = Path(results_root) / cycle_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + budget_seconds

    campaign = load_lock(speed_root / "campaign.lock.json",
                         read_json(speed_root / "campaign.lock.sha256")["sha256"])
    recipe = {k: v for k, v in load_hashed_json(speed_root / "optimizer_recipe.json",
                              campaign["optimizer_recipe_sha256"], "optimizer recipe").items()
              if not k.startswith("_")}
    schedule = load_hashed_json(speed_root / "schedule.json",
                                campaign["schedule_sha256"], "schedule")
    manifest = verify_manifest_identity(speed_root / "manifest.json",
                                        campaign["data_manifest_sha256"])
    (trial_dir / "manifest.json").write_text((speed_root / "manifest.json").read_text())

    summary = {"cycle_id": cycle_id, "plan": plan, "campaign_id": campaign["campaign_id"],
               "phases": {}, "deadline_seconds": budget_seconds}
    for phase in plan["phases"]:
        if time.monotonic() >= deadline:
            summary["phases"][phase] = {"status": "skipped", "reason": "trial time limit reached"}
            continue
        phase_dir = trial_dir / phase
        phase_dir.mkdir()
        config_payload = {"model": phase_model_config(campaign, phase),
                          "training": phase_training_config(campaign, schedule, manifest, phase),
                          "optimizer": recipe}
        config_path = phase_dir / "config.json"
        config_path.write_text(json.dumps(config_payload, indent=1, sort_keys=True) + "\n")
        record_path = phase_dir / "block.json"
        command([sys.executable, str(REPO / "scripts/s0_block.py"),
                 "--config", str(config_path), "--manifest", str(speed_root / "manifest.json"),
                 "--data-root", str(data_root), "--output-dir", str(phase_dir / "run"),
                 "--warmup-updates", str(plan["warmup_updates"]),
                 "--measured-updates", str(plan["measured_updates"]),
                 "--profile-updates", str(plan["profile_updates"]),
                 "--label", f"s0_{phase}", "--out", str(record_path)],
                check=True, cwd=REPO, timeout=max(1, deadline - time.monotonic()))
        summary["phases"][phase] = {"status": "completed",
                                    "block": json.loads(record_path.read_text())}
    summary_path = trial_dir / "s0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary
