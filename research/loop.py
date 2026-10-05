"""Karpathy-style research loop adapter with a fixed Terminus evaluation contract.

The agent supplies hypotheses and commits architecture changes. This runner
prepares matched training manifests, enforces the campaign contract, executes
one bounded trial, and appends immutable results. It never creates a VM.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .protocol import (POLICY_PATH, ROOT, assert_clean_git, assert_edit_scope,
                       campaign, compare_losses, count_candidate, load_candidate,
                       paired_batch_interval, policy, read_json, sha256,
                       validate_task_manifest, write_json)


def trial_manifest(source: dict, rules: dict) -> dict:
    """Same shard identities and mixture; only the exposure and physical batch change."""
    value = json.loads(json.dumps(source))
    config = value["config"]
    if not config.get("source_mix") or config.get("phase_source_mixes"):
        raise ValueError("architecture search needs one fixed source_mix, not a phase curriculum")
    if config["seq_len"] != rules["max_seq_len"]:
        raise ValueError("prepared data sequence length differs from architecture policy")
    config["batch_size"] = rules["microbatch"]
    config["total_train_tokens"] = rules["full_tokens"]
    return value


def trial_config(model: dict, rules: dict) -> dict:
    value = read_json(ROOT / rules["base_config"])
    value["model"] = model
    train = value["training"]
    train["device"] = "cuda"
    train["dtype"] = "bf16"
    train["development_mode"] = False
    train["schedule"] = "wsd"
    train["warmup_tokens"] = int(rules["full_tokens"] * rules["warmup_fraction"])
    train["decay_tokens"] = int(rules["full_tokens"] * rules["decay_fraction"])
    train["gradient_accumulation_steps"] = rules["gradient_accumulation_steps"]
    train["validation_batches_per_language"] = 1
    train["validation_interval"] = 250
    train["checkpoint_interval"] = 10_000_000
    train["stability_log_interval"] = rules["stability_log_interval"]
    train["log_interval"] = 10
    train["time_limit_seconds"] = None
    return value


def prepare(campaign_path: Path) -> dict:
    rules = policy()
    settings = campaign(campaign_path)
    commit = assert_clean_git()
    model = load_candidate(rules)
    params = count_candidate(model, rules)
    tokenizer_path = ROOT / "tokenizer/tokenizer-65k.json"
    if sha256(tokenizer_path) != rules["tokenizer_sha256"]:
        raise ValueError("frozen 65K tokenizer checksum changed")
    tasks = validate_task_manifest(Path(settings["task_manifest"]), rules["domains"])
    from terminus.data import TokenDataConfig, TokenManifest, TokenShardSpec
    from terminus.integrity import validate_manifest
    from terminus.model import BitNetLlamaConfig

    source = read_json(Path(settings["manifest"]))
    narrowed = trial_manifest(source, rules)
    manifest = TokenManifest(config=TokenDataConfig(**narrowed["config"]),
                             shards=[TokenShardSpec(**item) for item in narrowed["shards"]],
                             tokenizer=narrowed.get("tokenizer", {}),
                             provenance=narrowed.get("provenance", {}))
    data_report = validate_manifest(manifest, settings["data_root"],
                                    BitNetLlamaConfig(**model), development_mode=False)
    root = Path(settings["results_root"]) / settings["tag"]
    root.mkdir(parents=True, exist_ok=True)
    lock = {"tag": settings["tag"], "base_commit": commit,
            "policy_sha256": sha256(POLICY_PATH),
            "manifest_sha256": sha256(Path(settings["manifest"])),
            "task_manifest_sha256": sha256(Path(settings["task_manifest"])),
            "task_files": tasks, "tokenizer_sha256": rules["tokenizer_sha256"],
            "baseline_parameters": params, "data_report": data_report,
            "settings": settings}
    lock_path = root / "campaign_lock.json"
    if lock_path.exists():
        if read_json(lock_path) != lock:
            raise ValueError("existing campaign lock differs; use a new tag for a new protocol")
    else:
        write_json(lock_path, lock)
    return {"ready": True, "tag": settings["tag"], "baseline_parameters": params,
            "task_files": len(tasks), "results": str(root), "base_commit": commit}


def read_lock(campaign_path: Path) -> tuple[dict, dict, dict, Path]:
    rules = policy()
    settings = campaign(campaign_path)
    root = Path(settings["results_root"]) / settings["tag"]
    lock = read_json(root / "campaign_lock.json")
    if lock["settings"] != settings or lock["policy_sha256"] != sha256(POLICY_PATH):
        raise ValueError("campaign settings or protected policy changed")
    if lock["manifest_sha256"] != sha256(Path(settings["manifest"])):
        raise ValueError("prepared data manifest changed")
    if lock["task_manifest_sha256"] != sha256(Path(settings["task_manifest"])):
        raise ValueError("development task manifest changed")
    tasks = validate_task_manifest(Path(settings["task_manifest"]), rules["domains"])
    if tasks != lock["task_files"]:
        raise ValueError("development task files changed")
    if sha256(ROOT / "tokenizer/tokenizer-65k.json") != lock["tokenizer_sha256"]:
        raise ValueError("tokenizer changed")
    return rules, settings, lock, root


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def spent_gpu_hours(ledger: list[dict]) -> float:
    return sum(float(item.get("gpu_hours", 0.0)) for item in ledger)


def run_command(command: list[str], log_path: Path, timeout: float) -> tuple[int, float, bool]:
    start = time.monotonic()
    timed_out = False
    with log_path.open("wb") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.send_signal(signal.SIGTERM)
            try:
                code = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
    return code, time.monotonic() - start, timed_out


def latest_state(run_dir: Path) -> dict | None:
    path = run_dir / "checkpoints/latest.json"
    return read_json(path)["state"] if path.exists() else None


def trial(campaign_path: Path, trial_id: str, stage: str) -> dict:
    """Serialize the whole preflight/train/eval transaction on the campaign GPU."""
    _, _, _, root = read_lock(campaign_path)
    with (root / ".active_trial.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another trial is active in this campaign") from exc
        try:
            return _trial_locked(campaign_path, trial_id, stage)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _trial_locked(campaign_path: Path, trial_id: str, stage: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", trial_id):
        raise ValueError("trial id must be a short filesystem-safe name")
    if stage not in {"screen", "full"}:
        raise ValueError("stage must be screen or full")
    rules, settings, lock, root = read_lock(campaign_path)
    commit = assert_clean_git()
    assert_edit_scope(lock["base_commit"], rules)
    model = load_candidate(rules)
    params = count_candidate(model, rules)
    if trial_id == settings["baseline_trial"] and commit != lock["base_commit"]:
        raise ValueError("baseline trial must use the unmodified campaign base commit")
    if trial_id != settings["baseline_trial"] and commit == lock["base_commit"]:
        raise ValueError("candidate trial must commit a change from the baseline")
    trial_dir = root / "trials" / trial_id
    run_dir = trial_dir / "run"
    ledger_path = root / "ledger.jsonl"
    ledger = read_ledger(ledger_path)
    if any(item["trial_id"] == trial_id and item["stage"] == stage
           and item["outcome"] == "completed" for item in ledger):
        raise ValueError("this trial stage already completed")
    remaining_seconds = (rules["max_gpu_hours"] - spent_gpu_hours(ledger)) * 3600
    if remaining_seconds < 300:
        raise RuntimeError("campaign GPU-hour cap exhausted")
    # Leave room for SIGTERM, a final optimizer update and its checkpoint.
    allowance = min(float(rules["max_trial_seconds"]), remaining_seconds - 150)
    if stage == "screen":
        if trial_dir.exists():
            metadata = read_json(trial_dir / "metadata.json")
            if metadata["commit"] != commit:
                raise ValueError("trial id belongs to another commit")
        else:
            trial_dir.mkdir(parents=True)
            write_json(trial_dir / "metadata.json", {"trial_id": trial_id,
                        "commit": commit, "parameters": params,
                        "policy_sha256": lock["policy_sha256"]})
            write_json(trial_dir / "config.json", trial_config(model, rules))
            write_json(trial_dir / "manifest.json",
                       trial_manifest(read_json(Path(settings["manifest"])), rules))
    else:
        metadata = read_json(trial_dir / "metadata.json")
        if metadata["commit"] != commit or not (trial_dir / "eval_screen.json").exists():
            raise ValueError("full continuation needs the completed screen from this exact commit")
    if read_json(trial_dir / "config.json") != trial_config(model, rules):
        raise ValueError("trial training config changed")
    if read_json(trial_dir / "manifest.json") != trial_manifest(read_json(Path(settings["manifest"])), rules):
        raise ValueError("trial data view changed")
    report = trial_dir / "preflight.json"
    commands = []
    if not report.exists():
        commands.append(("preflight", [sys.executable, "-m", "terminus.preflight",
                         "--config", str(trial_dir / "config.json"), "--output", str(report)]))
    global_batch = rules["microbatch"] * rules["max_seq_len"] * rules["gradient_accumulation_steps"]
    screen_updates = math.ceil(rules["screen_tokens"] / global_batch)
    state = latest_state(run_dir)
    if stage == "full" and not state:
        raise ValueError("full continuation has no committed checkpoint")
    train_command = [sys.executable, "-m", "terminus.train", "--config", str(trial_dir / "config.json"),
                     "--manifest", str(trial_dir / "manifest.json"),
                     "--data-root", settings["data_root"], "--output-dir", str(run_dir),
                     "--preflight-report", str(report)]
    if state:
        train_command += ["--resume", str(run_dir / "checkpoints")]
    if stage == "screen":
        train_command += ["--stop-after-updates", str(screen_updates)]
    target_tokens = (screen_updates * global_batch if stage == "screen"
                     else (rules["full_tokens"] // rules["max_seq_len"]) * rules["max_seq_len"])
    if not state or state["tokens_seen"] < target_tokens:
        commands.append(("train", train_command))
    evaluation_path = trial_dir / f"eval_{stage}.json"
    commands.append(("evaluation", [sys.executable, "-m", "research.evaluate",
                     "--run-dir", str(run_dir), "--manifest", str(trial_dir / "manifest.json"),
                     "--data-root", settings["data_root"], "--task-manifest", settings["task_manifest"],
                     "--output", str(evaluation_path), *(["--full"] if stage == "full" else [])]))
    started = time.time()
    duration, outcome, failed_step = 0.0, "completed", None
    try:
        for name, command in commands:
            if allowance - duration < 60:
                outcome, failed_step = "budget_stop", name
                break
            code, seconds, timed_out = run_command(command, trial_dir / f"{name}_{stage}.log",
                                                   allowance - duration)
            duration += seconds
            if timed_out or code != 0:
                outcome, failed_step = ("timeout" if timed_out else "failed"), name
                break
        state = latest_state(run_dir)
        if outcome == "completed" and (not state or state["tokens_seen"] < target_tokens):
            outcome, failed_step = "incomplete", "train"
        if outcome == "completed" and not evaluation_path.exists():
            outcome, failed_step = "incomplete", "evaluation"
    except Exception as exc:
        outcome, failed_step = "harness_error", type(exc).__name__
        raise
    finally:
        record = {"trial_id": trial_id, "stage": stage, "commit": commit,
                  "parameters": params, "started_unix": started,
                  "elapsed_seconds": duration, "gpu_hours": duration / 3600,
                  "outcome": outcome, "failed_step": failed_step,
                  "tokens_seen": (latest_state(run_dir) or {}).get("tokens_seen", 0),
                  "evaluation": str(evaluation_path.relative_to(root)) if evaluation_path.exists() else None}
        if outcome == "completed":
            evaluated = read_json(evaluation_path)
            record["evaluation_sha256"] = sha256(evaluation_path)
            record["heldout_losses"] = {name: value["loss"] for name, value in evaluated["heldout"].items()}
            if evaluated["tasks"]:
                record["task_summary"] = evaluated["tasks"]["summary"]
        append_ledger(ledger_path, record)
    return record


def status(campaign_path: Path) -> dict:
    rules, settings, _, root = read_lock(campaign_path)
    ledger = read_ledger(root / "ledger.jsonl")
    complete = [item for item in ledger if item["outcome"] == "completed"]
    comparisons = []
    for item in complete:
        if item.get("evaluation_sha256") != sha256(root / item["evaluation"]):
            raise ValueError(f"completed evaluation changed after ledger append: {item['trial_id']}/{item['stage']}")
    for item in complete:
        stage = item["stage"]
        baseline_path = root / "trials" / settings["baseline_trial"] / f"eval_{stage}.json"
        if not baseline_path.exists():
            continue
        baseline_eval = read_json(baseline_path)["heldout"]
        candidate_eval = read_json(root / item["evaluation"])["heldout"]
        baseline = {name: value["loss"] for name, value in baseline_eval.items()}
        candidate = {name: value["loss"] for name, value in candidate_eval.items()}
        comparisons.append({"trial_id": item["trial_id"], "stage": stage,
                            **compare_losses(candidate, baseline, rules),
                            "paired_interval": paired_batch_interval(candidate_eval, baseline_eval, rules)})
    return {"tag": settings["tag"], "attempts": len(ledger),
            "spent_gpu_hours": spent_gpu_hours(ledger),
            "remaining_gpu_hours": rules["max_gpu_hours"] - spent_gpu_hours(ledger),
            "comparisons": comparisons,
            "note": "task outcomes, long-context diagnostics and near-ties require review; this is not an automatic final architecture selection"}


def main(*, default_command: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if default_command is None:
        parser.add_argument("command", choices=("prepare", "trial", "status"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--stage", choices=("screen", "full"), default="screen")
    args = parser.parse_args()
    command = default_command or args.command
    if command == "prepare":
        result = prepare(args.campaign)
    elif command == "trial":
        if not args.trial_id:
            parser.error("trial needs --trial-id")
        result = trial(args.campaign, args.trial_id, args.stage)
    else:
        result = status(args.campaign)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
