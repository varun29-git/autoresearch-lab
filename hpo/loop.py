"""Run and compare one dense HPO candidate at fixed token milestones."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

from terminus.train import load_config, prepare_training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "hpo/reference_200m.json"
DEFAULT_CANDIDATE = ROOT / "hpo/candidate.json"
OPTIMIZER_FIELDS = frozenset({"muon_lr", "adamw_lr", "norm_lr", "qk_norm_lr",
                              "weight_decay", "momentum", "betas", "eps"})
TRAINING_FIELDS = frozenset({"grad_clip"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = [*sorted((ROOT / "terminus").glob("*.py")), ROOT / "hpo/loop.py"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def load_reference(path: Path) -> dict:
    reference = json.loads(path.read_text())
    if set(reference) != {"campaign_id", "manifest_sha256", "stages",
                          "screen_max_updates", "config"}:
        raise ValueError("reference must pin campaign, manifest, stages, and config")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", reference["campaign_id"]):
        raise ValueError("invalid campaign ID")
    if not re.fullmatch(r"[0-9a-f]{64}", reference["manifest_sha256"]):
        raise ValueError("manifest SHA-256 must be a complete lowercase digest")
    stages = reference["stages"]
    if (not isinstance(stages, list) or not stages
            or any(type(stage) is not int or stage <= 0 for stage in stages)
            or stages != sorted(set(stages))):
        raise ValueError("stages must be sorted, unique positive update counts")
    if (type(reference["screen_max_updates"]) is not int
            or reference["screen_max_updates"] not in stages):
        raise ValueError("screen maximum must be one of the stages")
    if set(reference["config"]) != {"model", "training", "optimizer"}:
        raise ValueError("reference config must contain model, training, optimizer")
    return reference


def candidate_config(reference: dict, candidate: dict) -> dict:
    if set(candidate) != {"optimizer", "training"}:
        raise ValueError("candidate needs optimizer and training override objects")
    if not isinstance(candidate["optimizer"], dict) or not isinstance(candidate["training"], dict):
        raise ValueError("candidate overrides must be objects")
    if set(candidate["optimizer"]) - OPTIMIZER_FIELDS:
        raise ValueError("candidate changes a fixed optimizer field")
    if set(candidate["training"]) - TRAINING_FIELDS:
        raise ValueError("candidate changes the fixed training schedule or batch")
    config = copy.deepcopy(reference["config"])
    for section in ("optimizer", "training"):
        config[section].update(candidate[section])
    return config


def validate_config(config: dict, scratch: Path) -> None:
    scratch.write_text(json.dumps(config, sort_keys=True) + "\n")
    try:
        load_config(scratch)
    finally:
        scratch.unlink()


def validation_curve(path: Path, stages: list[int], tokens_per_update: int) -> dict[int, dict]:
    curve = {}
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") != "validation" or event.get("updates") not in stages:
            continue
        update = event["updates"]
        if event["tokens_seen"] != update * tokens_per_update:
            raise ValueError("validation token count differs from the fixed batch")
        english = event["languages"]["en"]
        loss = float(english["loss"])
        if not math.isfinite(loss):
            raise ValueError("nonfinite held-out cross-entropy")
        curve[update] = {"updates": update, "tokens": event["tokens_seen"],
                         "english_ce": loss, "validation_tokens": english["tokens"]}
    return curve


def throughput(path: Path) -> float | None:
    rates = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "train":
            rate = event.get("tokens_per_second")
            if isinstance(rate, (float, int)) and math.isfinite(rate) and rate > 0:
                rates.append(float(rate))
    return statistics.median(rates) if rates else None


def run(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.trial):
        raise ValueError("trial name must use lowercase letters, digits, underscores or hyphens")
    reference = load_reference(args.reference)
    if args.updates not in reference["stages"]:
        raise ValueError("updates must be a declared milestone")
    if args.updates > reference["screen_max_updates"] and not args.finalist:
        raise ValueError("a run past the screen limit requires --finalist")
    if sha256(args.manifest) != reference["manifest_sha256"]:
        raise ValueError("manifest differs from this campaign's fixed stream")
    results_root = args.results_root.resolve()
    if results_root == ROOT or ROOT in results_root.parents:
        raise ValueError("results and checkpoints must live outside Git")
    campaign_root = results_root / reference["campaign_id"]
    ledger = campaign_root / "results.jsonl"
    current_sources = source_hashes()
    reference_digest = sha256(args.reference)
    if ledger.exists() and ledger.stat().st_size:
        first = json.loads(next(line for line in ledger.read_text().splitlines() if line.strip()))
        if (first["reference_sha256"] != reference_digest
                or first["manifest_sha256"] != reference["manifest_sha256"]
                or first["source_hashes"] != current_sources):
            raise ValueError("campaign source, scorer, or stream changed after its first trial")
    candidate = json.loads(args.candidate.read_text())
    config = candidate_config(reference, candidate)
    from terminus.data import TokenManifest

    manifest = TokenManifest.load(args.manifest)
    tokens_per_update = (manifest.config.batch_size * manifest.config.seq_len
                         * config["training"]["gradient_accumulation_steps"])
    if tokens_per_update != 393216:
        raise ValueError("this HPO campaign requires 393,216 tokens per update")
    if args.updates * tokens_per_update > manifest.config.usable_train_tokens:
        raise ValueError("milestone exceeds the prepared stream")

    trial_dir = campaign_root / args.trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path, candidate_path = trial_dir / "config.json", trial_dir / "candidate.json"
    source_path = trial_dir / "source_hashes.json"
    if config_path.exists():
        if (json.loads(config_path.read_text()) != config
                or json.loads(candidate_path.read_text()) != candidate
                or json.loads(source_path.read_text()) != current_sources):
            raise ValueError("a resumed trial must keep its candidate and code unchanged")
    else:
        validate_config(config, trial_dir / ".config-check.json")
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
        source_path.write_text(json.dumps(current_sources, indent=2, sort_keys=True) + "\n")
    run_dir = trial_dir / "run"
    previous_result = trial_dir / "result.json"
    if previous_result.exists() and json.loads(previous_result.read_text())["updates"] >= args.updates:
        raise ValueError("trial has already reached this milestone")
    resume = run_dir / "checkpoints" if (run_dir / "run.json").exists() else None
    started = time.monotonic()
    # Research runs still execute the full train, validation, and checkpoint path.
    # The older production preflight report is unnecessary for this one-GPU loop.
    with prepare_training(config_path, args.manifest, args.data_root, run_dir,
                          resume=resume, research_timing_only=True) as workload:
        workload.stop_after_updates = args.updates
        while workload.state["next_micro_step"] < workload.total_micro_steps:
            workload.complete_update()
            if workload.cadence():
                break
        state = workload.state
    elapsed = time.monotonic() - started
    if state["updates"] != args.updates or state["tokens_seen"] != args.updates * tokens_per_update:
        raise RuntimeError("training did not reach the requested matched-token milestone")
    curve = validation_curve(run_dir / "events.jsonl", reference["stages"], tokens_per_update)
    if args.updates not in curve:
        raise RuntimeError("the requested held-out evaluation was not recorded")
    result = {"campaign_id": reference["campaign_id"], "trial": args.trial,
              "candidate": candidate, "updates": args.updates, "tokens": state["tokens_seen"],
              "english_ce": curve[args.updates]["english_ce"],
              "curve": [curve[step] for step in sorted(curve)],
              "median_train_tokens_per_second": throughput(run_dir / "events.jsonl"),
              "invocation_wall_seconds": elapsed, "reference_sha256": reference_digest,
              "manifest_sha256": reference["manifest_sha256"],
              "config_sha256": sha256(config_path), "source_hashes": current_sources,
              "events_sha256": sha256(run_dir / "events.jsonl")}
    previous_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with ledger.open("a") as stream:
        stream.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def rank(args: argparse.Namespace) -> list[dict]:
    reference = load_reference(args.reference)
    if args.updates not in reference["stages"]:
        raise ValueError("rank at one declared milestone")
    campaign_root = args.results_root.resolve() / reference["campaign_id"]
    rows = []
    for result_path in campaign_root.glob("*/result.json"):
        result = json.loads(result_path.read_text())
        if (result["reference_sha256"] != sha256(args.reference)
                or result["manifest_sha256"] != reference["manifest_sha256"]):
            continue
        match = next((point for point in result["curve"] if point["updates"] == args.updates), None)
        if match is not None:
            rows.append({"trial": result["trial"], "updates": args.updates,
                         "tokens": match["tokens"], "english_ce": match["english_ce"],
                         "candidate": result["candidate"]})
    return sorted(rows, key=lambda row: row["english_ce"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "rank"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
        sub.add_argument("--results-root", type=Path, required=True)
        sub.add_argument("--updates", type=int, required=True)
        if command == "run":
            sub.add_argument("--trial", required=True)
            sub.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
            sub.add_argument("--manifest", type=Path, required=True)
            sub.add_argument("--data-root", type=Path, required=True)
            sub.add_argument("--finalist", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args) if args.command == "run" else rank(args), sort_keys=True))


if __name__ == "__main__":
    main()
