"""Prospectively shorten a dense WSD run at a verified complete checkpoint.

The parent remains immutable. The child carries weights, optimizer moments,
RNG and the exact data cursor. Every LR and data choice before the fork must
match the parent; only the unconsumed WSD horizon may change. Run this on the
same native hardware/runtime used for training so the child contract is real.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from production.artifacts import external_path
from terminus.checkpoint import load_checkpoint, save_checkpoint
from terminus.data import TokenManifest
from terminus.integrity import file_sha256, fingerprint, source_hashes, validate_manifest, write_json_atomic
from terminus.model import BitNetLlama
from terminus.optim import build_optimizer
from terminus.stream_contract import verify_training_continuation
from terminus.train import load_config, make_resume_contract, schedule_scale

from . import campaign


# Reviewed source difference from the H parent at commit 784dc98: train.py
# gained a pre-step valid-target-count rejection. It does not alter a valid
# update. Any further core-source change needs a new reviewed fork policy.
PARENT_TRAIN_SHA256 = "33387ea84fa2e48b7fbfb9636f6181dedfc877c90e333b1eae6f0199da03c717"
CHILD_TRAIN_SHA256 = "118ce903e1d7e51088138987c012d76c16a744f7e39807eb339ce4c4f4e3b38d"


def _validated_parent_history(run_dir: Path, updates: int, batch_tokens: int) -> dict:
    """Require one healthy valid-target observation per consumed update."""
    events_path = run_dir / "events.jsonl"
    records = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    train = [item for item in records if item.get("event") == "train"]
    observed = {item.get("updates") for item in train}
    if len(train) != updates or observed != set(range(1, updates + 1)):
        raise ValueError("parent events do not contain every consumed complete update exactly once")
    if any(item.get("valid_targets") != item.get("loss_tokens")
           or item.get("loss_tokens") != batch_tokens for item in train):
        raise ValueError("parent valid-target history differs from the global update contract")
    qualification_path = run_dir / "qualification.jsonl"
    if not qualification_path.is_file():
        raise ValueError("parent full-state qualification record is missing")
    qualification = [json.loads(line) for line in qualification_path.read_text().splitlines()
                     if line.strip()]
    q_updates = [item for item in qualification if item.get("event") == "qualification_update"]
    if (len(q_updates) != updates
            or {item.get("updates") for item in q_updates} != observed
            or any(not item.get("finite", {}).get("finite") or
                   item.get("required_gradients", {}).get("checked", 0) <= 0
                   for item in q_updates)):
        raise ValueError("parent full-state checks have a gap or a failure")
    if any(item.get("event") == "review_alarm" for item in qualification):
        raise ValueError("parent qualification record contains a review alarm")
    incident_path = run_dir / "incidents.jsonl"
    incidents = ([json.loads(line) for line in incident_path.read_text().splitlines() if line.strip()]
                 if incident_path.is_file() else [])
    for incident in incidents:
        # H's operational S3 watchdog failure at step 100 was followed by an
        # exact recovery and 625 further healthy updates. New failure modes
        # must receive their own review and cannot be silently inherited.
        if not (incident.get("event") == "run_failure"
                and incident.get("error_type") == "NotImplementedError"
                and incident.get("error") == "remote re-read runs in the native recovery drill"
                and incident.get("state", {}).get("updates") == 100
                and updates >= 725):
            raise ValueError("parent contains an incident outside the reviewed recovery case")
    return {"events": records, "incidents": incidents,
            "valid_updates": len(train), "finite_updates": len(q_updates)}


def fork_dense_checkpoint(parent_run_dir, parent_manifest_path, child_config_path,
                          child_manifest_path, data_root, output_dir, *,
                          parent_checkpoint=None, device_name="cpu", world_size=1,
                          expected_parent_updates=campaign.CREDIT_H_UPDATES):
    """Create a fresh child run whose past is exactly the verified H prefix."""
    parent_run_dir = Path(external_path(parent_run_dir))
    parent_manifest_path = Path(external_path(parent_manifest_path))
    child_config_path = Path(external_path(child_config_path))
    child_manifest_path = Path(external_path(child_manifest_path))
    output_dir = Path(external_path(output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("fork target must be a fresh run directory")
    parent_run = json.loads((parent_run_dir / "run.json").read_text())
    parent_contract = parent_run["contract"]
    if fingerprint(parent_contract) != parent_run["contract_sha256"]:
        raise ValueError("parent run contract digest mismatch")
    source = Path(parent_checkpoint or parent_run_dir / "checkpoints")
    if source.is_dir():
        receipt = json.loads((source / "latest.json").read_text())
        source = source / receipt["file"]
    parent_payload = load_checkpoint(source, parent_contract)
    state = dict(parent_payload["state"])
    batch_tokens = (parent_contract["training"]["gradient_accumulation_steps"]
                    * TokenManifest.load(parent_manifest_path).config.tokens_per_step * world_size)
    if expected_parent_updates == campaign.CREDIT_H_UPDATES and batch_tokens != campaign.GLOBAL_BATCH_TOKENS:
        raise ValueError("H credit requires the declared global batch")
    if (world_size != parent_contract["world_size"]
            or len(parent_payload["rank_rng"]) != world_size
            or state["updates"] != expected_parent_updates
            or state["tokens_seen"] != expected_parent_updates * batch_tokens):
        raise ValueError("parent rank/update/token history differs from the declared fork point")
    history = _validated_parent_history(parent_run_dir, state["updates"], batch_tokens)

    model_config, training, optimizer_config = load_config(child_config_path)
    if asdict(model_config) != parent_contract["model"]:
        raise ValueError("dense fork must preserve the model exactly")
    if json.loads(json.dumps(asdict(optimizer_config))) != parent_contract["optimizer"]:
        raise ValueError("dense fork must preserve the optimizer and peak rates exactly")
    child_training = asdict(training)
    from terminus.train import OPERATIONAL_TRAINING_FIELDS
    for key in OPERATIONAL_TRAINING_FIELDS:
        child_training.pop(key)
    parent_training = dict(parent_contract["training"])
    if parent_training.pop("decay_tokens", None) == child_training.pop("decay_tokens", None):
        raise ValueError("fork requires a changed future decay length")
    if parent_training != child_training or training.schedule != "wsd":
        raise ValueError("fork may change only the future WSD decay length/horizon")
    old_total = parent_contract["data"]["training_stream"]["total_examples"] * model_config.max_seq_len
    parent_decay_start = old_total - parent_contract["training"]["decay_tokens"]
    child_manifest = TokenManifest.load(child_manifest_path)
    child_total = child_manifest.config.usable_train_tokens
    child_decay_start = child_total - training.decay_tokens
    if not state["tokens_seen"] < child_decay_start < child_total < old_total:
        raise ValueError("new decay must start after the parent checkpoint and end before old horizon")
    if not state["tokens_seen"] < parent_decay_start:
        raise ValueError("parent has already entered decay")
    for update in range(1, state["updates"] + 1):
        tokens = update * batch_tokens
        old = schedule_scale(tokens, old_total, training.warmup_tokens,
                             training.final_lr_fraction, schedule="wsd",
                             decay_tokens=parent_contract["training"]["decay_tokens"],
                             decay_style=training.decay_style)
        new = schedule_scale(tokens, child_total, training.warmup_tokens,
                             training.final_lr_fraction, schedule="wsd",
                             decay_tokens=training.decay_tokens, decay_style=training.decay_style)
        if old != new:
            raise ValueError(f"fork would change the LR already used at update {update}")

    parent_manifest = TokenManifest.load(parent_manifest_path)
    if fingerprint(asdict(parent_manifest)) != parent_contract["data"]["manifest_sha256"]:
        raise ValueError("supplied parent manifest differs from the checkpoint contract")
    parent_val = [asdict(spec) for spec in parent_manifest.shards
                  if spec.split == parent_manifest.config.val_split]
    child_val = [asdict(spec) for spec in child_manifest.shards
                 if spec.split == child_manifest.config.val_split]
    if parent_val != child_val or parent_manifest.tokenizer != child_manifest.tokenizer:
        raise ValueError("fork changed validation shards or tokenizer")
    manifest_report = validate_manifest(child_manifest, data_root, model_config,
                                        development_mode=training.development_mode)
    if child_manifest.config.tokens_per_step * world_size * training.gradient_accumulation_steps != batch_tokens:
        raise ValueError("fork changed the global batch")
    stream = verify_training_continuation(
        parent_contract["data"].get("training_stream"),
        manifest_report.get("training_stream"), state["tokens_seen"], policy="continue")
    current_source = source_hashes()
    changed = {key for key in set(current_source) | set(parent_contract["source"])
               if current_source.get(key) != parent_contract["source"].get(key)}
    if changed:
        if (changed != {"train.py"}
                or parent_contract["source"]["train.py"] != PARENT_TRAIN_SHA256
                or current_source["train.py"] != CHILD_TRAIN_SHA256):
            raise ValueError("core source changed outside the reviewed valid-target guard")
    device = torch.device(device_name)
    contract = make_resume_contract(model_config, training, optimizer_config,
                                    manifest_report, world_size, device)
    if contract["runtime"] != parent_contract["runtime"]:
        raise ValueError("native runtime changed across the checkpoint fork")
    micro_tokens = child_manifest.config.tokens_per_step * world_size
    if (state["next_micro_step"] != state["updates"] * training.gradient_accumulation_steps
            or state["tokens_seen"] != state["next_micro_step"] * micro_tokens):
        raise ValueError("fork requires the identical complete-update microbatch cursor")

    model = BitNetLlama(model_config).to(device)
    optimizer = build_optimizer(model, optimizer_config)
    model.load_state_dict(parent_payload["model"])
    optimizer.load_state_dict(parent_payload["optimizer"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_out = output_dir / "config.json"
    write_json_atomic(config_out, json.loads(child_config_path.read_text()))
    checkpoint = save_checkpoint(output_dir / "checkpoints", model, optimizer, state,
                                 contract, parent_payload["rank_rng"])
    write_json_atomic(output_dir / "run.json",
                      {"contract_sha256": fingerprint(contract), "contract": contract})
    retained = [item for item in history["events"]
                if item.get("event") == "validation" and item.get("updates") <= state["updates"]]
    with (output_dir / "events.jsonl").open("x") as handle:
        for item in retained:
            handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    receipt = {"schema": 1, "event": "prospective_dense_wsd_fork",
               "parent_checkpoint": str(source), "parent_sha256": file_sha256(source),
               "parent_contract_sha256": parent_run["contract_sha256"],
               "child_contract_sha256": fingerprint(contract),
               "child_checkpoint": checkpoint.name, "state": state,
               "past_lr_identical_updates": state["updates"],
               "stream_continuity": stream, "source_difference": sorted(changed),
               "parent_history": {"valid_updates": history["valid_updates"],
                                  "finite_updates": history["finite_updates"],
                                  "incidents": history["incidents"],
                                  "reviewed_operational_recovery": bool(history["incidents"])},
               "retained_validations": [item["updates"] for item in retained]}
    write_json_atomic(output_dir / "fork_receipt.json", receipt)
    return receipt


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-dir", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--child-config", required=True)
    parser.add_argument("--child-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(fork_dense_checkpoint(
        args.parent_run_dir, args.parent_manifest, args.child_config,
        args.child_manifest, args.data_root, args.output_dir,
        parent_checkpoint=args.parent_checkpoint, device_name=args.device),
        sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
