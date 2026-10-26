"""Deliberate dense -> ternary-QAT phase transition.

`convert_checkpoint` turns an actually trained dense-stage checkpoint into the
initial checkpoint of a QAT run, with an explicit receipt. It:

- verifies the parent checkpoint against the parent run's own contract
  (integrity + counters) before converting;
- builds the child (QAT) contract exactly as `train.py` will compute it at
  resume time, so exact-resume protection within the QAT phase is unchanged —
  run this tool on the training host, with the same world size, because the
  contract binds runtime and native-source identity;
- carries model weights, compatible optimizer state, per-rank RNG, and the
  data cursor (unless a reset is explicitly requested and recorded);
- can change microbatch while preserving global tokens per update, rebasing
  the microstep cursor without changing token exposure or update count;
- supports checkpoint branches: any parent checkpoint of the dense run may be
  converted, and the receipt records which one; different pre-switch histories
  are never presented as shared.

Data policies:
- "continue": the child manifest covers the same stream with a larger budget
  (parent tokens_seen + QAT budget); the carried cursor guarantees no data
  repeats or disappears across the switch.
- "phase_mix": the child manifest uses phase_source_mixes whose phase boundary
  must equal the parent's tokens_seen, so the router switches mixture exactly
  at the transition.
- "restart": the data and update counters (and optionally RNG) are reset;
  the receipt records this as an intentional reset. A manifest may start each
  source at the exact unused example offset, so this can avoid repetition.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import TokenManifest
from .integrity import fingerprint, source_hashes, validate_manifest, write_json_atomic
from .model import BitNetLlama
from .optim import build_optimizer
from .train import load_config, make_resume_contract

POLICIES = ("continue", "phase_mix", "restart")
RESETTABLE = {"optimizer", "data_cursor", "rng"}


def _cumulative_phase_boundaries(config):
    boundaries = []
    total = 0
    for phase in config.phase_source_mixes:
        total += phase["tokens"]
        boundaries.append(total)
    return boundaries


def convert_checkpoint(parent_run_dir, qat_config_path, qat_manifest_path, qat_data_root,
                       output_dir, *, world_size=1, device_name="cpu", parent_checkpoint=None,
                       data_policy="continue", intentional_resets=()):
    parent_run_dir = Path(parent_run_dir)
    output_dir = Path(output_dir)
    if data_policy not in POLICIES:
        raise ValueError(f"data_policy must be one of {POLICIES}")
    resets = set(intentional_resets)
    if unknown := resets - RESETTABLE:
        raise ValueError(f"unknown intentional resets: {sorted(unknown)}")
    if data_policy == "restart":
        resets.add("data_cursor")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("the QAT run directory must be fresh")
    parent_run = json.loads((parent_run_dir / "run.json").read_text())
    parent_contract = parent_run["contract"]
    if parent_run["contract_sha256"] != fingerprint(parent_contract):
        raise ValueError("parent run contract digest mismatch")
    if parent_contract["model"]["weight_mode"] != "dense":
        raise ValueError("dense-to-QAT transition requires a dense parent")
    parent_source = parent_checkpoint or parent_run_dir / "checkpoints"
    parent_source = Path(parent_source)
    if parent_source.is_dir():
        latest = json.loads((parent_source / "latest.json").read_text())
        if Path(latest["file"]).name != latest["file"]:
            raise ValueError("invalid parent checkpoint filename")
        parent_source = parent_source / latest["file"]
    parent_payload = load_checkpoint(parent_source, parent_contract)
    parent_state = parent_payload["state"]

    config_payload = json.loads(Path(qat_config_path).read_text())
    parent_seen = parent_payload["state"]["tokens_seen"]
    expected_offset = 0 if data_policy == "restart" else parent_seen
    declared = config_payload["training"].get("phase_start_tokens")
    if declared is not None and declared != expected_offset:
        raise ValueError(f"QAT config phase_start_tokens={declared} but the transition needs "
                         f"{expected_offset}; leave it null and let the transition set it")
    config_payload["training"]["phase_start_tokens"] = expected_offset
    from .model import BitNetLlamaConfig
    from .train import TrainConfig
    from .optim import OptimizerConfig
    model_config = BitNetLlamaConfig(**config_payload["model"])
    training = TrainConfig(**config_payload["training"])
    optimizer_config = OptimizerConfig(**config_payload["optimizer"])
    if model_config.weight_mode != "ternary":
        raise ValueError("the QAT phase must run the ternary weight mode")
    parent_model = BitNetLlamaConfig(**parent_contract["model"])
    architectural_changes = _diff(asdict(parent_model), asdict(model_config))
    # A phase transition is not an implicit architecture/checkpoint migration.
    if set(architectural_changes) - {"weight_mode", "quantization_strength"}:
        raise ValueError("QAT transition changes model settings beyond weight quantization")
    if "data_cursor" in resets and data_policy != "restart":
        raise ValueError("a data_cursor reset requires data_policy='restart'")
    if "optimizer" not in resets:
        parent_optimizer = asdict(OptimizerConfig(**parent_contract["optimizer"]))
        child_optimizer = asdict(optimizer_config)
        changes = _diff(json.loads(json.dumps(parent_optimizer)), json.loads(json.dumps(child_optimizer)))
        if any(not key.endswith("_lr") for key in changes):
            raise ValueError("carrying optimizer state permits LR changes only; explicitly reset optimizer otherwise")
    manifest = TokenManifest.load(qat_manifest_path)
    manifest_report = validate_manifest(manifest, qat_data_root, model_config,
                                        development_mode=training.development_mode)
    parent_tokenizer = parent_contract["data"].get("tokenizer") or {}
    child_tokenizer = manifest_report.get("tokenizer") or {}
    if parent_tokenizer and child_tokenizer and \
            parent_tokenizer.get("sha256") != child_tokenizer.get("sha256"):
        raise ValueError("parent and child manifests use different tokenizers")
    device = torch.device(device_name)
    contract = make_resume_contract(model_config, training, optimizer_config, manifest_report,
                                    world_size, device)

    model = BitNetLlama(model_config)
    optimizer = build_optimizer(model, optimizer_config)
    child_base_lrs = [group["base_lr"] for group in optimizer.param_groups]
    model.load_state_dict(parent_payload["model"])  # strict: shapes must match
    resets_done = []
    if "optimizer" in resets:
        resets_done.append("optimizer")
    else:
        optimizer.load_state_dict(parent_payload["optimizer"])
        if len(optimizer.param_groups) != len(child_base_lrs):
            raise ValueError("parent optimizer groups do not match QAT optimizer groups")
        # Loading moments also restores the dense phase's param-group rates.
        # Keep moments but apply the child phase's explicitly configured rates.
        for group, child_lr in zip(optimizer.param_groups, child_base_lrs):
            group["base_lr"] = child_lr
            group["lr"] = child_lr
    total_tokens = manifest.config.usable_train_tokens
    from .train import schedule_scale
    schedule_scale(0, total_tokens - expected_offset, training.warmup_tokens,
                   training.final_lr_fraction, schedule=training.schedule,
                   decay_tokens=training.decay_tokens, decay_style=training.decay_style)
    if training.quantization_warmup_tokens > total_tokens - expected_offset:
        raise ValueError("quantization warmup exceeds the QAT phase budget")
    if model_config.linear_backend == "deepgemm" and (training.quantization_warmup_tokens
            or training.quantization_start_strength != 1):
        raise ValueError("native DeepGEMM does not support a quantization curriculum")
    micro_tokens = manifest.config.tokens_per_step * world_size
    total_micro_steps = -(-manifest.config.total_examples //
                          (manifest.config.batch_size * world_size))
    if "data_cursor" in resets:
        state = {"next_micro_step": 0, "tokens_seen": 0, "updates": 0}
        resets_done.extend(("data_cursor", "update_counter"))
    else:
        state = dict(parent_state)
        parent_accumulation = parent_contract["training"]["gradient_accumulation_steps"]
        if state["next_micro_step"] != state["updates"] * parent_accumulation:
            raise ValueError("parent checkpoint is not at a complete global update boundary")
        if state["tokens_seen"] % micro_tokens:
            raise ValueError("carried token cursor is not divisible by the child microbatch tokens; "
                             "choose an aligned child microbatch or request data_policy='restart'")
        child_step = state["tokens_seen"] // micro_tokens
        if child_step != state["updates"] * training.gradient_accumulation_steps:
            raise ValueError("carried updates do not align with the child global batch; "
                             "keep tokens per update fixed or request data_policy='restart'")
        # Re-express the same global token cursor in the child's microbatch
        # units. This preserves update count, optimizer moments and data order.
        state["next_micro_step"] = child_step
        if state["tokens_seen"] >= total_tokens:
            raise ValueError("child budget must extend beyond the parent's consumed tokens")
        if data_policy == "phase_mix":
            boundaries = _cumulative_phase_boundaries(manifest.config)
            if state["tokens_seen"] not in boundaries:
                raise ValueError("phase_mix requires a phase boundary exactly at the parent's "
                                 f"consumed tokens ({state['tokens_seen']}); boundaries: {boundaries}")
    if len(parent_payload["rank_rng"]) != world_size and "rng" not in resets:
        raise ValueError("world size changed; request intentional_resets=('rng',) explicitly")
    if "rng" in resets:
        resets_done.append("rng")
        import random as random_module

        import numpy as np

        from .checkpoint import rng_state
        rank_rng = []
        for rank in range(world_size):  # same fresh-run seeding train.py applies per rank
            random_module.seed(training.seed + rank)
            np.random.seed((training.seed + rank) % (2**32))
            torch.manual_seed(training.seed + rank)
            rank_rng.append(rng_state())
    else:
        rank_rng = parent_payload["rank_rng"]

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_config_path = output_dir / "qat_config.json"
    write_json_atomic(effective_config_path, config_payload)
    checkpoints = output_dir / "checkpoints"
    path = save_checkpoint(checkpoints, model, optimizer, state, contract, rank_rng)
    write_json_atomic(output_dir / "run.json",
                      {"contract_sha256": fingerprint(contract), "contract": contract})
    parent_receipt_path = parent_payload_receipt(parent_source)
    parent_receipt = json.loads(parent_receipt_path.read_text()) if parent_receipt_path else {}
    child_training = asdict(training)
    from .train import OPERATIONAL_TRAINING_FIELDS
    for name in OPERATIONAL_TRAINING_FIELDS:
        child_training.pop(name)
    changed = {"model": _diff(parent_contract["model"], asdict(model_config)),
               "training": _diff(parent_contract["training"], child_training),
               "optimizer": _diff(parent_contract["optimizer"],
                                  json.loads(json.dumps(asdict(optimizer_config))))}
    receipt = {"event": "dense_to_qat_transition", "time_unix": time.time(),
               "parent": {"run_dir": str(parent_run_dir),
                          "checkpoint_sha256": parent_receipt.get("sha256"),
                          "checkpoint_file": parent_receipt.get("file"),
                          "contract_sha256": parent_run["contract_sha256"],
                          "state": parent_state},
               "child": {"config_sha256": fingerprint(config_payload),
                         "effective_config": str(effective_config_path),
                         "contract_sha256": fingerprint(contract),
                         "manifest_sha256": manifest_report["manifest_sha256"],
                         "initial_checkpoint": path.name, "state": state},
               "data_policy": data_policy, "intentional_resets": sorted(resets_done),
               "cursor_rebase": None if "data_cursor" in resets else {
                   "parent_next_micro_step": parent_state["next_micro_step"],
                   "child_next_micro_step": state["next_micro_step"],
                   "tokens_seen": state["tokens_seen"]},
               "changed_fields": changed,
               "tokenizer_sha256": child_tokenizer.get("sha256"),
               "source": source_hashes()}
    write_json_atomic(output_dir / "transition_receipt.json", receipt)
    return receipt


def parent_payload_receipt(source):
    source = Path(source)
    if source.is_dir():
        candidate = source / "latest.json"
        return candidate if candidate.exists() else None
    candidate = source.with_suffix(".json")
    return candidate if candidate.exists() else None


def _diff(parent, child):
    changed = {}
    for key in sorted(set(parent) | set(child)):
        if parent.get(key) != child.get(key):
            changed[key] = {"parent": parent.get(key), "child": child.get(key)}
    return changed


def main():
    parser = argparse.ArgumentParser(description="Convert a dense-stage checkpoint into a QAT run")
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--config", required=True, help="QAT phase config JSON")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-policy", choices=POLICIES, default="continue")
    parser.add_argument("--intentional-resets", default="", help="comma-separated: optimizer,data_cursor,rng")
    args = parser.parse_args()
    resets = tuple(name for name in args.intentional_resets.split(",") if name)
    receipt = convert_checkpoint(args.parent_run, args.config, args.manifest, args.data_root,
                                 args.output_dir, world_size=args.world_size, device_name=args.device,
                                 parent_checkpoint=args.parent_checkpoint,
                                 data_policy=args.data_policy, intentional_resets=resets)
    print(json.dumps({"transition": receipt["child"]["contract_sha256"],
                      "parent_state": receipt["parent"]["state"],
                      "child_state": receipt["child"]["state"]}, indent=1))


if __name__ == "__main__":
    main()
