"""Qualification config builder/validator: a separate, narrowly allowlisted path.

Generates fully resolved trainer configs for exactly the five reviewed proxy
trials and the single transferred full-size winner, and validates loaded
configs against that contract. This does not modify or weaken the production
locks in production.build_dense_config, production.inspect_run or
production/train.sh; those continue to reject the proxy geometry.

Nothing here launches training. Outputs must be fresh paths outside Git.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from production.artifacts import external_path

from . import campaign


def _resolved_sections(model_dict: dict, optimizer_dict: dict, training_dict: dict) -> dict:
    """Instantiate the trainer dataclasses and emit every default explicitly."""
    from terminus.model import BitNetLlamaConfig
    from terminus.optim import OptimizerConfig
    from terminus.train import TrainConfig

    return {
        "model": asdict(BitNetLlamaConfig(**model_dict)),
        "training": asdict(TrainConfig(**training_dict)),
        "optimizer": asdict(OptimizerConfig(**optimizer_dict)),
    }


def _training_dict(schedule: dict, *, microbatch: int) -> dict:
    if type(microbatch) is not int or microbatch <= 0:
        raise ValueError("microbatch must be a positive integer")
    steps, remainder = divmod(campaign.GLOBAL_BATCH_TOKENS, microbatch * 4096)
    if remainder or steps < 1:
        raise ValueError("global batch must be a positive multiple of microbatch tokens")
    return {
        **campaign.SELECTED_TRAINING_LIKE,
        "gradient_accumulation_steps": steps,
        "warmup_tokens": schedule["warmup_tokens"],
        "decay_tokens": schedule["decay_tokens"],
        "checkpoint_interval": campaign.CHECKPOINT_INTERVAL,
        "validation_interval": campaign.VALIDATION_INTERVAL,
        "validation_batches_per_language": campaign.VALIDATION_BATCHES_PER_LANGUAGE,
        "log_interval": campaign.LOG_INTERVAL,
        "stability_log_interval": campaign.STABILITY_LOG_INTERVAL,
        "time_limit_seconds": None,
    }


def build_trial_config(trial_id: str, *, microbatch: int) -> dict:
    """Fully resolved proxy config for one matrix arm (H, R0, R1, R2, R3)."""
    return _resolved_sections(campaign.proxy_model_dict(),
                              campaign.trial_optimizer_dict(trial_id),
                              _training_dict(campaign.PROXY_SCHEDULE, microbatch=microbatch))


def build_confirmation_config(trial_id: str, *, microbatch: int) -> dict:
    """Fully resolved full-size config for the single transferred winner."""
    return _resolved_sections(campaign.full_model_dict(),
                              campaign.confirmation_optimizer_dict(trial_id),
                              _training_dict(campaign.CONFIRMATION_SCHEDULE, microbatch=microbatch))


def _expected_optimizer_dict(trial_id: str, confirmation: bool) -> dict:
    return (campaign.confirmation_optimizer_dict(trial_id) if confirmation
            else campaign.trial_optimizer_dict(trial_id))


def _fingerprint(payload: dict) -> str:
    from terminus.integrity import fingerprint

    return fingerprint(payload)


def validate_qualification_config(config_path, manifest_path, *, trial_id: str,
                                  confirmation: bool = False, world_size: int = 1) -> dict:
    """Offline contract check for one generated qualification config.

    Rejects unknown trial IDs, any changed fixed setting, wrong world size,
    batch/horizon mismatches and non-external paths. CPU-only: it certifies
    the contract, not native execution.
    """
    from terminus.data import TokenManifest
    from terminus.model import BitNetLlamaConfig
    from terminus.optim import OptimizerConfig
    from terminus.preflight import count_model
    from terminus.train import TrainConfig, load_config, schedule_scale

    config_path = external_path(config_path)
    manifest_path = external_path(manifest_path)
    if trial_id not in campaign.TRIAL_LRS:
        raise ValueError(f"unknown qualification trial: {trial_id!r}")
    if type(world_size) is not int or world_size != campaign.WORLD_SIZE:
        raise ValueError("qualification requires exactly one H100 (world size 1)")
    model, training, optimizer = load_config(config_path)
    manifest = TokenManifest.load(manifest_path)

    expected_model = BitNetLlamaConfig(**(campaign.full_model_dict() if confirmation
                                          else campaign.proxy_model_dict()))
    if model != expected_model:
        raise ValueError("model differs from the reviewed qualification geometry")
    expected_optimizer = OptimizerConfig(**_expected_optimizer_dict(trial_id, confirmation))
    if _fingerprint(asdict(optimizer)) != _fingerprint(asdict(expected_optimizer)):
        raise ValueError("optimizer differs from the frozen trial matrix or fixed settings")
    for key, value in campaign.SELECTED_TRAINING_LIKE.items():
        if getattr(training, key) != value:
            raise ValueError("training semantics differ from the selected dense baseline")
    if (training.cuda_graphs or training.compile or training.phase_start_tokens
            or training.quantization_warmup_tokens or training.quantization_start_strength != 1):
        raise ValueError("qualification requires eager dense execution without a phase/QAT transition")

    schedule = campaign.CONFIRMATION_SCHEDULE if confirmation else campaign.PROXY_SCHEDULE
    if (training.warmup_tokens != schedule["warmup_tokens"]
            or training.decay_tokens != schedule["decay_tokens"]
            or training.schedule != "wsd" or training.decay_style != "cosine"
            or training.final_lr_fraction != 0.1):
        raise ValueError("schedule differs from the token-aligned complete WSD contract")
    total_tokens = manifest.config.usable_train_tokens
    if total_tokens != schedule["total_tokens"]:
        raise ValueError("manifest horizon differs from the exact qualification token budget")
    schedule_scale(0, total_tokens, training.warmup_tokens, training.final_lr_fraction,
                   schedule=training.schedule, decay_tokens=training.decay_tokens,
                   decay_style=training.decay_style)
    if manifest.config.seq_len != model.max_seq_len:
        raise ValueError("manifest sequence length differs from model context")

    cadence = {"checkpoint_interval": campaign.CHECKPOINT_INTERVAL,
               "validation_interval": campaign.VALIDATION_INTERVAL,
               "validation_batches_per_language": campaign.VALIDATION_BATCHES_PER_LANGUAGE,
               "log_interval": campaign.LOG_INTERVAL,
               "stability_log_interval": campaign.STABILITY_LOG_INTERVAL,
               "time_limit_seconds": None}
    for key, value in cadence.items():
        if getattr(training, key) != value:
            raise ValueError(f"frozen qualification cadence differs: {key}")

    micro_tokens = manifest.config.tokens_per_step * world_size
    batch_tokens = micro_tokens * training.gradient_accumulation_steps
    if batch_tokens != campaign.GLOBAL_BATCH_TOKENS:
        raise ValueError("global batch differs from 2,097,152 tokens at world size 1")
    expected_count = (campaign.FULL_PARAMETER_COUNT if confirmation
                      else campaign.PROXY_PARAMETER_COUNT)
    counts = count_model(model)
    if counts["total"] != expected_count:
        raise ValueError(f"parameter count {counts['total']} differs from the contract {expected_count}")

    return {
        "schema": 1,
        "status": "offline_contract_checks_passed",
        "scope": "CPU configuration/arithmetic checks only; not a launch gate",
        "trial_id": trial_id,
        "confirmation": confirmation,
        "identity": {
            "config_sha256": _fingerprint(json.loads(Path(config_path).read_text())),
            "model_sha256": _fingerprint(asdict(model)),
            "training_sha256": _fingerprint(asdict(training)),
            "optimizer_sha256": _fingerprint(asdict(optimizer)),
            "manifest_sha256": _fingerprint(asdict(manifest)),
            "source": campaign.qualification_source_hashes(),
        },
        "parameters": counts,
        "batch": {"world_size": world_size, "microbatch_sequences": manifest.config.batch_size,
                  "accumulation_steps": training.gradient_accumulation_steps,
                  "global_batch_tokens": batch_tokens},
        "schedule": schedule,
        "unverified": [
            "native H100 runtime, kernel and FP8 evidence",
            "data shard/tokenizer bytes and staging receipt freshness",
            "fit, throughput, memory headroom and cadence overhead",
            "durable checkpoint upload/restore and recovery drill",
            "live allocation ledger and billing-unit identity",
            "held-out learning, selection and transfer outcomes",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trial", help="Build the named proxy trial config")
    mode.add_argument("--confirm-from", metavar="TRIAL",
                      help="Build the full-size confirmation config for the selected trial")
    mode.add_argument("--validate", action="store_true",
                      help="Validate an existing config instead of building one")
    parser.add_argument("--config", help="Config path for --validate")
    parser.add_argument("--trial-id", help="Expected trial id for --validate")
    parser.add_argument("--confirmation", action="store_true",
                        help="Validate the confirmation contract for --validate")
    parser.add_argument("--microbatch", type=int,
                        help="Microbatch sequences per rank; must match the manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", help="Fresh external output path when building")
    args = parser.parse_args()
    manifest_path = external_path(args.manifest)

    if args.validate:
        if not args.config or not args.trial_id:
            parser.error("--validate requires --config and --trial-id")
        receipt = validate_qualification_config(args.config, manifest_path,
                                                trial_id=args.trial_id,
                                                confirmation=args.confirmation)
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return 0

    from terminus.data import TokenManifest

    if args.microbatch is None or args.output is None:
        parser.error("building requires --microbatch and --output")
    manifest = TokenManifest.load(manifest_path)
    if manifest.config.batch_size != args.microbatch:
        parser.error("microbatch must equal the manifest's per-rank batch size")
    trial_id = args.trial or args.confirm_from
    config = (build_confirmation_config(trial_id, microbatch=args.microbatch)
              if args.confirm_from else build_trial_config(trial_id, microbatch=args.microbatch))
    output = external_path(args.output, fresh=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n")
    receipt = validate_qualification_config(output, manifest_path, trial_id=trial_id,
                                            confirmation=bool(args.confirm_from))
    print(json.dumps({"output": str(output), "status": receipt["status"],
                      "parameters_total": receipt["parameters"]["total"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
