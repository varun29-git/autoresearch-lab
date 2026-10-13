"""Build the selected 1.5B dense config for an explicit production manifest.

This prepares a config; it does not launch training or choose the data,
global batch, or long-horizon learning-rate schedule for the operator.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from hpo.loop import DEFAULT_REFERENCE, ROOT, load_reference
from terminus.data import TokenManifest
from terminus.train import load_config


SELECTED_OPTIMIZER = {
    "kind": "muon",
    "muon_lr_rule": "spectral_unclamped",
    "muon_fan_in_base": None,
    "muon_lr": 0.017,
    "adamw_lr": 0.003,
    "norm_lr": 0.0003,
    "qk_norm_lr": 0.0001,
    "weight_decay": 0.02,
    "embedding_weight_decay": 0.0,
}


def build_config(*, manifest: TokenManifest, world_size: int,
                 global_batch_tokens: int, warmup_tokens: int,
                 decay_tokens: int) -> dict:
    """Return the model/optimizer choice with explicit run-dependent settings."""
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if manifest.config.seq_len != 4096:
        raise ValueError("selected dense model requires a 4096-token manifest")
    microstep_tokens = manifest.config.batch_size * 4096 * world_size
    if (type(global_batch_tokens) is not int or global_batch_tokens < microstep_tokens
            or global_batch_tokens % microstep_tokens):
        raise ValueError("global_batch_tokens must be a positive multiple of all-rank microstep tokens")
    total_tokens = manifest.config.usable_train_tokens
    if (type(warmup_tokens) is not int or warmup_tokens < 0
            or type(decay_tokens) is not int or decay_tokens <= 0
            or warmup_tokens + decay_tokens > total_tokens):
        raise ValueError("warmup and decay must fit the manifest's usable token budget")

    config = copy.deepcopy(load_reference(DEFAULT_REFERENCE)["config"])
    model = config["model"]
    if (model["n_layers"], model["dim"], model["ffn_dim"], model["n_heads"],
            model["n_kv_heads"], model["branch_norm"], model["value_residual"],
            model["weight_mode"]) != (25, 768, 1920, 6, 2, "peri", "learned", "dense"):
        raise ValueError("the selected Stack 4 reference changed")
    model.update(dim=2304, ffn_dim=5760, n_heads=18, n_kv_heads=6)
    config["optimizer"] = SELECTED_OPTIMIZER.copy()
    training = config["training"]
    training.update(gradient_accumulation_steps=global_batch_tokens // microstep_tokens,
                    warmup_tokens=warmup_tokens, decay_tokens=decay_tokens,
                    checkpoint_interval=1000, validation_interval=1000,
                    validation_batches_per_language=8,
                    log_interval=10, stability_log_interval=10,
                    time_limit_seconds=None)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--global-batch-tokens", type=int, required=True)
    parser.add_argument("--warmup-tokens", type=int, required=True)
    parser.add_argument("--decay-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("put production run configs outside this Git repository")
    manifest = TokenManifest.load(args.manifest)
    config = build_config(manifest=manifest, world_size=args.world_size,
                          global_batch_tokens=args.global_batch_tokens,
                          warmup_tokens=args.warmup_tokens, decay_tokens=args.decay_tokens)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    load_config(output)
    print(json.dumps({"output": str(output),
                      "global_batch_tokens": args.global_batch_tokens,
                      "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
                      "usable_train_tokens": manifest.config.usable_train_tokens}, sort_keys=True))


if __name__ == "__main__":
    main()
