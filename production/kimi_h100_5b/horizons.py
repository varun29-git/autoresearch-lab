"""Exact-horizon manifest views and the token-accounting launch gate.

Horizon views derive an exact-token-budget manifest from a verified source
manifest without rewriting any shard: tokenizer, shard hashes, source offsets,
mixture and provenance are retained, and the derivation is recorded in
provenance. A phase curriculum is truncated consistently or rejected; it is
never silently flattened.

The token-accounting audit proves, from frozen manifest arithmetic, that the
trainer's nominal update token count equals the actual valid (-100-excluded)
target count for every update of the campaign: the horizon is an exact
multiple of the global batch, so no padded tail microbatch can exist. The
runner additionally checks the trainer's measured valid-target counter at
runtime and fails closed on any mismatch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json

from production.artifacts import external_path

from . import campaign


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _truncate_phases(phases: list, total_tokens: int) -> list:
    """Keep whole phase budgets in order; the final kept phase takes the remainder."""
    kept = []
    remaining = total_tokens
    for phase in phases:
        if remaining <= 0:
            break
        budget = min(phase["tokens"], remaining)
        kept.append({**phase, "tokens": budget})
        remaining -= budget
    if remaining > 0:
        raise ValueError("phase curriculum budgets do not cover the requested horizon")
    return kept


def build_horizon_view(source, *, total_train_tokens: int, microbatch: int):
    """Return a new TokenManifest with an exact usable horizon, no shard rewrites.

    total_train_tokens must be sequence-aligned and no larger than the source
    horizon; microbatch becomes the view's per-rank batch size and must divide
    the global batch evenly at world size 1.
    """
    from terminus.data import TokenManifest

    total_train_tokens = _positive_int("total_train_tokens", total_train_tokens)
    microbatch = _positive_int("microbatch", microbatch)
    config = source.config
    if total_train_tokens % config.seq_len:
        raise ValueError("horizon must align to complete sequences")
    if total_train_tokens > config.usable_train_tokens:
        raise ValueError("horizon cannot exceed the source manifest's usable tokens")
    steps, remainder = divmod(campaign.GLOBAL_BATCH_TOKENS, microbatch * config.seq_len)
    if remainder or steps < 1:
        raise ValueError("microbatch must divide the 2,097,152-token global batch evenly")
    train_tokens = sum(spec.num_tokens for spec in source.shards
                       if spec.split == config.train_split)
    if total_train_tokens > train_tokens:
        raise ValueError("training shards do not cover the requested horizon")

    updates = {"total_train_tokens": total_train_tokens, "batch_size": microbatch}
    if config.phase_source_mixes:
        updates["phase_source_mixes"] = _truncate_phases(
            [dict(phase) for phase in config.phase_source_mixes], total_train_tokens)
    if config.cosmopedia_stories_prefix_tokens >= total_train_tokens:
        raise ValueError("prefix curriculum consumes the entire requested horizon")
    view_config = replace(config, **updates)  # dataclass validation re-runs here

    from terminus.integrity import fingerprint

    provenance = dict(source.provenance)
    provenance["horizon_view"] = {
        "parent_manifest_sha256": fingerprint(asdict(source)),
        "total_train_tokens": total_train_tokens,
        "microbatch_sequences_per_rank": microbatch,
        "accumulation_steps_at_world_size_1": steps,
        "created_by": "production.kimi_h100_5b.horizons",
        "shards_unchanged": True,
    }
    return TokenManifest(config=view_config, shards=list(source.shards),
                         tokenizer=dict(source.tokenizer), provenance=provenance)


def audit_token_accounting(manifest, *, global_batch_tokens: int = campaign.GLOBAL_BATCH_TOKENS,
                           world_size: int = campaign.WORLD_SIZE) -> dict:
    """Prove nominal update tokens equal actual valid targets for every update.

    With continuous EOS packing the only ignored (-100) targets come from the
    trainer's padded tail microbatch. An exactly aligned horizon has no tail,
    so every update must deliver exactly global_batch_tokens valid targets.
    """
    global_batch_tokens = _positive_int("global_batch_tokens", global_batch_tokens)
    world_size = _positive_int("world_size", world_size)
    config = manifest.config
    if config.total_train_tokens != config.usable_train_tokens:
        raise ValueError("horizon is not sequence-aligned; a padded tail would exist")
    micro_tokens = config.tokens_per_step * world_size
    steps, remainder = divmod(global_batch_tokens, micro_tokens)
    if remainder or steps < 1:
        raise ValueError("global batch is not a positive multiple of microstep tokens")
    updates, tail = divmod(config.usable_train_tokens, global_batch_tokens)
    if tail:
        raise ValueError("horizon is not an exact multiple of the global batch")
    return {
        "schema": 1,
        "status": "aligned_complete_updates",
        "total_tokens": config.usable_train_tokens,
        "global_batch_tokens": global_batch_tokens,
        "micro_tokens_per_step": micro_tokens,
        "accumulation_steps": steps,
        "total_updates": updates,
        "valid_targets_per_update": global_batch_tokens,
        "padded_tail_targets": 0,
        "runtime_check": "runner compares the trainer's measured valid-target "
                         "counter against update_tokens every update and fails closed",
    }


def main() -> int:
    from terminus.data import TokenManifest
    from terminus.integrity import write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    view = sub.add_parser("view", help="Write an exact-horizon manifest view")
    view.add_argument("--source-manifest", required=True)
    view.add_argument("--total-tokens", type=int, required=True)
    view.add_argument("--microbatch", type=int, required=True)
    view.add_argument("--output", required=True)
    audit = sub.add_parser("audit", help="Run the token-accounting launch gate")
    audit.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.command == "view":
        source = TokenManifest.load(external_path(args.source_manifest))
        manifest = build_horizon_view(source, total_train_tokens=args.total_tokens,
                                      microbatch=args.microbatch)
        output = external_path(args.output, fresh=True)
        write_json_atomic(output, asdict(manifest))
        receipt = audit_token_accounting(TokenManifest.load(output))
        print(json.dumps({"output": str(output), **receipt}, indent=2, sort_keys=True))
    else:
        manifest = TokenManifest.load(external_path(args.manifest))
        print(json.dumps(audit_token_accounting(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
