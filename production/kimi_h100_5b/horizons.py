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
from pathlib import Path

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


def reserve_val_holdout(source, *, data_root, output_dir, fraction: float = 0.75) -> dict:
    """Split each validation shard into physical prefix and reserved-suffix files.

    The originals are never modified. The prefix shards enter trial manifests
    (quick/boundary evaluations read only these); the suffix shards form the
    reserved final holdout, consulted once at the declared selection step.
    Both are ordinary shard files with their own hashes so the loader's
    file-size integrity check holds. The reservation links every derived file
    to its parent file and parent hash.
    """
    import hashlib
    import os
    import uuid

    import numpy as np

    from terminus.data import DTYPE_BY_NAME, TokenShardSpec
    from terminus.integrity import file_sha256

    if isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = source.config
    dtype = np.dtype(DTYPE_BY_NAME[config.token_dtype])
    prefix_specs, holdout_specs, reservation = [], [], []
    for spec in source.shards:
        if spec.split != config.val_split:
            continue
        original = data_root / str(spec.path)
        tokens = np.memmap(original, dtype=dtype, mode="r")
        if len(tokens) != spec.num_tokens:
            raise ValueError(f"validation shard size differs from its manifest: {spec.path}")
        boundary = int(spec.num_tokens * fraction) // config.seq_len * config.seq_len
        if boundary <= config.seq_len or spec.num_tokens - boundary <= config.seq_len:
            raise ValueError(f"reservation split leaves no full sequence: {spec.path}")
        parent_sha = spec.sha256 or file_sha256(original)
        parts = {}
        for label, view in (("prefix", tokens[:boundary]), ("holdout", tokens[boundary:])):
            name = f"{Path(spec.path).stem}.{label}_{int(fraction * 100):02d}{uuid.uuid4().hex[:8]}.bin"
            target = output_dir / name
            if target.exists():
                raise FileExistsError(target)
            digest = hashlib.sha256()
            with target.open("xb") as handle:
                for offset in range(0, len(view), 1 << 22):
                    chunk = np.asarray(view[offset:offset + (1 << 22)]).tobytes()
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            parts[label] = TokenShardSpec(
                path=str(target.relative_to(data_root)) if data_root in target.resolve().parents
                else str(target),
                source=spec.source, split=config.val_split, num_tokens=len(view),
                dtype=spec.dtype, sha256=digest.hexdigest(), language=spec.language,
                dataset_name=spec.dataset_name, dataset_config=spec.dataset_config,
                source_split=spec.source_split)
        prefix_specs.append(parts["prefix"])
        holdout_specs.append(parts["holdout"])
        reservation.append({"parent_path": str(spec.path), "parent_sha256": parent_sha,
                            "parent_tokens": spec.num_tokens,
                            "prefix": {"path": parts["prefix"].path,
                                       "tokens": parts["prefix"].num_tokens,
                                       "sha256": parts["prefix"].sha256},
                            "holdout": {"path": parts["holdout"].path,
                                        "tokens": parts["holdout"].num_tokens,
                                        "sha256": parts["holdout"].sha256},
                            "fraction": fraction})
    if not reservation:
        raise ValueError("source manifest has no validation shards to split")
    return {"prefix": prefix_specs, "holdout": holdout_specs, "reservation": reservation}


def build_horizon_view(source, *, total_train_tokens: int, microbatch: int,
                       val_shards=None, val_reservation=None):
    """Return a new TokenManifest with an exact usable horizon, no shard rewrites.

    total_train_tokens must be sequence-aligned and no larger than the source
    horizon; microbatch becomes the view's per-rank batch size and must divide
    the global batch evenly at world size 1. With val_shards (from
    reserve_val_holdout), the source's validation specs are replaced by the
    prefix specs and the reservation is recorded in provenance.
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

    if val_shards is None:
        shards = list(source.shards)
    else:
        shards = [spec for spec in source.shards if spec.split != config.val_split]
        shards.extend(val_shards)

    from terminus.integrity import fingerprint

    provenance = dict(source.provenance)
    provenance["horizon_view"] = {
        "parent_manifest_sha256": fingerprint(asdict(source)),
        "total_train_tokens": total_train_tokens,
        "microbatch_sequences_per_rank": microbatch,
        "accumulation_steps_at_world_size_1": steps,
        "created_by": "production.kimi_h100_5b.horizons",
        "shards_unchanged": val_shards is None,
    }
    if val_reservation is not None:
        provenance["horizon_view"]["val_reservation"] = val_reservation
    return TokenManifest(config=view_config, shards=shards,
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
    reserve = sub.add_parser("reserve", help="Split val shards into prefix and reserved holdout")
    reserve.add_argument("--source-manifest", required=True)
    reserve.add_argument("--data-root", required=True)
    reserve.add_argument("--output-dir", required=True)
    reserve.add_argument("--fraction", type=float, default=0.75)
    reserve.add_argument("--receipt", required=True)
    view = sub.add_parser("view", help="Write an exact-horizon manifest view")
    view.add_argument("--source-manifest", required=True)
    view.add_argument("--total-tokens", type=int, required=True)
    view.add_argument("--microbatch", type=int, required=True)
    view.add_argument("--val-receipt", help="Reservation receipt from 'reserve'")
    view.add_argument("--val-set", choices=("prefix", "holdout"), default="prefix")
    view.add_argument("--output", required=True)
    audit = sub.add_parser("audit", help="Run the token-accounting launch gate")
    audit.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.command == "reserve":
        from dataclasses import asdict as _asdict

        source = TokenManifest.load(external_path(args.source_manifest))
        result = reserve_val_holdout(source, data_root=external_path(args.data_root),
                                     output_dir=external_path(args.output_dir),
                                     fraction=args.fraction)
        receipt = {"schema": 1,
                   "prefix": [_asdict(spec) for spec in result["prefix"]],
                   "holdout": [_asdict(spec) for spec in result["holdout"]],
                   "reservation": result["reservation"]}
        write_json_atomic(external_path(args.receipt, fresh=True), receipt)
        print(json.dumps({"status": "reserved", "shards": len(result["reservation"]),
                          "receipt": args.receipt}, sort_keys=True))
        return 0

    if args.command == "view":
        source = TokenManifest.load(external_path(args.source_manifest))
        val_shards, val_reservation = None, None
        if args.val_receipt:
            from terminus.data import TokenShardSpec

            receipt = json.loads(Path(external_path(args.val_receipt)).read_text())
            val_shards = [TokenShardSpec(**spec) for spec in receipt[args.val_set]]
            val_reservation = receipt["reservation"]
        manifest = build_horizon_view(source, total_train_tokens=args.total_tokens,
                                      microbatch=args.microbatch, val_shards=val_shards,
                                      val_reservation=val_reservation)
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
