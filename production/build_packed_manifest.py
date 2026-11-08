"""Seal ordered, completed packed candidates against a frozen local artifact catalog.

CPU preparation only. Tokenizer lineage acceptance and heldout provenance must
be supplied by the source audit, not inferred from uint16 storage or vocabulary.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from production.artifacts import external_path
from terminus.data import TokenManifest, TokenShardSpec
from terminus.integrity import validate_manifest
from terminus.train import load_config


def build_manifest(base, train, validation, paths, *, evaluation_keys=None, tokens_before=0):
    if not base.tokenizer:
        raise ValueError("a frozen tokenizer contract is required")
    keys = evaluation_keys or {}
    shards = []
    # Copy candidates so adding evaluation metadata never edits their receipts.
    train, validation = json.loads(json.dumps([train, validation]))
    for split, phases in ((base.config.train_split, train), (base.config.val_split, validation)):
        for phase in phases:
            if phase.get("tokenizer_sha256") != base.tokenizer["sha256"]:
                raise ValueError("source acceptance must bind each phase to the frozen tokenizer")
            for shard in phase["shards"]:
                uri = shard["payload"]["uri"]
                if uri in keys:
                    shard["evaluation_key"] = keys[uri]
                shards.append(TokenShardSpec(path=paths[uri], source=shard["leaf"], split=split,
                    num_tokens=shard["sequences"]*base.config.seq_len, dtype="uint16",
                    sha256=shard["payload"]["sha256"], language=shard["leaf"].split("/")[0]))
    slots = sum(p["total_sequences"] for p in train)*base.config.seq_len
    used = {s[name]["uri"] for phases in (train, validation) for phase in phases
            for s in phase["shards"] for name in ("payload", "index", "index_offsets")}
    paths = {uri: paths[uri] for uri in sorted(used)}
    config = replace(base.config, total_train_tokens=slots, source_mix={}, phase_source_mixes=[],
                     source_start_examples={}, prefix_source="", cosmopedia_stories_prefix_tokens=0)
    provenance = base.provenance | {"packing": "document_segments_v1", "packed_stream":
        dict(schema=1, train=train, val=validation, paths=paths, tokens_before=tokens_before)}
    return TokenManifest(config, shards, base.tokenizer.copy(), provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-manifest", "paths", "config", "data-root", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--train", nargs="+", required=True, help="ordered accepted candidate JSON paths")
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--evaluation-keys", help="optional artifact URI to language/domain key JSON")
    p.add_argument("--tokens-before", type=int, default=0, help="actual dense checkpoint labels before a separate QAT stream")
    args = p.parse_args()
    read = lambda path: json.loads(external_path(path).read_text())
    manifest = build_manifest(TokenManifest.load(external_path(args.base_manifest)),
        [read(path) for path in args.train], [read(path) for path in args.validation], read(args.paths),
        evaluation_keys=read(args.evaluation_keys) if args.evaluation_keys else None, tokens_before=args.tokens_before)
    model, training, _ = load_config(external_path(args.config))
    report = validate_manifest(manifest, external_path(args.data_root), model,
                               development_mode=training.development_mode)
    output = external_path(args.output, fresh=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(output)
    print(json.dumps({"status": "cpu_verified", "manifest": str(output),
                      "input_slots": report["input_slots"], "scored_targets": report["scored_train_tokens"],
                      "native_training_accepted": False}))


if __name__ == "__main__":
    main()
