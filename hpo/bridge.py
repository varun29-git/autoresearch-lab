"""Create a separately locked English width bridge from the 200M stream."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

from terminus.data import TokenManifest

from .loop import DEFAULT_REFERENCE, ROOT, load_reference, sha256


WIDTHS = {1536: (12, 4, 3840, "700m"),
          2304: (18, 6, 5760, "1p5b")}
EXAMPLES_PER_UPDATE = 96
HELDOUT_EXAMPLES = 32


def bridge_reference(base: dict, *, width: int, microbatch: int,
                     manifest_sha256: str) -> dict:
    if width not in WIDTHS:
        raise ValueError("width bridge must be 1536 or 2304")
    if (type(microbatch) is not int or microbatch <= 0
            or EXAMPLES_PER_UPDATE % microbatch
            or HELDOUT_EXAMPLES % microbatch):
        raise ValueError("microbatch must divide both 96 training and 32 validation examples")
    if base["config"]["model"]["dim"] != 768:
        raise ValueError("width bridge needs the fixed 768-width reference")
    result = copy.deepcopy(base)
    heads, kv_heads, ffn_dim, size_label = WIDTHS[width]
    result["campaign_id"] = f"stack4-hpo-{size_label}-en-v1"
    result["manifest_sha256"] = manifest_sha256
    model = result["config"]["model"]
    model.update(dim=width, n_heads=heads, n_kv_heads=kv_heads, ffn_dim=ffn_dim)
    training = result["config"]["training"]
    training["gradient_accumulation_steps"] = EXAMPLES_PER_UPDATE // microbatch
    training["validation_batches_per_language"] = HELDOUT_EXAMPLES // microbatch
    return result


def prepare(args: argparse.Namespace) -> dict:
    base = load_reference(args.reference)
    if sha256(args.manifest) != base["manifest_sha256"]:
        raise ValueError("base manifest differs from the 200M fixed stream")
    output_dir = args.output_dir.resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError("derived manifests and references must live outside Git")
    manifest = TokenManifest.load(args.manifest)
    if manifest.config.batch_size != 32 or manifest.config.seq_len != 4096:
        raise ValueError("the base English stream must have batch 32 and context 4096")
    # Reject an invalid bridge before writing either output file.
    bridge_reference(base, width=args.width, microbatch=args.microbatch,
                     manifest_sha256="0" * 64)
    derived = TokenManifest(replace(manifest.config, batch_size=args.microbatch),
                            manifest.shards, manifest.tokenizer, manifest.provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest.width{args.width}.batch{args.microbatch}.json"
    reference_path = output_dir / f"reference.width{args.width}.batch{args.microbatch}.json"
    if manifest_path.exists() or reference_path.exists():
        raise FileExistsError("bridge already exists; do not overwrite a campaign stream")
    manifest_path.write_text(json.dumps(asdict(derived), indent=2, sort_keys=True) + "\n")
    reference = bridge_reference(base, width=args.width, microbatch=args.microbatch,
                                 manifest_sha256=sha256(manifest_path))
    reference_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    return {"manifest": str(manifest_path), "reference": str(reference_path),
            "manifest_sha256": sha256(manifest_path),
            "tokens_per_update": args.microbatch * 4096
                                 * reference["config"]["training"]["gradient_accumulation_steps"],
            "validation_examples": args.microbatch
                                   * reference["config"]["training"]["validation_batches_per_language"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=sorted(WIDTHS), required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
