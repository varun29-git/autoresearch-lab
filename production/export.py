"""Export a verified QAT checkpoint to a portable two-bit CPU bundle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from production.artifacts import external_path, verified_checkpoint, verify_tokenizer
from terminus.integrity import file_sha256, fsync_directory, source_hashes, write_json_atomic
from terminus.model import BitNetLlamaConfig
from terminus.packed_cpu import pack_model_state, load_packed_state


def export_checkpoint(checkpoint, tokenizer, output):
    output = external_path(output, fresh=True)
    payload, checkpoint_receipt = verified_checkpoint(checkpoint)
    tokenizer_sha = verify_tokenizer(tokenizer, payload["contract"])
    config = BitNetLlamaConfig(**payload["contract"]["model"])
    phase_start = payload["contract"].get("training", {}).get("phase_start_tokens") or 0
    if payload["state"]["tokens_seen"] <= phase_start:
        raise ValueError("QAT checkpoint has no training tokens after its phase transition")
    bundle = pack_model_state(config, payload["model"])
    bundle["checkpoint_sha256"] = checkpoint_receipt["sha256"]
    del payload
    # Validate all tensor names, shapes, scales and codes before committing.
    load_packed_state(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ternary-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        model_path = staging / "model.pt"
        with model_path.open("xb") as handle:
            torch.save(bundle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        tokenizer_name = "tokenizer" + Path(tokenizer).suffix
        shutil.copyfile(tokenizer, staging / tokenizer_name)
        with (staging / tokenizer_name).open("rb") as handle:
            os.fsync(handle.fileno())
        if file_sha256(staging / tokenizer_name) != tokenizer_sha:
            raise ValueError("tokenizer changed during export")
        receipt = {"schema": 1, "format": bundle["format"], "checkpoint": checkpoint_receipt,
                   "files": {"model.pt": file_sha256(model_path), tokenizer_name: tokenizer_sha},
                   "tokenizer": tokenizer_name, "model_bytes": model_path.stat().st_size,
                   "matrix_storage_bits": 2, "scale_dtype": "float32",
                   "unquantized_dtype": "preserved_from_checkpoint", "source": source_hashes(),
                   "validation": "tensor structure and ternary codes verified on CPU",
                   "unmeasured": ["native H100 versus CPU output/quality comparison",
                                  "production-size edge latency and memory"]}
        write_json_atomic(staging / "export.json", receipt)
        fsync_directory(staging)
        staging.rename(output)
        fsync_directory(output.parent)
    return receipt


def load_export(path, *, tile_rows=128):
    path = Path(path)
    receipt = json.loads((path / "export.json").read_text())
    if receipt.get("schema") != 1 or receipt.get("format") != "terminus-ternary-2bit-lsb-v1":
        raise ValueError("unknown CPU export schema")
    tokenizer_name = receipt["tokenizer"]
    if set(receipt["files"]) != {"model.pt", tokenizer_name}:
        raise ValueError("export file inventory mismatch")
    for name, digest in receipt["files"].items():
        if Path(name).name != name or file_sha256(path / name) != digest:
            raise ValueError("export content digest mismatch")
    bundle = torch.load(path / "model.pt", map_location="cpu", mmap=True, weights_only=True)
    if bundle["checkpoint_sha256"] != receipt["checkpoint"]["sha256"]:
        raise ValueError("export checkpoint identity mismatch")
    return load_packed_state(bundle, tile_rows=tile_rows), path / tokenizer_name, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.tokenizer, args.output), indent=2))


if __name__ == "__main__":
    main()
