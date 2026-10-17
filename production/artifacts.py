"""Shared identity and output rules for production commands."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from terminus.checkpoint import load_checkpoint
from terminus.integrity import file_sha256, fingerprint
from terminus.model import BitNetLlamaConfig


def external_path(path, *, fresh=False):
    path = Path(path).resolve()
    repository = Path(__file__).resolve().parents[1]
    if path == repository or repository in path.parents:
        raise ValueError("production artifacts must be outside the repository")
    if fresh and path.exists():
        raise FileExistsError(path)
    return path


def verified_checkpoint(source, *, model_config=None):
    source = Path(source).resolve()
    directory = source if source.is_dir() else source.parent
    run = json.loads((directory.parent / "run.json").read_text())
    contract = run["contract"]
    if run["contract_sha256"] != fingerprint(contract):
        raise ValueError("run contract digest mismatch")
    actual = BitNetLlamaConfig(**contract["model"])
    if model_config is not None and asdict(actual) != asdict(model_config):
        raise ValueError("evaluation/export model differs from checkpoint contract")
    # Pin latest to an exact file before loading, so concurrent saves cannot
    # associate a payload with a different receipt.
    receipt_path = directory / "latest.json" if source.is_dir() else source.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    name = receipt["file"]
    if Path(name).name != name:
        raise ValueError("invalid checkpoint filename")
    exact = directory / name
    if not source.is_dir() and exact != source:
        raise ValueError("checkpoint filename differs from receipt")
    payload = load_checkpoint(exact, contract)
    committed = json.loads(exact.with_suffix(".json").read_text())
    if committed != receipt:
        raise ValueError("checkpoint receipts disagree")
    return payload, receipt


def verify_tokenizer(path, contract):
    expected = (contract["data"].get("tokenizer") or {}).get("sha256")
    if not expected or file_sha256(path) != expected:
        raise ValueError("tokenizer does not match the checkpoint contract")
    return expected
