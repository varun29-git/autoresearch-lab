"""Rewrite a finished four-GPU checkpoint for the eight-GPU continuation.

The selected global batch stays 2,097,152 tokens. Doubling the ranks halves
accumulation. The selected four-GPU prefix is microbatch 32 and accumulation 4,
so the eight-GPU continuation is accumulation 2. Weights,
optimizer moments, and tokens already scored are copied. The microstep counter
is halved so the next sequence is the same one. Ranks 4–7 receive a fresh
generator seeded as ``seed + rank``.

Run this on the machine that will launch the eight GPUs, after the four-GPU
job has stopped on a finished update. Write a new checkpoint directory and
launch the eight-GPU job in a new output directory. The allocation record for
that launch says 8 GPUs. This module does not start training.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .checkpoint import SCHEMA_VERSION, commit_checkpoint_payload, rng_state, restore_rng_state
from .integrity import file_sha256


SOURCE_WORLD = 4
TARGET_WORLD = 8


def _read_payload(path):
    path = Path(path)
    if path.is_dir():
        receipt = json.loads((path / "latest.json").read_text())
        name = receipt["file"]
        if Path(name).name != name:
            raise ValueError("invalid checkpoint path in receipt")
        path = path / name
    else:
        receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["size_bytes"] != path.stat().st_size or receipt["sha256"] != file_sha256(path):
        raise ValueError("checkpoint integrity verification failed")
    payload = torch.load(os.fspath(path), map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("state") != receipt["state"]:
        raise ValueError("checkpoint schema or counters disagree with the committed receipt")
    return payload


def _fresh_rank_rng(seed, rank):
    """Match a fresh rank's torch seed without building the model."""
    previous = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed + rank)
        return rng_state()
    finally:
        restore_rng_state(previous)


def widen_payload(payload):
    """Return a new payload. The caller's checkpoint object is left unchanged."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    contract = payload.get("contract")
    state = payload.get("state")
    rank_rng = payload.get("rank_rng")
    if not isinstance(contract, dict) or not isinstance(state, dict) or not isinstance(rank_rng, list):
        raise ValueError("checkpoint is missing contract, counters, or rank generators")
    if contract.get("world_size") != SOURCE_WORLD:
        raise ValueError("only a four-rank checkpoint can be widened to eight ranks")
    training = contract.get("training")
    accumulation = training.get("gradient_accumulation_steps") if isinstance(training, dict) else None
    if type(accumulation) is not int or accumulation <= 0 or accumulation % 2:
        raise ValueError("four-rank accumulation must be a positive even number so eight ranks keep the same global batch")
    seed = training.get("seed")
    if type(seed) is not int:
        raise ValueError("training seed must be an integer")
    updates = state.get("updates")
    microstep = state.get("next_micro_step")
    if type(updates) is not int or type(microstep) is not int or updates <= 0 or microstep != updates * accumulation:
        raise ValueError("widen only a checkpoint that stopped on a finished update")
    if len(rank_rng) != SOURCE_WORLD:
        raise ValueError("checkpoint rank generators do not match four ranks")
    if "model" not in payload or "optimizer" not in payload:
        raise ValueError("checkpoint is missing weights or optimizer moments")
    widened = copy.deepcopy(contract)
    widened["world_size"] = TARGET_WORLD
    widened["training"]["gradient_accumulation_steps"] = accumulation // 2
    new_state = dict(state)
    new_state["next_micro_step"] = updates * (accumulation // 2)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": widened,
        "state": new_state,
        "model": payload["model"],
        "optimizer": payload["optimizer"],
        "rank_rng": list(rank_rng) + [_fresh_rank_rng(seed, rank) for rank in range(SOURCE_WORLD, TARGET_WORLD)],
    }


def widen_checkpoint(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to write into a nonempty checkpoint directory")
    source_path, output_path = source.resolve(), output.resolve()
    if (output_path == source_path or source_path in output_path.parents
            or output_path in source_path.parents):
        raise ValueError("write the widened checkpoint to a new directory")
    payload = widen_payload(_read_payload(source))
    path = commit_checkpoint_payload(output, payload)
    return path, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Four-GPU checkpoint file or directory")
    parser.add_argument("--output", required=True, help="New directory for the eight-GPU checkpoint")
    args = parser.parse_args(argv)
    path, payload = widen_checkpoint(args.checkpoint, args.output)
    state = payload["state"]
    print(json.dumps({
        "file": str(path),
        "world_size": payload["contract"]["world_size"],
        "gradient_accumulation_steps": payload["contract"]["training"]["gradient_accumulation_steps"],
        "updates": state["updates"],
        "next_micro_step": state["next_micro_step"],
        "tokens_seen": state["tokens_seen"],
        "rank_rng": len(payload["rank_rng"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
