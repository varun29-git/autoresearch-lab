"""Score frozen uint16 validation cells without re-encoding or padded targets."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from production.artifacts import external_path, verified_checkpoint
from production.validation_data import verify, weighted_summary
from terminus.integrity import file_sha256, write_json_atomic
from terminus.model import BitNetLlama, BitNetLlamaConfig


@torch.no_grad()
def score_cells(model, bundle, *, role="development", include_final=False,
                device="cpu", batch_size=1, max_sequences=None, weights=None):
    if role not in ("development", "final") or (role == "final" and not include_final):
        raise ValueError("final validation access must be explicit")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("invalid evaluation batch size")
    if max_sequences is not None and (type(max_sequences) is not int or max_sequences <= 0):
        raise ValueError("invalid sequence limit")
    root, manifest = external_path(bundle), verify(bundle)
    length = manifest["policy"]["sequence_length"]
    entries = [(name, meta) for name, meta in manifest["files"].items() if meta["role"] == role]
    if not entries:
        raise ValueError("validation role has no data")
    if weights is not None and {m["cell"] for _, m in entries} != set(weights):
        raise ValueError("evaluation coverage differs from the frozen contract")
    records = {}
    was_training = model.training
    model.eval()
    try:
        for name, meta in sorted(entries, key=lambda v: v[1]["cell"]):
            values = np.memmap(root / name, mode="r", dtype="<u2")
            sequences = (len(values) - 1) // length
            if max_sequences is not None:
                sequences = min(sequences, max_sequences)
            losses, tokens = 0., 0
            for start in range(0, sequences, batch_size):
                count = min(batch_size, sequences - start)
                # Every sequence has L inputs and L already-shifted targets.
                # Adjacent blocks share one token, never one scored target.
                blocks = np.stack([values[i * length:i * length + length + 1]
                                   for i in range(start, start + count)]).astype(np.int64)
                x = torch.tensor(blocks[:, :-1], device=device)
                y = torch.tensor(blocks[:, 1:], device=device)
                result = model(x, targets=y, return_logits=False)
                actual_tokens = int(result["loss_tokens"])
                loss = float(result["loss_sum"].double())
                if actual_tokens != count * length or not np.isfinite(loss):
                    raise ValueError("invalid validation loss or target count")
                losses += loss
                tokens += actual_tokens
            if not tokens or meta["cell"] in records:
                raise ValueError("empty or duplicate evaluation cell")
            records[meta["cell"]] = {"loss": losses / tokens, "loss_sum": losses,
                "tokens": tokens, "sequences": sequences, "language": meta["language"],
                "parent": meta["parent"], "file_sha256": meta["sha256"]}
            parent, identity = meta["parent"], meta["cell"].split("/")
            if meta["language"] == "en":
                variant = "english"
            elif parent.startswith("cc-"):
                variant = "translated_web"
            elif parent == "specialized":
                variant = ("latin" if "latn" in identity else "native") + "_specialized"
            else:
                variant = "latin_web" if parent == "fineweb2-latn" else "native_web"
            records[meta["cell"]]["variant"] = variant
    finally:
        model.train(was_training)
    groups = {}
    for key in ("language", "parent", "variant"):
        groups[key] = {}
        for record in records.values():
            group = groups[key].setdefault(record[key], {"loss_sum": 0., "tokens": 0})
            group["loss_sum"] += record["loss_sum"]
            group["tokens"] += record["tokens"]
        for group in groups[key].values():
            group["loss"] = group["loss_sum"] / group["tokens"]
    summary = weighted_summary(records, weights) if weights is not None else {"per_cell": records}
    summary.update({"sample_token_weighted_nll": sum(r["loss_sum"] for r in records.values()) /
                    sum(r["tokens"] for r in records.values()),
                    "sample_token_weighted_groups": groups,
                    "complete_contract_score": weights is not None})
    if weights is not None:
        summary["target_weighted_groups"] = {}
        for key in ("language", "parent", "variant"):
            summary["target_weighted_groups"][key] = {}
            for name in {r[key] for r in records.values()}:
                selected = [cell for cell, r in records.items() if r[key] == name]
                mass = sum(weights[cell] for cell in selected)
                summary["target_weighted_groups"][key][name] = {
                    "target_weight": mass,
                    "loss": sum(weights[cell] * records[cell]["loss"] for cell in selected) / mass}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "validation-bundle", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--weights", help="Frozen JSON mapping from cell ID to positive weight")
    parser.add_argument("--role", choices=("development", "final"), default="development")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()
    output = external_path(args.output, fresh=True)
    manifest = verify(args.validation_bundle)
    payload, receipt = verified_checkpoint(args.checkpoint)
    config = BitNetLlamaConfig(**payload["contract"]["model"])
    tokenizer_sha = (payload["contract"]["data"].get("tokenizer") or {}).get("sha256")
    if tokenizer_sha not in manifest["tokenizer_lineages"]:
        raise ValueError("checkpoint tokenizer is outside the frozen corpus lineage")
    if config.max_seq_len < manifest["policy"]["sequence_length"]:
        raise ValueError("checkpoint context is shorter than validation sequences")
    if args.device == "cpu" and config.attention_backend != "sdpa_reference":
        raise ValueError("native checkpoints require native device evaluation")
    model = BitNetLlama(config)
    model.load_state_dict(payload["model"])
    del payload
    model = model.to(args.device)
    weights = json.loads(external_path(args.weights).read_text()) if args.weights else None
    started = time.monotonic()
    with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                        enabled=torch.device(args.device).type == "cuda"):
        summary = score_cells(model, args.validation_bundle, role=args.role,
            include_final=args.include_final, device=args.device, batch_size=args.batch_size,
            max_sequences=args.max_sequences, weights=weights)
    report = {"schema": 1, "metric": "heldout_next_token_nll_not_generated_accuracy",
        "validation_manifest_sha256": manifest["manifest_sha256"], "checkpoint": receipt,
        "tokenizer_sha256": tokenizer_sha, "role": args.role,
        "max_sequences": args.max_sequences, "weights_sha256": file_sha256(args.weights) if args.weights else None,
        "evaluator_sha256": file_sha256(__file__), "wall_seconds": time.monotonic() - started,
        "summary": summary}
    write_json_atomic(output, report)
    print(json.dumps({"output": str(output), "complete_contract_score": weights is not None}))


if __name__ == "__main__":
    main()
