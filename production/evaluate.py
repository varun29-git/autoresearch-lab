"""Evaluate a receipt-verified checkpoint against frozen, hashed task files."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from production.artifacts import external_path, verified_checkpoint, verify_tokenizer
from terminus.integrity import file_sha256, fingerprint, source_hashes, write_json_atomic
from terminus.model import BitNetLlama, BitNetLlamaConfig
from terminus.task_eval import TaskItem, aggregate, score_items
from terminus.tokenizer import load_tokenizer


def load_tasks(manifest_path, root, *, tasks, include_final=False, max_items=None):
    if max_items is not None and (type(max_items) is not int or max_items <= 0):
        raise ValueError("max_items must be positive")
    manifest = json.loads(Path(manifest_path).read_text())
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("request a nonempty list of unique tasks")
    root = Path(root).resolve()
    items, files, identifiers = [], {}, set()
    for name in tasks:
        entry = manifest["tasks"].get(name)
        if not entry or entry.get("status") != "fetched":
            raise ValueError(f"required task is unavailable: {name}")
        version = entry.get("version") or entry.get("revision")
        if not version or version == "unspecified-gated" or not entry.get("sha256"):
            raise ValueError(f"task needs a version and SHA256: {name}")
        role = entry.get("role")
        if role not in {"development", "final"} or (role == "final" and not include_final):
            raise ValueError(f"task role is invalid or final access was not explicit: {name}")
        path = (root / entry["file"]).resolve()
        if root not in path.parents or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"task path or content digest mismatch: {name}")
        files[name] = {"version": version, "sha256": entry["sha256"], "role": role}
        count = 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                prompt, reference = row["input"], row["output"]
                if not isinstance(prompt, str) or not isinstance(reference, str) or not reference:
                    raise ValueError("task rows require text input and nonempty output")
                item_id = row.get("id") or fingerprint(row)
                if (name, item_id) in identifiers:
                    raise ValueError("duplicate task item identity")
                identifiers.add((name, item_id))
                items.append(TaskItem(task=name, item_id=str(item_id), prompt=prompt,
                                     reference=reference, language=row.get("language", entry.get("language", "en")),
                                     role=role))
                count += 1
                if max_items and count >= max_items:
                    break
        if not count:
            raise ValueError(f"task has no items: {name}")
    return items, files


def evaluate_checkpoint(checkpoint, tokenizer_path, manifest_path, root, output, *,
                        tasks, device="cpu", include_final=False, max_items=None, bundle=None):
    output = external_path(output, fresh=True)
    records_path = external_path(output.with_suffix(".items.jsonl"), fresh=True)
    if bundle is not None:
        if checkpoint is not None or device != "cpu":
            raise ValueError("packed bundle evaluation requires CPU and no checkpoint argument")
        from production.export import load_export
        model, bundled_tokenizer, export_receipt = load_export(bundle)
        tokenizer_sha = export_receipt["files"][bundled_tokenizer.name]
        if tokenizer_path and file_sha256(tokenizer_path) != tokenizer_sha:
            raise ValueError("tokenizer differs from CPU export")
        tokenizer_path = bundled_tokenizer
        config, receipt = model.config, export_receipt["checkpoint"]
    else:
        payload, receipt = verified_checkpoint(checkpoint)
        tokenizer_sha = verify_tokenizer(tokenizer_path, payload["contract"])
        config = BitNetLlamaConfig(**payload["contract"]["model"])
        if device == "cpu" and config.attention_backend != "sdpa_reference":
            raise ValueError("native H100 checkpoints require CUDA evaluation; use the CPU export for deployment evaluation")
        model = BitNetLlama(config)
        model.load_state_dict(payload["model"])
        del payload
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_piece_size() != config.vocab_size:
        raise ValueError("tokenizer vocabulary differs from model")
    items, files = load_tasks(manifest_path, root, tasks=tasks, include_final=include_final, max_items=max_items)
    # Require full examples. Truncation would change the benchmark being scored.
    for item in items:
        if not tokenizer.encode(item.prompt) or len(tokenizer.encode(item.prompt)) + len(tokenizer.encode(item.reference)) > config.max_seq_len + 1:
            raise ValueError(f"task item needs a nonempty prompt and must fit context: {item.item_id}")
    model = model.to(device).eval()
    started = time.monotonic()
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                        enabled=torch.device(device).type == "cuda"):
        records = score_items(model, tokenizer, items, device=torch.device(device), max_seq_len=config.max_seq_len)
    summary = aggregate(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    report = {"schema": 1, "metric": "teacher_forced_reference_nll_not_generated_task_accuracy",
              "evaluator_sha256": file_sha256(__file__),
              "checkpoint": receipt, "model_sha256": fingerprint(config.__dict__),
              "tokenizer_sha256": tokenizer_sha, "eval_manifest_sha256": file_sha256(manifest_path),
              "task_files": files, "max_items_per_task": max_items, "include_final": include_final,
              "summary": summary, "records_path": str(records_path), "records_sha256": file_sha256(records_path),
              "device": device, "wall_seconds": time.monotonic() - started, "source": source_hashes()}
    write_json_atomic(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--checkpoint")
    sources.add_argument("--bundle")
    parser.add_argument("--tokenizer")
    parser.add_argument("--config", help="Optional legacy argument; must match the checkpoint model exactly")
    for name in ("eval-manifest", "eval-root", "output", "tasks"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    if args.checkpoint and not args.tokenizer:
        parser.error("checkpoint evaluation requires --tokenizer")
    if args.config:
        if not args.checkpoint:
            parser.error("--config is only accepted with --checkpoint")
        from terminus.train import load_config
        verified_checkpoint(args.checkpoint, model_config=load_config(args.config)[0])
    report = evaluate_checkpoint(args.checkpoint, args.tokenizer, args.eval_manifest, args.eval_root,
                                 args.output, tasks=args.tasks.split(","), device=args.device,
                                 include_final=args.include_final, max_items=args.max_items, bundle=args.bundle)
    print(json.dumps(report["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
