"""Protected checkpoint evaluation on fixed held-out streams and answer-only tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .protocol import policy, read_json, resolve_from, validate_task_manifest, write_json


@torch.inference_mode()
def heldout_batches(model, manifest, data_root: Path, device: torch.device,
                    max_batches: int) -> dict:
    """Use the trainer's fixed spaced-block selection, retaining paired batches."""
    from terminus.data import TokenSpan
    from terminus.integrity import manifest_scope

    domains = set(manifest_scope(manifest)["domains"])
    key_of = lambda spec: spec.source if spec.source in domains else spec.language
    result = {}
    for domain in sorted({key_of(spec) for spec in manifest.shards_for(split=manifest.config.val_split)}):
        specs = [spec for spec in manifest.shards_for(split=manifest.config.val_split)
                 if key_of(spec) == domain]
        span = TokenSpan(specs, root=data_root)
        blocks = (span.num_tokens - 1) // manifest.config.seq_len
        selected = min(blocks, max_batches * manifest.config.batch_size)
        if selected <= 0:
            raise ValueError(f"no validation blocks for {domain}")
        batches = []
        for start in range(0, selected, manifest.config.batch_size):
            ids = list(range(start, min(start + manifest.config.batch_size, selected)))
            rows = [span.block((index * (blocks - 1)) // max(1, selected - 1),
                               manifest.config.seq_len) for index in ids]
            inputs = torch.stack([row[0] for row in rows]).to(device)
            targets = torch.stack([row[1] for row in rows]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(inputs, targets, return_logits=False)
            loss_sum, count = float(output["loss_sum"].double()), int(output["loss_tokens"])
            if count <= 0 or not math.isfinite(loss_sum):
                raise ValueError(f"nonfinite held-out loss for {domain}")
            batches.append({"start": start, "examples": len(ids),
                            "loss_sum": loss_sum, "tokens": count})
        total_loss = sum(item["loss_sum"] for item in batches)
        total_tokens = sum(item["tokens"] for item in batches)
        result[domain] = {"loss": total_loss / total_tokens, "tokens": total_tokens,
                          "batches": batches}
    return result


@torch.inference_mode()
def context_position_losses(model, manifest, data_root: Path, device: torch.device,
                            batch_size: int) -> dict:
    """Score fixed position buckets; this diagnoses position effects, not retrieval ability."""
    from terminus.data import TokenSpan
    from terminus.integrity import manifest_scope

    domains = set(manifest_scope(manifest)["domains"])
    key_of = lambda spec: spec.source if spec.source in domains else spec.language
    intervals = (("0-511", 0, 512), ("512-2047", 512, 2048),
                 ("2048-4095", 2048, 4096))
    output = {}
    for domain in sorted({key_of(spec) for spec in manifest.shards_for(split=manifest.config.val_split)}):
        specs = [spec for spec in manifest.shards_for(split=manifest.config.val_split)
                 if key_of(spec) == domain]
        span = TokenSpan(specs, root=data_root)
        blocks = (span.num_tokens - 1) // manifest.config.seq_len
        selected = min(blocks, batch_size)
        if selected == 0:
            raise ValueError(f"no validation blocks for {domain}")
        rows = [span.block((index * (blocks - 1)) // max(1, selected - 1),
                           manifest.config.seq_len) for index in range(selected)]
        inputs = torch.stack([row[0] for row in rows]).to(device)
        targets = torch.stack([row[1] for row in rows]).to(device)
        output[domain] = {}
        for name, start, end in intervals:
            masked = targets.clone()
            masked[:, :start] = -100
            masked[:, end:] = -100
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                result = model(inputs, masked, return_logits=False)
            total, count = float(result["loss_sum"].double()), int(result["loss_tokens"])
            output[domain][name] = {"loss": total / count, "tokens": count}
    return output


def task_metrics(model, task_manifest_path: Path, device: torch.device,
                 max_seq_len: int, max_items: int) -> dict:
    from terminus.task_eval import TaskItem, aggregate, score_items
    from terminus.tokenizer import load_tokenizer
    from .protocol import ROOT

    manifest = read_json(task_manifest_path)
    tokenizer = load_tokenizer(ROOT / "tokenizer/tokenizer-65k.json")
    items = []
    for entry in manifest["tasks"]:
        task_path = resolve_from(task_manifest_path, entry["path"])
        with task_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= max_items:
                    break
                row = json.loads(line)
                item_id = str(row.get("id") or hashlib.blake2b(line.encode(), digest_size=12).hexdigest())
                items.append(TaskItem(task=f"{entry['domain']}/{entry['task']}",
                                      item_id=item_id, prompt=row["prompt"],
                                      reference=row["reference"], language=entry["domain"],
                                      choices=tuple(row["choices"]) if row.get("choices") else None,
                                      correct_choice=row.get("correct_choice"),
                                      role="development"))
    for item in items:
        if len(tokenizer.encode(item.prompt)) + len(tokenizer.encode(item.reference)) > max_seq_len + 1:
            raise ValueError(f"development item exceeds fixed context: {item.task}/{item.item_id}")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        records = score_items(model, tokenizer, items, device=device, max_seq_len=max_seq_len)
    summary = aggregate(records)
    if summary["truncated_items"]:
        raise ValueError(f"development suite truncated {summary['truncated_items']} prompts; "
                         "fix the benchmark before comparing candidates")
    return {"summary": summary, "records": records,
            "note": "answer-only reference loss; task accuracy requires separately defined scoring"}


def evaluate_run(run_dir: Path, manifest_path: Path, data_root: Path,
                 task_manifest_path: Path | None, *, full: bool) -> dict:
    from terminus.checkpoint import load_checkpoint
    from terminus.data import TokenManifest
    from terminus.model import BitNetLlama, BitNetLlamaConfig

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("native Terminus architecture evaluation requires Hopper SM90")
    rules = policy()
    if task_manifest_path is not None:
        validate_task_manifest(task_manifest_path, rules["domains"])
    contract = read_json(run_dir / "run.json")["contract"]
    checkpoint = load_checkpoint(run_dir / "checkpoints", contract)
    model = BitNetLlama(BitNetLlamaConfig(**contract["model"])).to("cuda")
    model.load_state_dict(checkpoint["model"])
    model.clear_fp8_weight_cache()
    state = checkpoint["state"]
    del checkpoint
    model.eval()
    manifest = TokenManifest.load(manifest_path)
    batches = rules["validation_batches_per_domain"]
    with torch.inference_mode():
        heldout = heldout_batches(model, manifest, data_root, torch.device("cuda"), batches)
        positions = context_position_losses(model, manifest, data_root,
                                            torch.device("cuda"), min(rules["microbatch"], 8))
        tasks = (task_metrics(model, task_manifest_path, torch.device("cuda"),
                              rules["max_seq_len"], rules["max_task_items_per_task"])
                 if full and task_manifest_path is not None else None)
    losses = {domain: value["loss"] for domain, value in heldout.items()}
    if set(losses) != set(rules["domains"]) or any(not math.isfinite(v) for v in losses.values()):
        raise ValueError("held-out evaluation missing domains or returned nonfinite loss")
    return {"tokens_seen": state["tokens_seen"], "updates": state["updates"],
            "heldout": heldout, "position_buckets": positions, "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = evaluate_run(args.run_dir, args.manifest, args.data_root,
                          args.task_manifest, full=args.full)
    write_json(args.output, result)
    print(json.dumps({"tokens_seen": result["tokens_seen"],
                      "heldout": {k: v["loss"] for k, v in result["heldout"].items()},
                      "task_items": result["tasks"]["summary"]["items"] if result["tasks"] else 0},
                     sort_keys=True))


if __name__ == "__main__":
    main()
