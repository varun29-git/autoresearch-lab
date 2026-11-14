"""Task-aware evaluation: conditional reference loss, margins, NLL, and CIs.

Teacher-forced conditional reference loss: prompt and reference are
concatenated, prompt positions are masked with -100, and only output tokens
are scored (the first output token is predicted from the last prompt token —
the pre-shifted target convention the model already uses). Loss is normalized
per item (per output token) before any task averaging, so long answers do not
dominate. Answer-only loss and answer-choice margins are reported separately
from full-output (rationale) loss: generated rationales may be wrong.

Batching correctness: items are grouped by identical token length and stacked
without padding, so no attention mask is ever needed (production attention
backends reject masks). Partial final batches are normal. Distributed scoring
is file-based: each rank scores a deterministic shard (item_index %
world_size), writes per-item JSONL partials, and `reduce_partials` merges them
in item order with a seeded bootstrap confidence interval.

Nothing here trains on task data, and ScalingBench records are evaluation-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TaskItem:
    task: str
    item_id: str
    prompt: str
    reference: str = ""
    language: str = "en"
    answer: str | None = None          # short verified answer, if separable
    choices: tuple | None = None       # answer-choice tasks
    correct_choice: int | None = None
    role: str = "development"          # development | final


def masked_continuation(prompt_ids, continuation_ids, *, max_seq_len):
    """Build one pre-shifted pair. Prompt positions are masked except the last.

    The last prompt position predicts the first continuation token. A long
    prompt is truncated from the left so the continuation stays intact.
    """
    prompt_ids = list(prompt_ids)
    continuation_ids = list(continuation_ids)
    original_continuation_tokens = len(continuation_ids)
    if not continuation_ids:
        raise ValueError("empty reference cannot be scored")
    total = len(prompt_ids) + len(continuation_ids)
    truncated_prompt = 0
    if total > max_seq_len + 1:
        overflow = total - (max_seq_len + 1)
        truncated_prompt = min(overflow, len(prompt_ids))
        prompt_ids = prompt_ids[truncated_prompt:]
        if len(continuation_ids) + 1 > max_seq_len + 1:
            continuation_ids = continuation_ids[:max_seq_len]
    tokens = prompt_ids + continuation_ids
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens[1:]], dtype=torch.long)
    if prompt_ids:
        targets[0, :len(prompt_ids) - 1] = IGNORE_INDEX
    scored_tokens = int(targets.ne(IGNORE_INDEX).sum())
    if not scored_tokens:
        raise ValueError("reference has no predictable tokens; provide a prompt")
    meta = {"output_tokens": scored_tokens, "prompt_tokens": len(prompt_ids),
            "truncated_prompt_tokens": truncated_prompt,
            "truncated_reference_tokens": original_continuation_tokens - len(continuation_ids),
            "unscored_reference_prefix_tokens": int(not prompt_ids)}
    return inputs, targets, meta


def encode_pair(tokenizer, prompt, reference, *, max_seq_len):
    """Tokenize prompt+reference; truncate the prompt from the left if needed.

    Returns (inputs, targets, meta): pre-shifted tensors with prompt positions
    masked; the first output token position is never masked.
    """
    inputs, targets, meta = masked_continuation(tokenizer.encode(prompt), tokenizer.encode(reference),
                                                max_seq_len=max_seq_len)
    meta["reference_utf8_bytes"] = len(reference.encode("utf-8"))
    return inputs, targets, meta


@torch.no_grad()
def score_item(model, tokenizer, item: TaskItem, *, device, max_seq_len):
    inputs, targets, meta = encode_pair(tokenizer, item.prompt, item.reference,
                                        max_seq_len=max_seq_len)
    inputs, targets = inputs.to(device), targets.to(device)
    output = model(inputs, targets=targets, return_logits=False)
    loss_sum = float(output["loss_sum"].double())
    record = {"task": item.task, "item_id": item.item_id, "language": item.language,
              "role": item.role, "loss": loss_sum / meta["output_tokens"], **meta,
              "bits_per_byte": (loss_sum / math.log(2) / max(1, meta["reference_utf8_bytes"])
                                if not meta["truncated_reference_tokens"] and
                                not meta["unscored_reference_prefix_tokens"] else None)}
    if item.answer is not None and item.answer != item.reference:
        a_inputs, a_targets, a_meta = encode_pair(tokenizer, item.prompt, item.answer,
                                                  max_seq_len=max_seq_len)
        a_output = model(a_inputs.to(device), targets=a_targets.to(device), return_logits=False)
        a_sum = float(a_output["loss_sum"].double())
        record["answer_loss"] = a_sum / a_meta["output_tokens"]
    return record


@torch.no_grad()
def choice_margins(model, tokenizer, item: TaskItem, *, device, max_seq_len):
    """Score each choice continuation; report margins and argmax accuracy."""
    if item.choices is None or item.correct_choice is None:
        raise ValueError("choice margins need choices and a correct index")
    losses = []
    for choice in item.choices:
        inputs, targets, meta = encode_pair(tokenizer, item.prompt, choice,
                                            max_seq_len=max_seq_len)
        output = model(inputs.to(device), targets=targets.to(device), return_logits=False)
        losses.append(float(output["loss_sum"].double()) / meta["output_tokens"])
    correct = losses[item.correct_choice]
    best_wrong = min(loss for index, loss in enumerate(losses) if index != item.correct_choice)
    predicted = min(range(len(losses)), key=lambda i: (losses[i], i))
    return {"task": item.task, "item_id": item.item_id, "language": item.language,
            "choice_losses": losses, "correct_loss": correct, "margin": best_wrong - correct,
            "correct": predicted == item.correct_choice}


def group_by_length(tokenizer, items, *, max_seq_len):
    """Group item indices by total token length for padding-free batching."""
    groups = {}
    for index, item in enumerate(items):
        ids = tokenizer.encode(item.prompt) + tokenizer.encode(item.reference)
        groups.setdefault(min(len(ids), max_seq_len + 1), []).append(index)
    return groups


@torch.no_grad()
def score_items(model, tokenizer, items, *, device, max_seq_len, choices=False):
    """Score items one sequence at a time in deterministic length order.

    Per-item scoring keeps masking exact and needs no padding or attention
    masks (production attention backends reject masks), so partial "batches"
    are trivially safe; length grouping only orders work for cache locality.
    """
    was_training = model.training
    model.eval()
    records = []
    try:
        groups = group_by_length(tokenizer, items, max_seq_len=max_seq_len)
        for length in sorted(groups):
            for index in groups[length]:
                item = items[index]
                if choices and item.choices is not None:
                    records.append(choice_margins(model, tokenizer, item, device=device,
                                                  max_seq_len=max_seq_len))
                else:
                    records.append(score_item(model, tokenizer, item, device=device,
                                              max_seq_len=max_seq_len))
    finally:
        if was_training:
            model.train()
    records.sort(key=lambda record: (record["task"], record["item_id"]))
    return records


def bootstrap_ci(values, *, resamples=10_000, confidence=0.95, seed=1337):
    """Deterministic percentile bootstrap CI over per-item means."""
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    generator = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(generator.choices(values, k=n)) / n)
    means.sort()
    tail = (1 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1 - tail) * resamples))]
    return {"mean": sum(values) / n, "low": low, "high": high, "n": n}


def aggregate(records, *, seed=1337):
    """Per-task and per-language aggregates with bootstrap CIs."""
    tasks, languages = {}, {}
    for record in records:
        if "loss" in record:
            tasks.setdefault(record["task"], []).append(record["loss"])
            languages.setdefault(record["language"], []).append(record["loss"])
    answer_losses = [record["answer_loss"] for record in records if "answer_loss" in record]
    choice_records = [record for record in records if "correct" in record]
    return {
        "items": len(records),
        "per_task": {task: bootstrap_ci(losses, seed=seed) for task, losses in sorted(tasks.items())},
        "per_language": {language: bootstrap_ci(losses, seed=seed)
                         for language, losses in sorted(languages.items())},
        "macro_task_loss": (sum(sum(losses) / len(losses) for losses in tasks.values())
                            / len(tasks)) if tasks else None,
        "truncated_items": sum(1 for record in records if record.get("truncated_prompt_tokens")),
        "answer_loss_mean": (sum(answer_losses) / len(answer_losses)) if answer_losses else None,
        "choice_accuracy": (sum(1 for record in choice_records if record["correct"])
                            / len(choice_records)) if choice_records else None,
    }


def shard_items(items, *, rank, world_size):
    return [item for index, item in enumerate(items) if index % world_size == rank]


def write_partials(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def reduce_partials(paths, *, seed=1337):
    records = []
    for path in sorted(paths, key=str):
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
    records.sort(key=lambda record: (record["task"], record["item_id"]))
    return records, aggregate(records, seed=seed)


def load_scalingbench_items(path, *, task, role="development", max_items=None):
    """ScalingBench rows: {"input", "output"} teacher-forced pairs."""
    items = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            items.append(TaskItem(task=task, item_id=hashlib.blake2b(
                line.encode(), digest_size=8).hexdigest(), prompt=row["input"],
                reference=row["output"], role=role))
            if max_items and len(items) >= max_items:
                break
    return items


def load_development_choices(root, *, expected_tokenizer_sha256=None):
    """Load the frozen development choice rows scored at every validation stop.

    Final rows and reference-answer tasks stay out of this set. Each token
    file must match the manifest digest, and the atom tokenizer digest must
    match the training manifest when one is supplied.
    """
    root = Path(root)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"benchmark token manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if expected_tokenizer_sha256 is not None and manifest.get("tokenizer_sha256") != expected_tokenizer_sha256:
        raise ValueError("benchmark tokens use a different tokenizer than the training manifest")
    records = []
    for task, info in sorted(manifest.get("tasks", {}).items()):
        if info.get("kind") != "choice" or info.get("cadence") != "every_validation_stop":
            continue
        path = root / info["file"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise ValueError(f"benchmark token file differs from its manifest: {task}")
        for line in payload.decode("utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("role") != "development" or row.get("kind") != "choice":
                continue
            if not row.get("choice_ids") or type(row.get("correct_choice")) is not int:
                raise ValueError(f"choice record is incomplete: {task} {row.get('item_id')}")
            records.append(row)
    if not records:
        raise ValueError("benchmark directory has no development choice items")
    return records


@torch.no_grad()
def score_frozen_choices(model, records, *, device, max_seq_len, rank=0, world_size=1, amp_dtype=None,
                         progress_callback=None):
    """Score stored choice continuations and reduce accuracy across ranks.

    This uses the token ids frozen by the atom tokenizer. It does not
    re-encode text. Every rank must call it so the reduction stays aligned.
    """
    import torch.distributed as distributed

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid benchmark rank/world size")
    was_training = model.training
    model.eval()
    buckets = {}
    completed = 0
    total_choices = sum(len(row["choice_ids"]) for index, row in enumerate(records) if index % world_size == rank)
    try:
        for index, row in enumerate(records):
            if index % world_size != rank:
                continue
            losses = []
            for choice in row["choice_ids"]:
                inputs, targets, meta = masked_continuation(row["prompt_ids"], choice, max_seq_len=max_seq_len)
                inputs, targets = inputs.to(device), targets.to(device)
                enabled = amp_dtype is not None and amp_dtype != torch.float32
                with torch.autocast(device_type=device.type, dtype=amp_dtype if enabled else torch.float32, enabled=enabled):
                    output = model(inputs, targets=targets, return_logits=False)
                losses.append(float(output["loss_sum"].double()) / meta["output_tokens"])
                completed += 1
                if progress_callback is not None:
                    progress_callback(dict(phase="benchmark", key=row["task"],
                        completed_choices=completed, total_choices=total_choices))
            correct = row["correct_choice"]
            if len(losses) < 2 or not 0 <= correct < len(losses):
                raise ValueError(f"choice index differs: {row['task']} {row['item_id']}")
            predicted = min(range(len(losses)), key=lambda i: (losses[i], i))
            margin = min(loss for choice_index, loss in enumerate(losses) if choice_index != correct) - losses[correct]
            bucket = buckets.setdefault(row["task"], [0, 0, 0.0])
            bucket[0] += int(predicted == correct)
            bucket[1] += 1
            bucket[2] += margin
    finally:
        model.train(was_training)
    tasks = sorted({row["task"] for row in records})
    vector = torch.zeros(len(tasks) * 3, dtype=torch.float64, device=device)
    for index, task in enumerate(tasks):
        correct, count, margin = buckets.get(task, (0, 0, 0.0))
        vector[index * 3] = correct
        vector[index * 3 + 1] = count
        vector[index * 3 + 2] = margin
    if world_size > 1:
        distributed.all_reduce(vector)
    result = {}
    for index, task in enumerate(tasks):
        correct, count, margin = vector[index * 3:index * 3 + 3].tolist()
        if count <= 0 or not math.isfinite(margin):
            raise ValueError(f"benchmark task has no finite scored items: {task}")
        result[task] = {"correct": int(correct), "items": int(count),
                        "accuracy": correct / count, "mean_margin": margin / count}
    return result


def task_text_hashes(paths, *, max_rows_per_file=None):
    """Normalized task texts for the corpus contamination guard."""
    texts = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if max_rows_per_file and index >= max_rows_per_file:
                    break
                row = json.loads(line)
                texts.extend(value for value in (row.get("input"), row.get("output"))
                             if isinstance(value, str))
    return texts
