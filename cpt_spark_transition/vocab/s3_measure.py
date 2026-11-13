"""Fertility and step-0 loss for the 32k English keep-sets plus Sarvam Indic.

Reads the campaign parquet in s3://nemotron-training-data-us-east-1.
Frequency for method 2 is English, code, and math only. Each keep-set then
gains every Spark row in the seven target scripts, the merge ancestors of
those rows, and the Sarvam pieces whose surface is not already kept.

Loss is one forward of Spark-X2.6-1.7B-Base. Kept rows are copied. New rows
are the scaled constituent mean. Nothing is written back to a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
REPO = FOLDER.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vocab.english_keepsets import (  # noqa: E402
    dropped_ancestor_count,
    lengths_for,
    load_spark,
    method1,
    method2,
    restore_ancestors,
    script_mix,
)
from vocab.indic import SPACE_MARK, has_target_script, is_target_char  # noqa: E402
from vocab.model import DONOR_PATH, Cut, decode_token  # noqa: E402
from vocab.transplant import assemble, kept_rows_match  # noqa: E402

BUCKET = "nemotron-training-data-us-east-1"
MODEL_ID = "XHToken/Spark-X2.6-1.7B-Base"
REPORT_DIR = FOLDER / "reports"
SLICE_CHARS = 4000
FREQ_CHARS = 100_000
LOSS_TOKENS = 2048

SOURCES = {
    "english": {
        "freq_key": "datasets/nemotron-cc-hq/part_000000.parquet",
        "eval_key": "datasets/nemotron-cc-hq/part_000400.parquet",
        "freq_tokens": 20_000_000,
        "eval_tokens": 300_000,
    },
    "code": {
        "freq_key": (
            "datasets/Nemotron-Pretraining-Specialized-v1.1/"
            "Nemotron-Pretraining-Code-Concepts/part_000000.parquet"
        ),
        "eval_key": (
            "datasets/Nemotron-Pretraining-Specialized-v1.1/"
            "Nemotron-Pretraining-Code-Concepts/part_000040.parquet"
        ),
        "freq_tokens": 4_000_000,
        "eval_tokens": 150_000,
    },
    "math": {
        "freq_key": (
            "datasets/Nemotron-Pretraining-Specialized-v1/"
            "Nemotron-Pretraining-Math-Textbooks/part_000000.parquet"
        ),
        "eval_key": (
            "datasets/Nemotron-Pretraining-Specialized-v1/"
            "Nemotron-Pretraining-Math-Textbooks/part_000008.parquet"
        ),
        "freq_tokens": 4_000_000,
        "eval_tokens": 150_000,
    },
}
INDIC_KEYS = {
    "hi": "datasets/sangraha/verified/hin/data-0.parquet",
    "mr": "datasets/sangraha/verified/mar/data-0.parquet",
    "kn": "datasets/sangraha/verified/kan/data-0.parquet",
    "te": "datasets/sangraha/verified/tel/data-0.parquet",
    "ta": "datasets/sangraha/verified/tam/data-0.parquet",
    "gu": "datasets/sangraha/verified/guj/data-0.parquet",
    "bn": "datasets/sangraha/verified/ben/data-0.parquet",
}
INDIC_EVAL_TOKENS = 80_000
TOKENIZER_ORDER = (
    "stock",
    "script_cut",
    "method1_indic",
    "method2_indic",
    "method2_ancestors_indic",
    "sarvam65k",
)
TOKENIZER_LABELS = {
    "stock": "Stock Spark",
    "script_cut": "Script cut + Indic",
    "method1_indic": "Method 1 + Indic",
    "method2_indic": "Method 2 + Indic",
    "method2_ancestors_indic": "Method 2 + ancestors + Indic",
    "sarvam65k": "Sarvam 65k",
}
LOSS_TOKENIZERS = (
    "stock",
    "script_cut",
    "method1_indic",
    "method2_indic",
    "method2_ancestors_indic",
)


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def target_script_ids(token_at):
    found = set()
    for index, token in token_at.items():
        surface = decode_token(token)
        if surface and any(is_target_char(char) for char in surface):
            found.add(index)
    return found


def extend_with_indic(core, token_at, parents):
    indic = target_script_ids(token_at)
    closed = restore_ancestors(indic, parents)
    keep = set(core) | closed
    return keep, {
        "core_ids": len(core),
        "spark_indic_ids": len(indic),
        "indic_ids_not_in_core": len(indic - core),
        "indic_ancestor_ids_not_in_core": len((closed - indic) - core),
        "spark_ids": len(keep),
    }


def open_parquet(key):
    import pyarrow.parquet as pq
    import s3fs

    filesystem = s3fs.S3FileSystem()
    handle = filesystem.open(f"{BUCKET}/{key}", "rb")
    parquet = pq.ParquetFile(handle)
    names = parquet.schema_arrow.names
    if "text" not in names:
        raise RuntimeError(f"{key} columns are {names}, expected a text column")
    return parquet


def iter_slices(key, char_cap):
    log(f"opening s3://{BUCKET}/{key}")
    parquet = open_parquet(key)
    for batch in parquet.iter_batches(batch_size=32, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if isinstance(text, str) and text.strip():
                yield text[:char_cap]


def take_texts(key, tokenizer, token_cap, char_cap):
    texts = []
    tokens = 0
    for text in iter_slices(key, char_cap):
        count = len(tokenizer.encode(text).ids)
        if count == 0:
            continue
        texts.append(text)
        tokens += count
        if tokens >= token_cap:
            break
        if len(texts) % 2000 == 0:
            log(f"{key} {len(texts)} documents, {tokens} tokens")
    log(f"{key} stopped at {len(texts)} documents, {tokens} tokens")
    return texts, tokens


def count_ids(key, tokenizer, token_cap, counter):
    tokens = 0
    documents = 0
    pending = []

    def flush():
        nonlocal tokens, documents
        if not pending:
            return
        for encoded in tokenizer.encode_batch(pending):
            if not encoded.ids:
                continue
            counter.update(encoded.ids)
            tokens += len(encoded.ids)
            documents += 1
        pending.clear()

    for text in iter_slices(key, FREQ_CHARS):
        pending.append(text)
        if len(pending) >= 32:
            flush()
            if documents % 2000 < 32:
                log(f"count {key} {documents} documents, {tokens} tokens")
            if tokens >= token_cap:
                break
    if tokens < token_cap:
        flush()
    log(f"count {key} done: {documents} documents, {tokens} tokens")
    return {"key": key, "documents": documents, "tokens": tokens}


def donor_encoder():
    from terminus.tokenizer65k import Frozen65kTokenizer

    return Frozen65kTokenizer(DONOR_PATH)


def stock_words_bytes(text):
    words = text.split()
    return (len(words) if words else 1), len(text.encode("utf-8"))


def prepare_document(text, encode_ids, decode_ids):
    ids = list(encode_ids(text))
    words, nbytes = stock_words_bytes(text)
    score_ids = ids[:LOSS_TOKENS]
    truncated = len(ids) > LOSS_TOKENS
    if len(score_ids) < 2:
        score_bytes = 0
        score_words = 0
    elif truncated:
        decoded = decode_ids(score_ids)
        score_words, score_bytes = stock_words_bytes(decoded)
    else:
        score_words, score_bytes = words, nbytes
    return {
        "ids": ids,
        "score_ids": score_ids,
        "words": words,
        "bytes": nbytes,
        "score_words": score_words,
        "score_bytes": score_bytes,
        "truncated": truncated,
    }


def fertility_of(documents):
    tokens = sum(len(doc["ids"]) for doc in documents)
    words = sum(doc["words"] for doc in documents)
    nbytes = sum(doc["bytes"] for doc in documents)
    return {
        "documents": len(documents),
        "tokens": tokens,
        "words": words,
        "bytes": nbytes,
        "tokens_per_word": round(tokens / words, 4) if words else None,
        "tokens_per_byte": round(tokens / nbytes, 4) if nbytes else None,
    }


def identical_fraction(pairs):
    if not pairs:
        return None
    same = sum(1 for left, right in pairs if left == right)
    return round(same / len(pairs), 4)


def parent_walk(stock_ids, keep, parents):
    span = lengths_for(keep, parents)
    flat = [index for ids in stock_ids for index in ids]
    if not flat:
        return None
    expanded = sum(span(index) for index in flat)
    missing = {index for index in flat if index not in keep}
    return {
        "stock_tokens": len(flat),
        "expanded_tokens": expanded,
        "fertility_ratio": round(expanded / len(flat), 4),
        "missing_distinct": len(missing),
        "missing_occurrence_pct": round(
            100 * sum(1 for index in flat if index not in keep) / len(flat), 2
        ),
    }


def nats_of(model, weight, ids, window):
    import torch
    import torch.nn.functional as functional

    if len(ids) < 2:
        return 0.0, 0
    if max(ids) >= weight.shape[0]:
        raise RuntimeError(
            f"id {max(ids)} is outside the {weight.shape[0]}-row matrix"
        )
    device = weight.device
    total = 0.0
    counted = 0
    start = 0
    while start < len(ids) - 1:
        chunk = ids[start:start + window]
        if len(chunk) < 2:
            break
        batch = torch.tensor([chunk], dtype=torch.long, device=device)
        embeds = functional.embedding(batch, weight)
        hidden = model.model(
            inputs_embeds=embeds, use_cache=False
        ).last_hidden_state
        logits = functional.linear(hidden, weight)
        loss = functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total += float(loss.item())
        counted += len(chunk) - 1
        start += window
        del batch, embeds, hidden, logits, loss
    return total, counted


def summarize_loss(rows):
    nats = sum(row[0] for row in rows)
    predictions = sum(row[1] for row in rows)
    score_bytes = sum(row[2] for row in rows)
    per_doc = [row[3] for row in rows if row[3] is not None]
    mean = sum(per_doc) / len(per_doc) if per_doc else None
    error = None
    if len(per_doc) >= 2 and mean is not None:
        variance = sum((value - mean) ** 2 for value in per_doc) / (len(per_doc) - 1)
        error = math.sqrt(variance / len(per_doc))
    nats_per_token = nats / predictions if predictions else None
    nats_per_byte = nats / score_bytes if score_bytes else None
    bits_per_byte = nats_per_byte / math.log(2) if nats_per_byte is not None else None
    return {
        "documents": len(rows),
        "predictions": predictions,
        "scored_bytes": score_bytes,
        "truncated_documents": sum(1 for row in rows if row[4]),
        "nats": round(nats, 4),
        "nats_per_token": round(nats_per_token, 4) if nats_per_token is not None else None,
        "nats_per_byte": round(nats_per_byte, 4) if nats_per_byte is not None else None,
        "bits_per_byte": round(bits_per_byte, 4) if bits_per_byte is not None else None,
        "bits_per_byte_stderr": round(error, 4) if error is not None else None,
    }


def build_matrix(stock_bf16, cpu_f32, cut):
    import torch

    lists = []
    for piece in cut.new_pieces:
        surface = piece.replace(SPACE_MARK, " ")
        constituents = cut.stock_constituents(surface)
        if not constituents:
            raise RuntimeError(f"no Spark constituents for {piece!r}")
        lists.append(constituents)
    log(f"assembling {len(cut.kept_old_ids())} kept rows and {len(lists)} new rows")
    matrix = assemble(cpu_f32, cut.kept_old_ids(), lists).to(dtype=torch.bfloat16)
    if not kept_rows_match(matrix, stock_bf16, cut.kept_old_ids()):
        raise RuntimeError("assembled kept rows do not match the original embedding")
    return matrix


def score_tokenizer(model, weight, prepared, window):
    import torch

    out = {}
    for domain, documents in prepared.items():
        rows = []
        for doc in documents:
            try:
                nats, predictions = nats_of(model, weight, doc["score_ids"], window)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if window == 1024:
                    raise
                log(f"out of memory at window {window}; retrying {domain} at 1024")
                nats, predictions = nats_of(model, weight, doc["score_ids"], 1024)
                window = 1024
            bits = None
            if predictions and doc["score_bytes"]:
                bits = (nats / doc["score_bytes"]) / math.log(2)
            rows.append((nats, predictions, doc["score_bytes"], bits, doc["truncated"]))
        out[domain] = summarize_loss(rows)
        log(
            f"loss {domain}: {out[domain]['bits_per_byte']} bits/byte "
            f"over {out[domain]['predictions']} predictions"
        )
    return out, window


def cut_record(cut, extra):
    counts = cut.counts()
    return extra | counts | {
        "closure_ok": cut.closure_ok,
        "dropped_ancestors": None,
    }


def recommend(summary):
    fertility = summary["fertility"]
    loss = summary["loss"]
    stock_bits = loss["english"]["stock"]["bits_per_byte"]
    ranked = []
    for key in ("method1_indic", "method2_indic", "method2_ancestors_indic"):
        bits = loss["english"][key]["bits_per_byte"]
        gaps = []
        for domain in ("english", "code", "math"):
            ratio = fertility[domain][key]["ratio_vs_stock"]
            gaps.append(abs(ratio - 1.0))
        ranked.append((
            abs(bits - stock_bits),
            sum(gaps) / len(gaps),
            summary["tokenizers"][key]["dropped_ancestors"],
            key,
        ))
    ranked.sort()
    winner = ranked[0][3]
    script_bits = loss["english"]["script_cut"]["bits_per_byte"]
    script_ratio = fertility["english"]["script_cut"]["ratio_vs_stock"]
    label = TOKENIZER_LABELS[winner]
    return (
        f"Among the three 32k-English plus Indic tokenizers, {label} stays closest "
        f"to stock English. Its fertility ratio is "
        f"{fertility['english'][winner]['ratio_vs_stock']} and its step-0 loss is "
        f"{loss['english'][winner]['bits_per_byte']} bits/byte, against stock "
        f"{stock_bits}. The script cut keeps every Spark piece outside the dropped "
        f"scripts. Its English fertility ratio is {script_ratio} and its step-0 "
        f"loss is {script_bits} bits/byte, at "
        f"{summary['tokenizers']['script_cut']['vocab_size']} rows. Indic step-0 "
        f"loss is the loss of the copied Spark rows plus the untrained new rows, "
        f"before any alignment step."
    )


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(summary):
    lines = [
        "# Keep-sets with Indic: fertility and step-0 loss",
        "",
        "Measured on one AWS `g5.2xlarge` against "
        f"`s3://{BUCKET}`. No checkpoint was written. No tokenizer file was saved.",
        "The id lists are the Spark rows that stay. Appended Sarvam pieces follow",
        "those rows, in the order stored beside each list.",
        "",
        f"Stock tokenizer sha256 `{summary['spark_sha256']}`.",
        f"Donor table sha256 `{summary['donor_sha256']}`.",
        f"Model `{summary['model']}`, loaded in bf16. Instance `{summary['instance_id']}`.",
        "",
        "Method 2's frequency count is the S3 sample below, not the earlier local",
        "HellaSwag cache. Method 1 is still the earliest Spark ids and does not",
        "depend on the corpus. After either English core is chosen, every Spark",
        "row whose surface is in the seven target scripts is force-kept, the merge",
        "ancestors of those rows are restored, and Sarvam pieces that are not",
        "already a kept surface are appended. Method 2 is also reported without",
        "restoring ancestors of the English frequency picks. Those tokens stay in",
        "the list and the pruned merge table cannot emit them.",
        "",
        "Each frequency document is capped at "
        f"{summary['freq_chars']} characters. Each evaluation document is the",
        f"first {summary['slice_chars']} characters. Loss scores at most the first",
        f"{summary['loss_tokens']} tokens of that slice. Bits per byte uses the",
        "UTF-8 length of the text that those tokens cover. It is the loss figure",
        "that stays comparable when the tokenizers cut the same text into different",
        "numbers of pieces. Nats per token is the mean next-token loss on each",
        "tokenizer's own sequence.",
        "",
        "New rows use the scaled constituent mean: the mean of the stock Spark",
        "rows of that surface, rescaled to the mean norm of those rows. No",
        "optimizer step was run. A lower nats-per-token after a smaller softmax",
        "is not, by itself, evidence that the model got better.",
        "",
        "## Sample",
        "",
        "| Domain | Role | S3 key | Documents | Stock tokens |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for name, row in summary["frequency"].items():
        lines.append(
            f"| {name} | frequency | `{row['key']}` | {row['documents']} | {row['tokens']} |"
        )
    for name, row in summary["evaluation"].items():
        lines.append(
            f"| {name} | held-out | `{row['key']}` | {row['documents']} | {row['tokens']} |"
        )
    lines.extend([
        "",
        f"Frequency saw {summary['frequency_types']} distinct stock ids across "
        f"{summary['frequency_tokens']} tokens.",
        "",
        "## Vocabulary",
        "",
        "| | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic | Script cut + Indic |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    order = (
        "method1_indic",
        "method2_indic",
        "method2_ancestors_indic",
        "script_cut",
    )
    rows = (
        ("English core ids", "core_ids"),
        ("Spark seven-script ids added beyond the core", "indic_ids_not_in_core"),
        ("Ancestors of those rows added beyond the core", "indic_ancestor_ids_not_in_core"),
        ("Spark rows kept", "spark_kept"),
        ("Sarvam pieces reused", "sarvam_reused"),
        ("Sarvam pieces appended", "sarvam_added"),
        ("Finished vocabulary", "vocab_size"),
        ("Kept ids with a dropped merge ancestor", "dropped_ancestors"),
        ("Kept ids in a dropped script", "dropped_script"),
    )
    for label, field in rows:
        cells = []
        for key in order:
            record = summary["tokenizers"][key]
            if field == "dropped_script":
                value = record["script_mix"]["dropped_script"]
            else:
                value = record.get(field)
            cells.append(fmt(value, 0) if isinstance(value, int) else fmt(value))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        f"Overlap of the method 2 English core with method 1: "
        f"{summary['overlap_method2']}.",
        f"Overlap of the ancestor-closed method 2 core with method 1: "
        f"{summary['overlap_method2_ancestors']}.",
        "",
        "## Fertility, tokens per word",
        "",
        header(),
    ])
    for domain in summary["domain_order"]:
        lines.append(metric_row(summary, domain, "tokens_per_word", section="fertility"))
    lines.extend([
        "",
        "## Fertility ratio versus stock tokenization",
        "",
        "Actual encoder length divided by the stock Spark length on the same",
        "held-out slices. The parent-walk column expands a stock id whenever that",
        "id, or one of its merge ancestors, is outside the Spark keep-set. The",
        "loss uses the actual encoder, not the parent walk.",
        "",
        header(),
    ])
    for domain in summary["domain_order"]:
        lines.append(metric_row(summary, domain, "ratio_vs_stock", section="fertility"))
    lines.extend([
        "",
        "| Domain | Method 1 parent-walk | Method 2 parent-walk | Method 2 + ancestors parent-walk |",
        "| --- | ---: | ---: | ---: |",
    ])
    for domain in summary["domain_order"]:
        walk = summary["parent_walk"][domain]
        lines.append(
            f"| {domain} | {fmt(walk['method1_indic']['fertility_ratio'])} | "
            f"{fmt(walk['method2_indic']['fertility_ratio'])} | "
            f"{fmt(walk['method2_ancestors_indic']['fertility_ratio'])} |"
        )
    lines.extend([
        "",
        "Missing stock ids on the held-out slice, counted against the Spark rows",
        "of that keep-set:",
        "",
        "| Domain | Method 1 distinct missing | Method 2 distinct missing | Method 2 + ancestors distinct missing |",
        "| --- | ---: | ---: | ---: |",
    ])
    for domain in summary["domain_order"]:
        walk = summary["parent_walk"][domain]
        lines.append(
            f"| {domain} | {walk['method1_indic']['missing_distinct']} | "
            f"{walk['method2_indic']['missing_distinct']} | "
            f"{walk['method2_ancestors_indic']['missing_distinct']} |"
        )
    lines.extend([
        "",
        "Share of held-out documents whose Spark piece strings match stock.",
        "Indic text is routed, so this check is only on English, code, and math.",
        "",
        "| Domain | Script cut | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for domain in ("english", "code", "math"):
        cells = []
        for key in ("script_cut", "method1_indic", "method2_indic", "method2_ancestors_indic"):
            cells.append(fmt(summary["fertility"][domain][key]["identical_documents"]))
        lines.append(f"| {domain} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Step-0 loss, bits per byte",
        "",
        "Mean document bits/byte, with the standard error of that document mean.",
        "Sarvam 65k has no Spark embedding, so it has fertility and no loss.",
        "",
        header(loss=True),
    ])
    for domain in summary["domain_order"]:
        lines.append(loss_row(summary, domain, "bits_per_byte"))
    lines.extend([
        "",
        "## Step-0 loss, nats per token",
        "",
        header(loss=True),
    ])
    for domain in summary["domain_order"]:
        lines.append(loss_row(summary, domain, "nats_per_token"))
    lines.extend([
        "",
        "## Recommendation",
        "",
        summary["recommendation"],
        "",
        "Lists:",
        "",
        "- `reports/s3_keep_method1_plus_indic.json`",
        "- `reports/s3_keep_method1_appended_pieces.json`",
        "- `reports/s3_keep_method2_plus_indic.json`",
        "- `reports/s3_keep_method2_appended_pieces.json`",
        "- `reports/s3_keep_method2_ancestors_plus_indic.json`",
        "- `reports/s3_keep_method2_ancestors_appended_pieces.json`",
        "",
    ])
    return "\n".join(lines) + "\n"


def header(loss=False):
    labels = [TOKENIZER_LABELS[key] for key in TOKENIZER_ORDER]
    if loss:
        labels = [TOKENIZER_LABELS[key] for key in LOSS_TOKENIZERS]
    return (
        "| Domain | " + " | ".join(labels) + " |\n| --- | "
        + " | ".join(["---:"] * len(labels)) + " |"
    )


def metric_row(summary, domain, field, section):
    cells = []
    for key in TOKENIZER_ORDER:
        row = summary[section][domain].get(key)
        cells.append(fmt(None if row is None else row.get(field)))
    return f"| {domain} | " + " | ".join(cells) + " |"


def loss_row(summary, domain, field):
    cells = []
    for key in LOSS_TOKENIZERS:
        row = summary["loss"][domain][key]
        if field == "bits_per_byte" and row.get("bits_per_byte_stderr") is not None:
            cells.append(f"{fmt(row[field])} ± {fmt(row['bits_per_byte_stderr'])}")
        else:
            cells.append(fmt(row.get(field)))
    return f"| {domain} | " + " | ".join(cells) + " |"


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def caps(smoke):
    if not smoke:
        return {name: dict(spec) for name, spec in SOURCES.items()}, dict(INDIC_KEYS), INDIC_EVAL_TOKENS
    small = {}
    for name, spec in SOURCES.items():
        copied = dict(spec)
        copied["freq_tokens"] = 30_000
        copied["eval_tokens"] = 8_000
        small[name] = copied
    return small, {"hi": INDIC_KEYS["hi"]}, 8_000


def load_or_count(cache_dir, reuse, tokenizer, smoke):
    cache_dir.mkdir(parents=True, exist_ok=True)
    freq_path = cache_dir / "frequency.json"
    eval_path = cache_dir / "evaluation.json"
    source_caps, indic_keys, indic_cap = caps(smoke)
    if reuse and freq_path.is_file():
        log("reusing cached frequency counts")
        freq = json.loads(freq_path.read_text(encoding="utf-8"))
        counter = Counter({int(key): value for key, value in freq["counts"].items()})
        domains = freq["domains"]
    else:
        counter = Counter()
        domains = {}
        for name, spec in source_caps.items():
            domains[name] = count_ids(spec["freq_key"], tokenizer, spec["freq_tokens"], counter)
        freq_path.write_text(json.dumps({
            "domains": domains,
            "counts": {str(key): value for key, value in counter.items()},
        }), encoding="utf-8")
        log(f"cached frequency counts at {freq_path}")
    if reuse and eval_path.is_file():
        log("reusing cached evaluation text")
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    else:
        evaluation = {}
        for name, spec in source_caps.items():
            texts, tokens = take_texts(
                spec["eval_key"], tokenizer, spec["eval_tokens"], SLICE_CHARS
            )
            evaluation[name] = {
                "key": spec["eval_key"],
                "documents": len(texts),
                "tokens": tokens,
                "texts": texts,
            }
        for name, key in indic_keys.items():
            texts, tokens = take_texts(key, tokenizer, indic_cap, SLICE_CHARS)
            evaluation[name] = {
                "key": key, "documents": len(texts), "tokens": tokens, "texts": texts,
            }
        eval_path.write_text(json.dumps(evaluation), encoding="utf-8")
        log(f"cached evaluation text at {eval_path}")
    return counter, domains, evaluation


def prefetch_model(box):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, ignore_patterns=["images/*", "*.svg", "*.jpg", "*.md"])
        box.append(None)
    except Exception as exc:
        box.append(exc)


def run_loss(cuts, prepared, stock_decode_docs):
    import torch
    from transformers import AutoModelForCausalLM

    log(f"loading {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    stock_weight = model.model.embedding.weight.detach()
    stock_cpu = stock_weight.cpu()
    cpu_f32 = stock_cpu.float()
    window = LOSS_TOKENS
    loss = {}
    prepared_stock = {
        domain: [doc["stock"] for doc in docs] for domain, docs in prepared.items()
    }
    with torch.inference_mode():
        loss["stock"], window = score_tokenizer(model, stock_weight, prepared_stock, window)
        for name, cut in cuts.items():
            matrix = build_matrix(stock_cpu, cpu_f32, cut)
            weight = matrix.to(device="cuda")
            del matrix
            prepared_cut = {
                domain: [doc[name] for doc in docs] for domain, docs in prepared.items()
            }
            loss[name], window = score_tokenizer(model, weight, prepared_cut, window)
            del weight
            torch.cuda.empty_cache()
    del model, stock_weight, cpu_f32
    torch.cuda.empty_cache()
    return loss, window


def loss_by_domain(loss):
    """Turn tokenizer -> domain -> stats into domain -> tokenizer -> stats."""
    domains = list(next(iter(loss.values())))
    return {
        domain: {name: table[domain] for name, table in loss.items()}
        for domain in domains
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-dir", default="/tmp/keepset-cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--skip-loss", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    box = []
    thread = threading.Thread(target=prefetch_model, args=(box,), daemon=True)
    thread.start()
    log("loading stock Spark tokenizer")
    _raw, _vocab, token_at, specials, byte_ids, parents, tokenizer = load_spark()
    counter, freq_domains, evaluation = load_or_count(
        Path(args.cache_dir), args.reuse_cache, tokenizer, args.smoke
    )
    log("building keep-sets")
    core1 = method1(token_at, specials, byte_ids)
    core2 = method2(counter, specials, byte_ids)
    core2a = restore_ancestors(core2, parents)
    cores = {"method1": core1, "method2": core2, "method2_ancestors": core2a}
    keeps = {}
    extras = {}
    for name, core in cores.items():
        keep, extra = extend_with_indic(core, token_at, parents)
        keeps[name] = keep
        extra["dropped_ancestors"] = dropped_ancestor_count(keep, parents)
        extra["script_mix"] = script_mix(keep, token_at)
        extras[name] = extra
        log(f"{name} spark keep {len(keep)}")
    extras = {
        "method1_indic": extras["method1"],
        "method2_indic": extras["method2"],
        "method2_ancestors_indic": extras["method2_ancestors"],
    }
    cuts = {
        "method1_indic": Cut(keep_ids=keeps["method1"]),
        "method2_indic": Cut(keep_ids=keeps["method2"]),
        "method2_ancestors_indic": Cut(keep_ids=keeps["method2_ancestors"]),
        "script_cut": Cut(),
    }
    for name, cut in cuts.items():
        if name == "script_cut":
            extras[name] = {
                "core_ids": None,
                "indic_ids_not_in_core": None,
                "indic_ancestor_ids_not_in_core": None,
                "dropped_ancestors": dropped_ancestor_count(
                    set(cut.old_to_new), parents
                ),
                "script_mix": script_mix(set(cut.old_to_new), token_at),
            }
        extras[name]["spark_kept"] = cut.counts()["spark_kept"]
        extras[name]["sarvam_reused"] = cut.counts()["sarvam_reused"]
        extras[name]["sarvam_added"] = cut.counts()["sarvam_added"]
        extras[name]["vocab_size"] = cut.counts()["vocab_size"]
        extras[name]["closure_ok"] = cut.closure_ok
        log(
            f"{name} vocab {extras[name]['vocab_size']} "
            f"reused {extras[name]['sarvam_reused']} added {extras[name]['sarvam_added']}"
        )
    donor = None
    donor_error = None
    try:
        donor = donor_encoder()
        log("loaded Sarvam 65k encoder")
    except Exception as exc:
        donor_error = str(exc)
        log(f"Sarvam 65k encoder unavailable: {exc}")
    domain_order = list(evaluation)
    fertility = {domain: {} for domain in domain_order}
    parent = {domain: {} for domain in domain_order}
    prepared = {domain: [] for domain in domain_order}
    spark_keeps = {
        "method1_indic": keeps["method1"],
        "method2_indic": keeps["method2"],
        "method2_ancestors_indic": keeps["method2_ancestors"],
        "script_cut": set(cuts["script_cut"].old_to_new),
    }
    for domain, payload in evaluation.items():
        texts = payload["texts"]
        log(f"encoding {domain}, {len(texts)} documents")
        stock_ids = []
        per_cut = {name: [] for name in cuts}
        donor_docs = []
        identical = {name: [] for name in cuts}
        donor_failed = None
        for text in texts:
            stock_encoded = tokenizer.encode(text)
            stock_ids.append(stock_encoded.ids)
            words, nbytes = stock_words_bytes(text)
            score_ids = stock_encoded.ids[:LOSS_TOKENS]
            truncated = len(stock_encoded.ids) > LOSS_TOKENS
            if len(score_ids) < 2:
                score_bytes, score_words = 0, 0
            elif truncated:
                score_words, score_bytes = stock_words_bytes(tokenizer.decode(score_ids))
            else:
                score_words, score_bytes = words, nbytes
            stock_doc = {
                "ids": stock_encoded.ids,
                "score_ids": score_ids,
                "words": words,
                "bytes": nbytes,
                "score_words": score_words,
                "score_bytes": score_bytes,
                "truncated": truncated,
            }
            row = {"stock": stock_doc}
            routed = has_target_script(text)
            for name, cut in cuts.items():
                doc = prepare_document(text, cut.encode_ids, cut.decode_ids)
                per_cut[name].append(doc)
                row[name] = doc
                if not routed:
                    identical[name].append((
                        stock_encoded.tokens,
                        cut.compact.encode(text).tokens,
                    ))
            if donor is not None and donor_failed is None:
                try:
                    donor_ids = donor.encode(text)
                except Exception as exc:
                    donor_failed = f"{type(exc).__name__}: {exc}"
                    donor_docs = []
                    log(f"Sarvam 65k failed on {domain}: {donor_failed}")
                else:
                    donor_docs.append({
                        "ids": donor_ids,
                        "words": words,
                        "bytes": nbytes,
                    })
            prepared[domain].append(row)
        fertility[domain]["stock"] = fertility_of([row["stock"] for row in prepared[domain]])
        fertility[domain]["stock"]["ratio_vs_stock"] = 1.0
        fertility[domain]["stock"]["identical_documents"] = 1.0
        stock_fert = fertility[domain]["stock"]
        for name in cuts:
            stats = fertility_of(per_cut[name])
            stats["ratio_vs_stock"] = (
                round(stats["tokens"] / stock_fert["tokens"], 4) if stock_fert["tokens"] else None
            )
            stats["identical_documents"] = identical_fraction(identical[name])
            fertility[domain][name] = stats
            parent[domain][name] = parent_walk(stock_ids, spark_keeps[name], parents)
        if donor_docs:
            stats = fertility_of(donor_docs)
            stats["ratio_vs_stock"] = (
                round(stats["tokens"] / stock_fert["tokens"], 4) if stock_fert["tokens"] else None
            )
            stats["identical_documents"] = None
            fertility[domain]["sarvam65k"] = stats
        else:
            fertility[domain]["sarvam65k"] = None
        log(
            f"{domain} fertility stock {stock_fert['tokens_per_word']} "
            f"method2 {fertility[domain]['method2_indic']['tokens_per_word']}"
        )
    thread.join()
    if box and box[0] is not None:
        raise box[0]
    loss = None
    window = None
    loss_error = None
    if not args.skip_loss:
        try:
            loss, window = run_loss(cuts, prepared, None)
            loss = loss_by_domain(loss)
        except Exception as exc:
            loss_error = f"{type(exc).__name__}: {exc}"
            log(f"loss failed: {loss_error}")
            raise
    if args.smoke:
        log(json.dumps({
            "smoke": True,
            "vocab": {name: extras[name]["vocab_size"] for name in cuts},
            "loss_window": window,
            "english_bits": None if loss is None else {
                name: loss[name]["english"]["bits_per_byte"] for name in loss
            },
        }))
        return
    if loss is None:
        raise RuntimeError(loss_error or "loss was not computed")
    (Path(args.cache_dir) / "loss.json").write_text(json.dumps(loss), encoding="utf-8")
    summary = {
        "spark_sha256": cuts["script_cut"].spark_sha256,
        "donor_sha256": cuts["script_cut"].donor_sha256,
        "model": MODEL_ID,
        "instance_id": os.environ.get("MEASURE_INSTANCE_ID", "local"),
        "freq_chars": FREQ_CHARS,
        "slice_chars": SLICE_CHARS,
        "loss_tokens": LOSS_TOKENS,
        "loss_window": window,
        "donor_error": donor_error,
        "frequency": freq_domains,
        "frequency_tokens": int(sum(counter.values())),
        "frequency_types": sum(1 for count in counter.values() if count > 0),
        "evaluation": {
            name: {"key": row["key"], "documents": row["documents"], "tokens": row["tokens"]}
            for name, row in evaluation.items()
        },
        "domain_order": domain_order,
        "overlap_method2": len(core1 & core2),
        "overlap_method2_ancestors": len(core1 & core2a),
        "tokenizers": extras,
        "fertility": fertility,
        "parent_walk": parent,
        "loss": loss,
    }
    summary["recommendation"] = recommend(summary)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cut_name in (
        ("method1", "method1_indic"),
        ("method2", "method2_indic"),
        ("method2_ancestors", "method2_ancestors_indic"),
    ):
        write_json(REPORT_DIR / f"s3_keep_{name}_plus_indic.json", sorted(keeps[name]))
        write_json(
            REPORT_DIR / f"s3_keep_{name}_appended_pieces.json",
            cuts[cut_name].new_pieces,
        )
    (REPORT_DIR / "s3_keepset_fertility_loss.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (REPORT_DIR / "s3_keepset_fertility_loss.md").write_text(
        render(summary), encoding="utf-8"
    )
    log("wrote reports/s3_keepset_fertility_loss.md")
    print(json.dumps({
        "vocab": {name: extras[name]["vocab_size"] for name in cuts},
        "english_bits": {
            name: loss["english"][name]["bits_per_byte"] for name in loss["english"]
        },
        "english_fertility": {
            name: fertility["english"][name]["ratio_vs_stock"]
            for name in ("stock", "script_cut", "method1_indic", "method2_indic", "method2_ancestors_indic")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
