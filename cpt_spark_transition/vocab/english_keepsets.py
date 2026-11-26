"""Two ~32k English keep-sets. No Indic pieces, no weights, no new tokenizer.

Method 1 keeps the earliest 32,000 non-special Spark ids, plus specials and the
byte symbols Spark actually has. Method 2 keeps specials and byte symbols, then
fills to 32,000 with the most common ids on a local English, code, and math
sample. A third list adds the merge ancestors of that frequency set.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from .indic import BYTE_ALPHABET, is_target_char
from .model import SPARK_PATH, SPARK_SHA256, decode_token, has_dropped_script, sha256

FOLDER = Path(__file__).resolve().parents[1]
REPORT_DIR = FOLDER / "reports"
TARGET = 32_000
PINNED = 14
CODE_TRAIN_CAP = 30_000
CODE_HELD_CAP = 1_500
ENGLISH_HELD_CAP = 1_500
MATH_HELD_CAP = 400
GSM_HELD_CAP = 200


def load_spark():
    digest = sha256(SPARK_PATH)
    if digest != SPARK_SHA256:
        raise ValueError(f"Spark tokenizer sha256 {digest} does not match {SPARK_SHA256}")
    with SPARK_PATH.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    vocab = raw["model"]["vocab"]
    token_at = {index: token for token, index in vocab.items()}
    specials = set(range(PINNED))
    for item in raw["added_tokens"]:
        if item.get("special"):
            specials.add(item["id"])
    byte_ids = {vocab[char] for char in BYTE_ALPHABET if char in vocab}
    parents = {}
    for pair in raw["model"]["merges"]:
        left, right = pair if not isinstance(pair, str) else pair.split(" ")
        produced = left + right
        if produced in parents or produced not in vocab or left not in vocab or right not in vocab:
            continue
        parents[vocab[produced]] = (vocab[left], vocab[right])
    return raw, vocab, token_at, specials, byte_ids, parents, Tokenizer.from_file(str(SPARK_PATH))


def method1(token_at, specials, byte_ids):
    non_special = [index for index in range(len(token_at)) if index not in specials]
    if len(non_special) < TARGET:
        raise RuntimeError("tokenizer has fewer than 32,000 non-special ids")
    chosen = set(non_special[:TARGET])
    return chosen | specials | byte_ids


def texts_of(rows, *fields):
    out = []
    for row in rows:
        parts = []
        for field in fields:
            value = row[field]
            if isinstance(value, str) and value:
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(item for item in value if isinstance(item, str))
        if parts:
            out.append("\n".join(parts))
    return out


def take_codesearch(cap_train, cap_held):
    from datasets import load_dataset

    train, held = [], []
    dataset = load_dataset("kejian/codesearchnet-python-raw", split="train", streaming=True)
    for row in dataset:
        bucket = train if row["partition"] == "train" else held if row["partition"] == "valid" else None
        if bucket is None:
            continue
        limit = cap_train if bucket is train else cap_held
        if len(bucket) >= limit:
            if len(train) >= cap_train and len(held) >= cap_held:
                break
            continue
        text = "\n".join(part for part in (row["code"], row["docstring"]) if part)
        if text:
            bucket.append(text)
    return train, held


def load_corpora():
    from datasets import load_dataset

    hellaswag_train = load_dataset("Rowan/hellaswag", split="train")
    hellaswag_valid = load_dataset("Rowan/hellaswag", split="validation")
    english_train = texts_of(hellaswag_train, "ctx", "endings", "activity_label")
    english_held = texts_of(list(hellaswag_valid)[:ENGLISH_HELD_CAP], "ctx", "endings", "activity_label")
    code_train, code_held = take_codesearch(CODE_TRAIN_CAP, CODE_HELD_CAP)
    math_train, math_held = [], []
    for config in (
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"):
        train = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        test = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        math_train.extend(texts_of(train, "problem", "solution"))
        math_held.extend(texts_of(test, "problem", "solution"))
    math_held = math_held[:MATH_HELD_CAP]
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    math_train.extend(texts_of(gsm_train, "question", "answer"))
    math_held.extend(texts_of(list(gsm_test)[:GSM_HELD_CAP], "question", "answer"))
    return {
        "english": (english_train, english_held),
        "code": (code_train, code_held),
        "math": (math_train, math_held),
    }


def encode_all(tokenizer, texts):
    ids = []
    words = 0
    for start in range(0, len(texts), 256):
        chunk = texts[start:start + 256]
        words += sum(len(text.split()) for text in chunk)
        for encoded in tokenizer.encode_batch(chunk):
            ids.extend(encoded.ids)
    return ids, words


def method2(counts, specials, byte_ids):
    forced = set(specials) | set(byte_ids)
    if len(forced) > TARGET:
        raise RuntimeError("specials and byte symbols already exceed 32,000")
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    chosen = set(forced)
    for index, _count in ranked:
        if index in chosen:
            continue
        chosen.add(index)
        if len(chosen) == TARGET:
            break
    return chosen


def restore_ancestors(keep, parents):
    restored = set(keep)
    stack = list(keep)
    while stack:
        index = stack.pop()
        pair = parents.get(index)
        if pair is None:
            continue
        for parent in pair:
            if parent not in restored:
                restored.add(parent)
                stack.append(parent)
    return restored


def dropped_ancestor_count(keep, parents):
    """Kept ids whose merge chain includes an id outside the keep set."""
    memo = {}

    def broken(index):
        cached = memo.get(index)
        if cached is not None:
            return cached
        pair = parents.get(index)
        if pair is None:
            memo[index] = False
            return False
        result = any(parent not in keep or broken(parent) for parent in pair)
        memo[index] = result
        return result

    return sum(1 for index in keep if broken(index))


def lengths_for(keep, parents):
    formable = {}
    length = {}

    def can_form(index):
        cached = formable.get(index)
        if cached is not None:
            return cached
        pair = parents.get(index)
        if pair is None:
            formable[index] = index in keep
            return formable[index]
        result = index in keep and can_form(pair[0]) and can_form(pair[1])
        formable[index] = result
        return result

    def span(index):
        cached = length.get(index)
        if cached is not None:
            return cached
        if can_form(index):
            length[index] = 1
            return 1
        pair = parents.get(index)
        if pair is None:
            length[index] = 1
            return 1
        result = span(pair[0]) + span(pair[1])
        length[index] = result
        return result

    return span


def coverage(id_stream, keep, span):
    missing = {index for index in id_stream if index not in keep}
    stock = len(id_stream)
    pruned = sum(span(index) for index in id_stream)
    occurrences = sum(1 for index in id_stream if index not in keep)
    return {
        "stock_tokens": stock,
        "distinct": len(set(id_stream)),
        "missing_distinct": len(missing),
        "missing_occurrences": occurrences,
        "missing_occurrence_pct": round(100 * occurrences / stock, 2) if stock else 0.0,
        "pruned_tokens": pruned,
        "fertility_ratio": round(pruned / stock, 4) if stock else None,
    }


def script_mix(keep, token_at):
    dropped = indic = other = 0
    for index in keep:
        surface = decode_token(token_at[index])
        if surface is None:
            other += 1
        elif has_dropped_script(surface):
            dropped += 1
        elif any(is_target_char(char) for char in surface):
            indic += 1
        else:
            other += 1
    return {"dropped_script": dropped, "target_indic": indic, "other": other}


def write_ids(path, keep):
    path.write_text(json.dumps(sorted(keep)), encoding="utf-8")


def pct(part, whole):
    return round(100 * part / whole, 2) if whole else 0.0


def main():
    _raw, _vocab, token_at, specials, byte_ids, parents, tokenizer = load_spark()
    keep1 = method1(token_at, specials, byte_ids)
    corpora = load_corpora()
    counts = Counter()
    encoded = {}
    sizes = {}
    for name, (train, held) in corpora.items():
        train_ids, _train_words = encode_all(tokenizer, train)
        held_ids, held_words = encode_all(tokenizer, held)
        counts.update(train_ids)
        encoded[name] = {"ids": held_ids, "words": held_words, "documents": len(held)}
        sizes[name] = {"train_documents": len(train), "train_tokens": len(train_ids),
                       "held_documents": len(held), "held_tokens": len(held_ids),
                       "held_words": held_words}
    observed = sum(1 for _index, count in counts.items() if count > 0)
    keep2 = method2(counts, specials, byte_ids)
    keep2_ancestors = restore_ancestors(keep2, parents)
    span1 = lengths_for(keep1, parents)
    span2 = lengths_for(keep2, parents)
    span2a = lengths_for(keep2_ancestors, parents)
    sets = {
        "method1": keep1,
        "method2": keep2,
        "method2_ancestors": keep2_ancestors,
    }
    spans = {"method1": span1, "method2": span2, "method2_ancestors": span2a}
    report_sets = {}
    for key, keep in sets.items():
        by_domain = {
            name: coverage(encoded[name]["ids"], keep, spans[key]) | {"words": encoded[name]["words"]}
            for name in encoded
        }
        english = by_domain["english"]
        report_sets[key] = {
            "size": len(keep),
            "dropped_ancestors": dropped_ancestor_count(keep, parents),
            "script_mix": script_mix(keep, token_at),
            "by_domain": by_domain,
            "english_tokens_per_word_stock": round(english["stock_tokens"] / english["words"], 4),
            "english_tokens_per_word_pruned": round(english["pruned_tokens"] / english["words"], 4),
        }
    overlap = {
        "method2": len(keep1 & keep2),
        "method2_ancestors": len(keep1 & keep2_ancestors),
    }
    forced = len(specials | byte_ids)
    summary = {
        "spark_sha256": SPARK_SHA256,
        "target_non_special": TARGET,
        "specials": len(specials),
        "byte_symbols_present": len(byte_ids),
        "forced_skeleton": forced,
        "frequency_token_types_seen": observed,
        "frequency_tokens_counted": int(sum(counts.values())),
        "corpora": sizes,
        "overlap_with_method1": overlap,
        "sets": report_sets,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_ids(REPORT_DIR / "keep_method1_bpe_order.json", keep1)
    write_ids(REPORT_DIR / "keep_method2_frequency.json", keep2)
    write_ids(REPORT_DIR / "keep_method2_frequency_ancestors.json", keep2_ancestors)
    (REPORT_DIR / "english_32k_keepsets.md").write_text(render(summary), encoding="utf-8")
    (REPORT_DIR / "english_32k_keepsets_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "method1": report_sets["method1"]["size"],
        "method2": report_sets["method2"]["size"],
        "method2_ancestors": report_sets["method2_ancestors"]["size"],
        "missing_en": {
            key: report_sets[key]["by_domain"]["english"]["missing_distinct"] for key in report_sets
        },
    }, indent=2))


def render(summary):
    method1 = summary["sets"]["method1"]
    method2 = summary["sets"]["method2"]
    method2a = summary["sets"]["method2_ancestors"]
    corpora = summary["corpora"]
    lines = [
        "# Frequent-English keep-sets, about 32k",
        "",
        "No Indic pieces were added. No embedding rows were deleted. No tokenizer file was written.",
        "The id lists are the analysis. A later cut can apply one of them.",
        "",
        "Stock tokenizer: `research/spark_tokenizer/tokenizer.json`,",
        f"sha256 `{summary['spark_sha256']}`.",
        "",
        "The CI-1 training bucket is not on this machine, so the frequency count uses",
        "local caches that are already downloaded. They are not the pretraining mix.",
        "",
        "| Domain | Frequency documents | Frequency tokens | Held-out documents | Held-out tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "english": "HellaSwag train / validation",
        "code": "CodeSearchNet Python train / valid",
        "math": "MATH train plus GSM8K train / MATH test plus GSM8K test",
    }
    for name in ("english", "code", "math"):
        row = corpora[name]
        lines.append(
            f"| {labels[name]} | {row['train_documents']} | {row['train_tokens']} | "
            f"{row['held_documents']} | {row['held_tokens']} |"
        )
    lines.extend([
        "",
        f"Frequency count saw {summary['frequency_token_types_seen']} distinct ids "
        f"across {summary['frequency_tokens_counted']} tokens. "
        f"The forced skeleton is {summary['specials']} specials "
        f"(ids 0–13 plus every added token Spark marks special) and "
        f"{summary['byte_symbols_present']} byte symbols present in the vocab. "
        "Spark does not allocate the other 13 byte symbols.",
        "",
        "## Side by side",
        "",
        "| | Method 1, earliest ids | Method 2, frequency | Method 2 plus ancestors |",
        "| --- | ---: | ---: | ---: |",
        f"| Ids kept | {method1['size']} | {method2['size']} | {method2a['size']} |",
        f"| Overlap with method 1 | {method1['size']} | "
        f"{summary['overlap_with_method1']['method2']} | "
        f"{summary['overlap_with_method1']['method2_ancestors']} |",
        f"| Kept ids with a dropped merge ancestor | {method1['dropped_ancestors']} | "
        f"{method2['dropped_ancestors']} | {method2a['dropped_ancestors']} |",
        f"| Kept ids whose surface is a dropped script | {method1['script_mix']['dropped_script']} | "
        f"{method2['script_mix']['dropped_script']} | {method2a['script_mix']['dropped_script']} |",
        f"| Kept ids whose surface is a target Indic script | {method1['script_mix']['target_indic']} | "
        f"{method2['script_mix']['target_indic']} | {method2a['script_mix']['target_indic']} |",
    ])
    for domain, title in (("english", "English"), ("code", "Code"), ("math", "Math")):
        lines.append(
            f"| {title} held-out distinct ids missing | "
            f"{method1['by_domain'][domain]['missing_distinct']} | "
            f"{method2['by_domain'][domain]['missing_distinct']} | "
            f"{method2a['by_domain'][domain]['missing_distinct']} |"
        )
        lines.append(
            f"| {title} held-out token occurrences missing | "
            f"{method1['by_domain'][domain]['missing_occurrence_pct']}% | "
            f"{method2['by_domain'][domain]['missing_occurrence_pct']}% | "
            f"{method2a['by_domain'][domain]['missing_occurrence_pct']}% |"
        )
    lines.extend([
        f"| English tokens per word, stock | {method1['english_tokens_per_word_stock']} | "
        f"{method2['english_tokens_per_word_stock']} | {method2a['english_tokens_per_word_stock']} |",
        f"| English tokens per word, after the cut | {method1['english_tokens_per_word_pruned']} | "
        f"{method2['english_tokens_per_word_pruned']} | {method2a['english_tokens_per_word_pruned']} |",
        f"| English fertility ratio, cut / stock | {method1['by_domain']['english']['fertility_ratio']} | "
        f"{method2['by_domain']['english']['fertility_ratio']} | "
        f"{method2a['by_domain']['english']['fertility_ratio']} |",
        "",
        "A missing id is a stock token on the held-out text that is not in the keep list.",
        "Fertility recomputes the length by walking merge parents whenever a token,",
        "or one of its ancestors, is not kept. A kept id whose parent was dropped",
        "cannot be emitted by the pruned merge table, so it is expanded too.",
        "",
        "## Method 1",
        "",
        "Non-special means every id outside the special set. The keep list is the",
        f"first {TARGET} of those ids, in id order, union the specials and the byte symbols.",
        f"That is {method1['size']} ids. "
        f"{method1['dropped_ancestors']} of them have a merge ancestor outside the list. "
        f"{method1['script_mix']['dropped_script']} kept surfaces are in a script this",
        "experiment is not treating as English.",
        "",
        "## Method 2",
        "",
        "Specials and byte symbols are forced in. The rest of the 32,000 slots are",
        "the most common remaining ids on the frequency sample. Ties break toward",
        "the lower id. The ancestor version starts from that same 32,000 and then",
        f"adds every missing merge parent, which brings it to {method2a['size']} ids.",
        f"Without that restore, {method2['dropped_ancestors']} kept ids are unreachable.",
        "With the restore, that count is "
        f"{method2a['dropped_ancestors']}.",
        "",
        "## Recommendation",
        "",
        recommendation(summary),
        "",
        "Lists:",
        "",
        "- `reports/keep_method1_bpe_order.json`",
        "- `reports/keep_method2_frequency.json`",
        "- `reports/keep_method2_frequency_ancestors.json`",
        "",
    ])
    return "\n".join(lines)


def recommendation(summary):
    method1 = summary["sets"]["method1"]
    method2 = summary["sets"]["method2"]
    method2a = summary["sets"]["method2_ancestors"]
    safer = min(
        (("method 2 with ancestors", method2a),
         ("method 2 without ancestors", method2),
         ("method 1", method1)),
        key=lambda item: (
            item[1]["by_domain"]["english"]["fertility_ratio"],
            item[1]["by_domain"]["english"]["missing_distinct"],
            item[1]["dropped_ancestors"],
        ),
    )
    return (
        f"For English, {safer[0]} is the safer of these three lists. "
        f"Its English fertility ratio is {safer[1]['by_domain']['english']['fertility_ratio']} "
        f"and it misses {safer[1]['by_domain']['english']['missing_distinct']} distinct "
        "held-out English ids. Method 1 spends keep-slots on whatever Spark merged first, "
        "including scripts that are not English. Method 2 spends them on tokens this "
        "English, code, and math sample actually emits. The ancestor restore is what "
        "makes those frequent tokens buildable. This sample is not the pretraining "
        "bucket, so the frequency ranking should be rerun on that bucket before a "
        "cut is applied."
    )


if __name__ == "__main__":
    main()
