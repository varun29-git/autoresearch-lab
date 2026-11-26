"""Probe 8: validate the 96k plan — prune ALL CJK + all small scripts (keep Arabic+Indic),
check tokenization equivalence on English/Indic/code. CPU-only.

Variant C: remove cjk + [cyrillic, greek, hebrew, hangul, kana, myanmar, ethiopic,
armenian, khmer, georgian, tibetan, lao, mongolian, thai, sinhala] tokens.
"""
import json
import os
import sys

from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BYTE_DECODER, INDIC_SCRIPTS, classify, decode_token,
                    fetch_flores, load_vocab, parse_merges)

CODE_SAMPLE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "probe3_prune_dryrun.py")).read()  # any python source

tj, vocab, merges_raw = load_vocab("models/spark/tokenizer.json")
merges = parse_merges(merges_raw)

DROP_SCRIPTS = {"cjk", "cjk_ext_a", "cyrillic", "hangul", "kana", "sinhala"}
EXTRA_RANGES = [  # scripts not in common.SCRIPT_RANGES keyed names we drop:
    ("greek", 0x0370, 0x03FF), ("hebrew", 0x0590, 0x05FF),
    ("armenian", 0x0530, 0x058F), ("georgian", 0x10A0, 0x10FF),
    ("ethiopic", 0x1200, 0x137F), ("khmer", 0x1780, 0x17FF),
    ("myanmar", 0x1000, 0x109F), ("thai", 0x0E00, 0x0E7F),
    ("lao", 0x0E80, 0x0EFF), ("tibetan", 0x0F00, 0x0FFF),
    ("mongolian", 0x1800, 0x18AF),
]
KEEP_SCRIPTS = INDIC_SCRIPTS | {"ascii", "arabic"}


def droppable(s):
    """True if token contains only droppable-script chars (no keep-script chars)."""
    has_drop, has_keep = False, False
    for ch in s:
        o = ord(ch)
        if o < 128:
            has_keep = True  # any ASCII char => keep (mixed token)
            continue
        cats = classify(ch)
        if cats & KEEP_SCRIPTS:
            has_keep = True
        elif cats & DROP_SCRIPTS or any(lo <= o <= hi for _, lo, hi in EXTRA_RANGES):
            has_drop = True
        else:
            has_keep = True  # unknown/other => conservative keep
    return has_drop and not has_keep


def rebuild():
    pruned = set()
    for tok in vocab:
        if len(tok) == 1:
            continue
        s, valid = decode_token(tok)
        if valid and s and droppable(s):
            pruned.add(tok)
    kept_merges, removed, produced = [], set(pruned), set()
    for a, b in merges:
        out = a + b
        if a in removed or b in removed or out in removed:
            removed.add(out)
            continue
        kept_merges.append([a, b])
        produced.add(out)
    always_keep = {t for t in vocab
                   if len(t) == 1 or any(c not in BYTE_DECODER for c in t)}
    kept_tokens = {t for t in vocab
                   if t not in removed and (t in always_keep or t in produced)}
    ordered = sorted(kept_tokens, key=lambda t: vocab[t])
    new_vocab = {t: i for i, t in enumerate(ordered)}
    new_tj = dict(tj)
    new_tj["model"] = dict(tj["model"])
    new_tj["model"]["vocab"] = new_vocab
    new_tj["model"]["merges"] = kept_merges
    return new_tj, {"tokens_pruned": len(pruned),
                    "tokens_removed_total": len(vocab) - len(new_vocab),
                    "merges_removed": len(merges) - len(kept_merges),
                    "new_vocab_size": len(new_vocab)}


corpus = fetch_flores("devtest", max_lines=300)
corpus["code_python"] = [CODE_SAMPLE]

orig = Tokenizer.from_str(json.dumps(tj))
new_tj, stats = rebuild()
print(json.dumps(stats, indent=1), flush=True)
new = Tokenizer.from_str(json.dumps(new_tj))

report = {"stats": stats, "equivalence": {}}
for lang, lines in corpus.items():
    old = [orig.encode(t, add_special_tokens=False).tokens for t in lines]
    new_ids = [new.encode(t, add_special_tokens=False).tokens for t in lines]
    ident = sum(1 for a, b in zip(old, new_ids) if a == b)
    ol, nl = sum(len(x) for x in old), sum(len(x) for x in new_ids)
    report["equivalence"][lang] = {
        "pct_identical": round(100 * ident / len(lines), 2),
        "token_growth_pct": round(100 * (nl - ol) / max(ol, 1), 2)}
    print(f"{lang:12s} identical={report['equivalence'][lang]['pct_identical']:6.2f}% "
          f"growth={report['equivalence'][lang]['token_growth_pct']:+6.2f}%", flush=True)

os.makedirs("results", exist_ok=True)
json.dump(report, open("results/probe8_prune_96k.json", "w"), indent=1)
print("saved results/probe8_prune_96k.json")
