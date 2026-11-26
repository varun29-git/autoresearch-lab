"""Probe 2: tokenizer fertility on FLORES-200 — Spark vs Sarvam-1 (Indic-native anchor)."""
import json
import os

from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LANGS, fetch_flores

MODELS = {
    "spark": "XHToken/Spark-X2.6-1.7B",
    "sarvam1": "sarvamai/sarvam-1",
}

corpus = fetch_flores("devtest", max_lines=200)
report = {}

for name, repo in MODELS.items():
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    except Exception as e:
        report[name] = {"error": str(e)[:300]}
        print(f"{name}: tokenizer load failed: {e}")
        continue
    m = {"vocab_size": tok.vocab_size, "langs": {}}
    for lang, lines in corpus.items():
        n_tokens = 0
        n_words = 0
        n_bytes = 0
        for line in lines:
            enc = tok.encode(line, add_special_tokens=False)
            n_tokens += len(enc)
            n_words += len(line.split())
            n_bytes += len(line.encode("utf-8"))
        m["langs"][lang] = {
            "tokens_per_word": round(n_tokens / max(n_words, 1), 3),
            "tokens_per_byte": round(n_tokens / max(n_bytes, 1), 4),
            "total_tokens": n_tokens,
        }
    report[name] = m
    print(f"{name} done")

os.makedirs("results", exist_ok=True)
with open("results/probe2_fertility.json", "w") as f:
    json.dump(report, f, indent=2)

# readable table
langs = [l for l in LANGS if l in corpus]
hdr = ["lang"] + list(MODELS.keys())
print("\t".join(hdr))
for lang in langs:
    row = [lang]
    for name in MODELS:
        v = report.get(name, {}).get("langs", {}).get(lang, {})
        row.append(str(v.get("tokens_per_word", "ERR")))
    print("\t".join(row))
