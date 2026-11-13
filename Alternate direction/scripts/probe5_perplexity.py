"""Probe 5: Indic perplexity probe on FLORES-200 devtest.

Models: Spark base (primary), Spark instruct, Sarvam-1 (Indic-native anchor).
Corpus perplexity per language = exp(sum NLL / sum tokens).
"""
import json
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LANGS, fetch_flores

MAX_LINES = 200  # per language per model; full devtest is 1012
BATCH = 8

MODELS = {
    "spark_base": "XHToken/Spark-X2.6-1.7B-Base",
    "spark_instruct": "XHToken/Spark-X2.6-1.7B",
    "sarvam1": "sarvamai/sarvam-1",
}

corpus = fetch_flores("devtest", max_lines=MAX_LINES)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")
report = {"max_lines_per_lang": MAX_LINES, "models": {}}

for name, repo in MODELS.items():
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(device).eval()
    except Exception as e:
        report["models"][name] = {"error": str(e)[:300]}
        print(f"{name}: LOAD FAILED: {e}")
        continue
    mres = {}
    for lang in LANGS:
        if lang not in corpus:
            continue
        lines = corpus[lang]
        total_nll = 0.0
        total_tok = 0
        for i in range(0, len(lines), BATCH):
            batch = lines[i:i + BATCH]
            encs = [tok.encode(t, add_special_tokens=False, return_tensors="pt") for t in batch]
            maxlen = max(e.shape[1] for e in encs)
            if maxlen < 2:
                continue
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            input_ids = torch.full((len(encs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(encs), maxlen), dtype=torch.long)
            for j, e in enumerate(encs):
                L = e.shape[1]
                input_ids[j, :L] = e[0]
                attn[j, :L] = 1
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn).logits
            tgt = input_ids[:, 1:]
            mask = attn[:, 1:]
            # chunked log_softmax to avoid materializing fp32 logits over 131k vocab
            for j in range(logits.shape[0]):
                lp = torch.log_softmax(logits[j, :-1].float(), dim=-1)
                nll = -lp.gather(-1, tgt[j].unsqueeze(-1)).squeeze(-1) * mask[j]
                total_nll += nll.sum().item()
            total_tok += mask.sum().item()
            del logits
        ppl = math.exp(total_nll / max(total_tok, 1))
        mres[lang] = {"ppl": round(ppl, 3), "tokens": total_tok}
        print(f"{name} {lang}: ppl={ppl:.2f} ({total_tok} toks)")
    report["models"][name] = mres
    del model
    torch.cuda.empty_cache()

os.makedirs("results", exist_ok=True)
with open("results/probe5_perplexity.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n=== corpus ppl (FLORES-200 devtest, 200 lines/lang) ===")
langs = [l for l in LANGS if l in corpus]
print("lang\t" + "\t".join(MODELS))
for lang in langs:
    row = [lang]
    for name in MODELS:
        v = report["models"].get(name, {}).get(lang, {})
        row.append(str(v.get("ppl", "ERR")))
    print("\t".join(row))
