"""GPQA-Diamond only run (uses fast_bench machinery). usage: gpqa_only.py <model> <tag> <outdir>"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_bench as fb

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_repo, tag, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_repo, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
import re
qs, golds = [], []
for r in ds:
    m = re.search(r"\\boxed\{[^}]*?([A-D])[^}]*?\}", str(r["solution"]))
    qs.append(r["problem"])
    golds.append(m.group(1) if m else None)

t0 = time.time()
outs = fb.generate_all(model, tok, fb.make_prompts(tok, qs), 16, 2048)
scored = [(fb.extract_letter(o or ""), g) for o, g in zip(outs, golds) if g]
correct = sum(1 for p, g in scored if p == g)
res = {"model": model_repo, "gpqa_diamond": {
    "n": len(scored), "acc": round(100 * correct / max(len(scored), 1), 2),
    "minutes": round((time.time() - t0) / 60, 1)}}
json.dump(res, open(os.path.join(outdir, f"gpqa_{tag}.json"), "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
print("SAVED")
