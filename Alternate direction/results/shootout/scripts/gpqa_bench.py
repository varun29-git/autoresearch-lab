#!/usr/bin/env python3
"""GPQA Diamond (0-shot CoT, chat template): extract boxed letter, exact match.

Dataset: hendrydong/gpqa_diamond_mc (mirror of GPQA diamond, 198 questions,
choices embedded in the problem text, gold = \\boxed{<letter>}).
The official Idavidrein/gpqa dataset is gated; this mirror carries the same
diamond questions with fixed choice ordering. Identical prompt/ordering for all
models, so the comparison is internally fair.

Usage: python gpqa_bench.py --model <repo> --out results/gpqa_<tag>.json
"""
import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_letter(text):
    if not text:
        return None
    m = re.findall(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", text)
    if m:
        return m[-1]
    m = re.findall(r"[Tt]he answer is \(?([A-D])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"\(([A-D])\)", text)
    if m:
        return m[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="hendrydong/gpqa_diamond_mc")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    print(f"dataset={args.dataset} n={len(ds)} features={ds.features}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    if getattr(model.generation_config, "top_k", None) is not None and model.generation_config.top_k <= 0:
        model.generation_config.top_k = None

    rows = []
    for row in ds:
        gold_m = re.search(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", row["solution"])
        rows.append({"question": row["problem"], "gold": gold_m.group(1) if gold_m else None})

    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": r["question"]}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if args.reasoning:
            kwargs["reasoning_effort"] = args.reasoning
        try:
            prompts.append(tok.apply_chat_template(msgs, **kwargs))
        except TypeError:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    t_start = time.time()
    for bstart in range(0, len(rows), args.batch_size):
        idxs = list(range(bstart, min(bstart + args.batch_size, len(rows))))
        bprompts = [prompts[i] for i in idxs]
        inputs = tok(bprompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        plen = inputs["input_ids"].shape[1]
        for j, i in enumerate(idxs):
            text = tok.decode(out[j][plen:], skip_special_tokens=True)
            pred = extract_letter(text)
            rows[i]["pred"] = pred
            rows[i]["correct"] = bool(pred is not None and pred == rows[i]["gold"])
            rows[i]["gen_tokens"] = int(out.shape[1] - plen)
        done = min(bstart + args.batch_size, len(rows))
        el = time.time() - t_start
        print(f"progress {done}/{len(rows)} elapsed={el/60:.1f}min", flush=True)

    n_correct = sum(1 for r in rows if r.get("correct"))
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "n": len(rows),
        "acc": n_correct / max(1, len(rows)),
        "n_unparsed": sum(1 for r in rows if r.get("pred") is None),
        "max_new_tokens": args.max_new,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "do_sample": True},
        "reasoning_effort": args.reasoning,
        "wall_minutes": (time.time() - t_start) / 60,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_question": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("GPQA_BENCH_OK")


if __name__ == "__main__":
    main()
