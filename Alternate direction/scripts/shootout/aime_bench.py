#!/usr/bin/env python3
"""AIME benchmark: k samples per problem, extract boxed/final integer, exact match.

Usage:
  python aime_bench.py --model <repo> --dataset math-ai/aime25 \
      --samples 4 --batch-size 8 --max-new 8192 --out results/aime25_<tag>.json
"""
import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    """Extract final integer answer: prefer last \\boxed{...}, else last integer."""
    if not text:
        return None
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        s = m[-1].strip().replace(",", "").replace("$", "").strip()
        mm = re.search(r"-?\d+", s)
        if mm:
            return int(mm.group(0))
    # fall back: last standalone integer in the text
    nums = re.findall(r"-?\d[\d,]*", text)
    for s in reversed(nums):
        try:
            return int(s.replace(",", ""))
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    print(f"dataset={args.dataset} split={args.split} n={len(ds)} features={ds.features}")

    def pick(row, keys):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

    problems = []
    for row in ds:
        q = pick(row, ["problem", "question", "prompt"])
        a = pick(row, ["answer", "final_answer", "solution"])
        if isinstance(a, str):
            a = extract_answer(a) if not a.strip().isdigit() else int(a.strip())
        problems.append({"question": q, "answer": int(a)})
    if args.limit:
        problems = problems[: args.limit]
    print(f"n_problems={len(problems)} first_answer={problems[0]['answer']}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    # some model cards ship top_k=-1 (vLLM-style "disabled"); HF requires positive int
    if getattr(model.generation_config, "top_k", None) is not None and model.generation_config.top_k <= 0:
        model.generation_config.top_k = None

    # build prompts
    prompts = []
    for p in problems:
        msgs = [{"role": "user", "content": p["question"]}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if args.reasoning:
            kwargs["reasoning_effort"] = args.reasoning
        try:
            prompts.append(tok.apply_chat_template(msgs, **kwargs))
        except TypeError:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    # jobs: (problem_idx, sample_idx)
    jobs = [(i, s) for i in range(len(problems)) for s in range(args.samples)]
    results = {i: {"idx": i, "answer": problems[i]["answer"], "preds": [], "correct": [], "gen_tokens": []} for i in range(len(problems))}

    t_start = time.time()
    done = 0
    for bstart in range(0, len(jobs), args.batch_size):
        batch = jobs[bstart : bstart + args.batch_size]
        bprompts = [prompts[i] for i, _ in batch]
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
        for j, (i, s) in enumerate(batch):
            text = tok.decode(out[j][plen:], skip_special_tokens=True)
            pred = extract_answer(text)
            gold = problems[i]["answer"]
            results[i]["preds"].append(pred)
            results[i]["correct"].append(bool(pred is not None and pred == gold))
            results[i]["gen_tokens"].append(int(out.shape[1] - plen))
        done += len(batch)
        el = time.time() - t_start
        print(f"progress {done}/{len(jobs)} elapsed={el/60:.1f}min eta={(el/done)*(len(jobs)-done)/60:.1f}min", flush=True)

    accs = []
    for i in results:
        r = results[i]
        r["acc"] = sum(r["correct"]) / max(1, len(r["correct"]))
        accs.append(r["acc"])
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_problems": len(problems),
        "samples_per_problem": args.samples,
        "mean_acc": sum(accs) / max(1, len(accs)),
        "max_new_tokens": args.max_new,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "do_sample": True},
        "reasoning_effort": args.reasoning,
        "wall_minutes": (time.time() - t_start) / 60,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    out_obj = {"summary": summary, "per_problem": [results[i] for i in sorted(results)]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("AIME_BENCH_OK")


if __name__ == "__main__":
    main()
