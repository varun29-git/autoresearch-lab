"""Fast unified benchmark: one model load, GSM8K + AIME25 + GPQA-D in a single process.

Identical settings for every model: temp 1.0, top_p 0.95, do_sample,
thinking high (if supported), GSM8K max_new 1024, GPQA max_new 2048, AIME max_new 4096.
Usage: python fast_bench.py --model <repo> --tag <tag> --outdir <dir>
"""
import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_int(text):
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        s = m[-1].replace(",", "").replace("$", "").strip()
        try:
            return int(float(s))
        except ValueError:
            pass
    nums = re.findall(r"-?\d[\d,]*", text)
    if nums:
        try:
            return int(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def extract_letter(text):
    m = re.findall(r"\\boxed\{[^}]*?([A-D])[^}]*?\}", text)
    if m:
        return m[-1]
    m = re.findall(r"(?:answer|ANSWER|Answer)\s*(?:is|:)?\s*\(?([A-D])\)?", text)
    if m:
        return m[-1]
    m = re.findall(r"\b([A-D])\b", text[-200:])
    return m[-1] if m else None


def make_prompts(tok, questions):
    prompts = []
    for q in questions:
        msgs = [{"role": "user", "content": q}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        reasoning_effort="high")
        except Exception:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    return prompts


def generate_all(model, tok, prompts, batch, max_new):
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), batch):
        idxs = order[s:s + batch]
        encs = [tok.encode(prompts[i], return_tensors="pt") for i in idxs]
        maxlen = max(e.shape[1] for e in encs)
        input_ids = torch.full((len(encs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(encs), maxlen), dtype=torch.long)
        for j, e in enumerate(encs):
            L = e.shape[1]
            input_ids[j, maxlen - L:] = e[0]
            attn[j, maxlen - L:] = 1
        with torch.no_grad():
            g = model.generate(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                               max_new_tokens=max_new, do_sample=True, temperature=1.0,
                               top_p=0.95, top_k=0, pad_token_id=pad_id)
        for j, i in enumerate(idxs):
            outs[i] = tok.decode(g[j][maxlen:], skip_special_tokens=True)
        del g
        torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gsm8k-cap", type=int, default=1024)
    ap.add_argument("--gpqa-cap", type=int, default=2048)
    ap.add_argument("--aime-cap", type=int, default=4096)
    ap.add_argument("--aime-samples", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    report = {"model": args.model, "settings": {
        "temperature": 1.0, "top_p": 0.95, "do_sample": True, "thinking": "high",
        "gsm8k_cap": args.gsm8k_cap, "gpqa_cap": args.gpqa_cap,
        "aime_cap": args.aime_cap, "aime_samples": args.aime_samples}}

    # ---- GSM8K ----
    t0 = time.time()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    qs = [r["question"] for r in ds]
    golds = [extract_int(r["answer"]) for r in ds]
    outs = generate_all(model, tok, make_prompts(tok, qs), 32, args.gsm8k_cap)
    correct = sum(1 for o, g in zip(outs, golds) if extract_int(o or "") == g)
    report["gsm8k"] = {"n": len(qs), "acc": round(100 * correct / len(qs), 2),
                       "minutes": round((time.time() - t0) / 60, 1)}
    print(f"GSM8K {report['gsm8k']}", flush=True)

    # ---- AIME 2025 ----
    t0 = time.time()
    ds = load_dataset("math-ai/aime25", split="test")
    qs, golds = [], []
    for r in ds:
        a = r.get("answer")
        try:
            golds.append(int(a))
            qs.append(r["problem"] if "problem" in r else r["question"])
        except (TypeError, ValueError):
            pass
    all_q, all_g = [], []
    for q, g in zip(qs, golds):
        all_q += [q] * args.aime_samples
        all_g += [g] * args.aime_samples
    outs = generate_all(model, tok, make_prompts(tok, all_q), 16, args.aime_cap)
    correct = sum(1 for o, g in zip(outs, all_g) if extract_int(o or "") == g)
    report["aime25"] = {"n_problems": len(qs), "samples": args.aime_samples,
                        "mean_acc": round(100 * correct / len(all_q), 2),
                        "minutes": round((time.time() - t0) / 60, 1)}
    print(f"AIME25 {report['aime25']}", flush=True)

    # ---- GPQA Diamond ----
    t0 = time.time()
    ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    sample = ds[0]
    qcol = "question" if "question" in sample else list(sample)[0]
    qs, golds = [], []
    for r in ds:
        q = r[qcol]
        gold = None
        for cand in ["answer", "label", "correct", "gold", "solution"]:
            if cand in r:
                v = str(r[cand]).strip().upper()
                if v in "ABCD" and len(v) == 1:
                    gold = v
                    break
                m = re.search(r"\\boxed\{[^}]*?([A-D])[^}]*?\}", v)
                if m:
                    gold = m.group(1)
                    break
                m = re.search(r"\b([A-D])\b", v)
                if m:
                    gold = m.group(1)
                    break
        qs.append(q)
        golds.append(gold)
    outs = generate_all(model, tok, make_prompts(tok, qs), 16, args.gpqa_cap)
    scored = [(extract_letter(o or ""), g) for o, g in zip(outs, golds) if g]
    correct = sum(1 for p, g in scored if p == g)
    report["gpqa_diamond"] = {"n": len(scored), "acc": round(100 * correct / max(len(scored), 1), 2),
                              "minutes": round((time.time() - t0) / 60, 1),
                              "example_row_keys": list(sample)}
    print(f"GPQA {report['gpqa_diamond']}", flush=True)

    with open(os.path.join(args.outdir, f"fastbench_{args.tag}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
