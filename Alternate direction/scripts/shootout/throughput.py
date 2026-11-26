#!/usr/bin/env python3
"""Throughput measurement: prefill tok/s, decode tok/s @batch1 and @batch16, peak VRAM.

Method: timed PyTorch loop (HF transformers generate), bf16, A10G.
- Prefill: forward pass over a ~2048-token prompt (median of 3 reps).
- Decode: generate 512 new tokens; decode rate = new_tokens / (t_generate - t_prefill_est).
- Batch 16: same prompt x16, aggregate tok/s = 16*512 / (t_generate - t_prefill_est_b16).

Usage: python throughput.py --model <repo> --out results/throughput_<tag>.json
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOREM = (
    "The history of computational mathematics stretches from ancient Babylonian tablets "
    "through the mechanical calculators of the seventeenth century to the electronic "
    "computers of the modern era. Each generation of hardware enabled new algorithms, "
    "and each new algorithm demanded new hardware. Numerical linear algebra, root finding, "
    "integration, optimization, and simulation all matured together. In the twenty-first "
    "century, machine learning reframed computation as a differentiable programming "
    "problem, and hardware accelerators reshaped the landscape once again. "
)


def build_prompt(tok, target_tokens=2048):
    text = LOREM
    while len(tok(text)["input_ids"]) < target_tokens:
        text += LOREM
    ids = tok(text)["input_ids"][:target_tokens]
    return tok.decode(ids)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    if getattr(model.generation_config, "top_k", None) is not None and model.generation_config.top_k <= 0:
        model.generation_config.top_k = None

    prompt = build_prompt(tok, args.prompt_tokens)
    enc = tok(prompt, return_tensors="pt").to("cuda")
    n_prompt = enc["input_ids"].shape[1]
    print(f"prompt_tokens={n_prompt}")

    res = {"model": args.model, "prompt_tokens": n_prompt, "new_tokens": args.new_tokens, "method": "timed_torch_loop_hf_generate_bf16"}

    # ---- prefill (batch 1) ----
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(**enc, use_cache=True)
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            model(**enc, use_cache=True)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    t_pf = median(ts)
    res["prefill_b1"] = {"seconds": t_pf, "tok_per_s": n_prompt / t_pf,
                         "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    print(f"prefill_b1: {n_prompt/t_pf:.0f} tok/s")

    # ---- decode batch 1 ----
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        warm = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**enc, max_new_tokens=args.new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t0
    n_new = out.shape[1] - n_prompt
    t_dec = t_gen - t_pf
    res["decode_b1"] = {"new_tokens": int(n_new), "gen_seconds": t_gen,
                        "decode_seconds_est": t_dec, "tok_per_s": n_new / t_dec,
                        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    print(f"decode_b1: {n_new/t_dec:.1f} tok/s")

    # ---- batch N ----
    B = args.batch
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    enc_b = {k: v.repeat(B, 1) for k, v in enc.items()}
    with torch.no_grad():
        # prefill timing for the batch
        for _ in range(1):
            model(**enc_b, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(**enc_b, use_cache=True)
        torch.cuda.synchronize()
        t_pf_b = time.perf_counter() - t0
        warm = model.generate(**enc_b, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_b = model.generate(**enc_b, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t_gen_b = time.perf_counter() - t0
    n_new_b = out_b.shape[1] - n_prompt
    t_dec_b = t_gen_b - t_pf_b
    res["prefill_bN"] = {"batch": B, "seconds": t_pf_b, "tok_per_s": B * n_prompt / t_pf_b}
    res["decode_bN"] = {"batch": B, "new_tokens_per_seq": int(n_new_b), "gen_seconds": t_gen_b,
                        "decode_seconds_est": t_dec_b,
                        "aggregate_tok_per_s": B * n_new_b / t_dec_b,
                        "per_seq_tok_per_s": n_new_b / t_dec_b,
                        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    print(f"decode_b{B}: {B*n_new_b/t_dec_b:.0f} aggregate tok/s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("THROUGHPUT_OK")


if __name__ == "__main__":
    main()
