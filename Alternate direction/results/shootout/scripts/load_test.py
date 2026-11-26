#!/usr/bin/env python3
"""One-shot model load + tiny chat-template generation sanity check.

Usage: python load_test.py <model_repo_or_path> [--reasoning high]
Prints arch, transformers version, whether the chat template mentions
reasoning_effort, and a short sample generation.
"""
import argparse
import json
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    print(f"transformers_version={transformers.__version__}")
    print(f"torch_version={torch.__version__} cuda={torch.version.cuda}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"load_seconds={time.time()-t0:.1f}")
    print(f"arch={model.config.architectures} model_type={model.config.model_type}")
    if getattr(model.generation_config, "top_k", None) is not None and model.generation_config.top_k <= 0:
        model.generation_config.top_k = None

    template = tok.chat_template or ""
    print(f"template_mentions_reasoning_effort={'reasoning_effort' in template}")
    print(f"template_len={len(template)}")

    msgs = [{"role": "user", "content": "What is 17 + 25? Think briefly, then answer."}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if args.reasoning:
        kwargs["reasoning_effort"] = args.reasoning
    try:
        prompt = tok.apply_chat_template(msgs, **kwargs)
        used_reasoning = args.reasoning
    except TypeError as e:
        print(f"reasoning kwarg rejected: {e}")
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        used_reasoning = None
    print(f"reasoning_effort_used={used_reasoning}")
    print("=== PROMPT START ===")
    print(prompt[:2000])
    print("=== PROMPT END ===")

    inputs = tok(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )
    dt = time.time() - t0
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"gen_seconds={dt:.1f} new_tokens={new_tokens}")
    print("=== GEN START ===")
    print(text[:2000])
    print("=== GEN END ===")
    print(f"peak_vram_gb={torch.cuda.max_memory_allocated()/1e9:.2f}")
    print("LOAD_TEST_OK")


if __name__ == "__main__":
    main()
