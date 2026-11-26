"""GSM8K-Indic: generative math eval on sarvamai/gsm8k-indic.

Spark instruct uses the chat template (thinking=low, greedy); sarvam-1 (base) uses
8-shot completion prompting. Metric: exact match of final numeric answer.
"""
import json
import os
import re
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGS = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "en"]
N = 500
device = "cuda"


def extract_num(text):
    m = re.findall(r"####\s*([\-0-9\.,]+)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gold_num(ans):
    m = re.search(r"####\s*([\-0-9\.,]+)", ans)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def gen_batch(model, tok, prompts, chat, max_new=512):
    outs = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(0, len(prompts), 8):
        chunk = prompts[i:i + 8]
        if chat:
            ids_list = []
            for p in chunk:
                try:
                    ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                                  add_generation_prompt=True,
                                                  reasoning_effort="low")
                except TypeError:
                    ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                                  add_generation_prompt=True)
                ids_list.append(ids)
        else:
            ids_list = [tok.encode(p, add_special_tokens=False) for p in chunk]
        maxlen = max(len(x) for x in ids_list)
        # left-pad for generation
        input_ids = torch.full((len(ids_list), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids_list), maxlen), dtype=torch.long)
        for j, ids in enumerate(ids_list):
            input_ids[j, maxlen - len(ids):] = torch.tensor(ids)
            attn[j, maxlen - len(ids):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attn,
                                 max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=pad_id)
        for j in range(len(chunk)):
            gen = out[j][maxlen:]
            outs.append(tok.decode(gen, skip_special_tokens=True))
        del out
    return outs


def run(model, tok, chat):
    res = {}
    for code in LANGS:
        ds = load_dataset("sarvamai/gsm8k-indic", code, split="test")
        rows = list(ds)[:N]
        prompts = [r["question"] for r in rows]
        golds = [gold_num(r["answer"]) for r in rows]
        outs = gen_batch(model, tok, prompts, chat)
        correct = 0
        for o, g in zip(outs, golds):
            p = extract_num(o)
            try:
                if p is not None and g is not None and abs(float(p) - float(g)) < 1e-4:
                    correct += 1
            except ValueError:
                pass
        res[code] = {"n": len(rows), "acc": round(100 * correct / len(rows), 2)}
        print(f"gsm8k_indic {code}: {res[code]}", flush=True)
    return res


def main():
    os.makedirs("results/evals", exist_ok=True)
    jobs = [("spark_instruct", "XHToken/Spark-X2.6-1.7B", True),
            ("sarvam1", "sarvamai/sarvam-1", False)]
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    for mname, repo, chat in jobs:
        if only and mname not in only:
            continue
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
        res = run(model, tok, chat)
        out = f"results/evals/gsm8k_indic__{mname}.json"
        json.dump(res, open(out, "w"), indent=1)
        print(f"saved {out}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
