"""Generate GSM8K-Indic sample outputs from Spark instruct + Sarvam-1 for eyeball review."""
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
N_PROBLEMS = 4
LANGS = ["hi", "bn", "ta", "gu", "en"]

MODELS = [
    ("spark_instruct", "XHToken/Spark-X2.6-1.7B", True),
    ("sarvam1", "sarvamai/sarvam-1", False),
]

out = {}
for mname, repo, chat in MODELS:
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for code in LANGS:
        ds = load_dataset("sarvamai/gsm8k-indic", code, split="test")
        for i in range(N_PROBLEMS):
            r = ds[i]
            key = f"{mname}__{code}__{i}"
            if chat:
                ids = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                              add_generation_prompt=True, return_tensors="pt",
                                              reasoning_effort="low").to(device)
            else:
                ids = tok.encode(r["question"], return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=700, do_sample=False, pad_token_id=pad_id)
            text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=False)
            out[key] = {"lang": code, "question": r["question"],
                        "gold": r["answer"].split("####")[-1].strip(),
                        "output": text}
            print(f"{key} done", flush=True)
    del model
    torch.cuda.empty_cache()

json.dump(out, open(os.path.expanduser("~/samples/gsm8k_indic_samples.json"), "w"),
          indent=1, ensure_ascii=False)
print("SAVED")
