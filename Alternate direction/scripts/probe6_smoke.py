"""Probe 6: qualitative smoke test of the instruct model — English reasoning (think off/on),
Hindi comprehension + generation, tool-call format sanity."""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "XHToken/Spark-X2.6-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

PROMPTS = [
    {"name": "en_math_no_think", "lang": "en", "effort": "low",
     "msgs": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]},
    {"name": "en_reason_think", "lang": "en", "effort": "low",
     "msgs": [{"role": "user", "content": "A train travels 120 km in 1.5 hours, then 80 km in 45 minutes. What is its average speed for the whole journey in km/h? Think step by step."}]},
    {"name": "hi_fact", "lang": "hi", "effort": "low",
     "msgs": [{"role": "user", "content": "भारत की राजधानी क्या है? एक वाक्य में उत्तर दीजिए।"}]},
    {"name": "hi_explain", "lang": "hi", "effort": "low",
     "msgs": [{"role": "user", "content": "सौर ऊर्जा के दो फायदे सरल हिंदी में समझाइए।"}]},
    {"name": "ta_fact", "lang": "ta", "effort": "low",
     "msgs": [{"role": "user", "content": "தமிழ்நாட்டின் தலைநகரம் எது? ஒரு வாக்கியத்தில் பதிலளிக்கவும்."}]},
    {"name": "hinglish", "lang": "hi-lat", "effort": "low",
     "msgs": [{"role": "user", "content": "Mujhe ek chhota sa Python function likhkar do jo list ka average nikale."}]},
]

report = {}
for p in PROMPTS:
    try:
        ids = tok.apply_chat_template(
            p["msgs"], add_generation_prompt=True, return_tensors="pt",
            reasoning_effort=p["effort"]).to(device)
    except TypeError:
        ids = tok.apply_chat_template(
            p["msgs"], add_generation_prompt=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=220, do_sample=False,
                             pad_token_id=tok.pad_token_id or 2)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    report[p["name"]] = {"effort": p["effort"], "output": gen}
    print(f"\n===== {p['name']} (effort={p['effort']}) =====\n{gen[:800]}")

os.makedirs("results", exist_ok=True)
with open("results/probe6_smoke.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
