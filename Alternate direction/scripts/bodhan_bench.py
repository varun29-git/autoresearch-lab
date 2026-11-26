"""Bodhan indic-translate speed + correctness test on L40S.

Serves bodhan-ai/indic-translate via vLLM (FP8), runs a sentence-translation
throughput sweep (EN->Hindi), and prints sample translations for quality check.
"""
import json
import os
import time

from vllm import LLM, SamplingParams

MODEL = "bodhan-ai/indic-translate"
HF_TOKEN = open("/tmp/hf_token").read().strip()

SENTENCES = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast and uses four for muffins. How many does she sell?",
    "The monsoon arrives in Kerala in early June and covers the whole country by mid-July.",
    "A train travels 120 km in 1.5 hours, then 80 km in 45 minutes. What is its average speed?",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent.",
    "She sells the remaining eggs at the farmers' market for two dollars per egg.",
    "Regular exercise improves cardiovascular health and reduces stress.",
    "The Constitution of India came into effect on 26 January 1950.",
    "If a rectangle has length 12 cm and width 5 cm, what is its area?",
    "Water harvesting structures help recharge groundwater in arid regions.",
    "The value of pi is approximately 3.14159.",
    "Mumbai is the financial capital of India and home to the Bombay Stock Exchange.",
    "A shopkeeper buys pens at 8 rupees each and sells them at 12 rupees each. What is the profit percentage?",
    "The Himalayas were formed by the collision of the Indian and Eurasian tectonic plates.",
    "Deforestation leads to loss of biodiversity and increased carbon emissions.",
    "The recipe requires two cups of flour, one teaspoon of baking soda, and a pinch of salt.",
]

TARGET = "Hindi"

# Official prompt contract (scripts/indic_translate_prompt.py):
# "Translate the following text into {tgt}:\n\n{text}" — target only, single user turn,
# no system turn, stop on "<turn|>".
PROMPT_TMPL = "Translate the following text into {lang}:\n\n{text}"

llm = LLM(model=MODEL, quantization="fp8", max_model_len=4096,
          gpu_memory_utilization=0.85, download_dir=os.path.expanduser("~/hf_cache"))
sp = SamplingParams(temperature=0, max_tokens=300, stop=["<turn|>"])


def make_convs(texts):
    return [[{"role": "user", "content": PROMPT_TMPL.format(lang=TARGET, text=t)}]
            for t in texts]

# --- correctness spot-check ---
t0 = time.time()
outs = llm.chat(make_convs(SENTENCES[:4]), sp)
samples = [{"en": s, "hi": o.outputs[0].text.strip()} for s, o in zip(SENTENCES[:4], outs)]
for s in samples:
    print("EN:", s["en"])
    print("HI:", s["hi"][:300])
    print("---")

# --- throughput sweep ---
results = {"model": MODEL, "quantization": "fp8", "gpu": "L40S", "samples": samples, "sweep": []}
for conc in [1, 16, 64, 128]:
    batch = (SENTENCES * (conc // len(SENTENCES) + 1))[:conc]
    # warmup
    if conc == 1:
        llm.chat(make_convs(batch[:1]), sp)
    t0 = time.time()
    outs = llm.chat(make_convs(batch), sp)
    dt = time.time() - t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    rate = out_tokens / dt
    row = {"concurrency": conc, "seconds": round(dt, 2),
           "output_tokens": out_tokens, "output_tok_per_s": round(rate, 1)}
    results["sweep"].append(row)
    print(f"conc={conc}: {out_tokens} tok in {dt:.1f}s -> {rate:.0f} tok/s", flush=True)

peak = max(results["sweep"], key=lambda r: r["output_tok_per_s"])
results["tokens_per_hr_at_peak"] = int(peak["output_tok_per_s"] * 3600)
results["gpu_hours_for_400M"] = round(400e6 / peak["output_tok_per_s"] / 3600, 1)
json.dump(results, open(os.path.expanduser("~/bench/results/bodhan_l40s.json"), "w"),
          indent=1, ensure_ascii=False)
print("SAVED", json.dumps({k: v for k, v in results.items() if k != "samples"}, indent=1))
