"""Probe 1: repo facts not on the model card — Base model existence, tokenizer identity, license."""
import hashlib
import json
import os

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
report = {}

for repo in ["XHToken/Spark-X2.6-1.7B", "XHToken/Spark-X2.6-1.7B-Base"]:
    entry = {}
    try:
        info = api.model_info(repo, files_metadata=True)
        entry["exists"] = True
        entry["created"] = str(info.created_at)
        entry["last_modified"] = str(info.last_modified)
        entry["downloads"] = info.downloads
        entry["license_tag"] = (info.card_data or {}).get("license")
        entry["files"] = {s.rfilename: s.size for s in info.siblings}
    except Exception as e:
        entry["exists"] = False
        entry["error"] = str(e)[:300]
    report[repo] = entry

# If Base exists, compare config + tokenizer with instruct
base = "XHToken/Spark-X2.6-1.7B-Base"
inst = "XHToken/Spark-X2.6-1.7B"
if report[base].get("exists"):
    comp = {}
    for fname in ["config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"]:
        try:
            pb = hf_hub_download(base, fname, local_dir="models/base_meta")
            pi = hf_hub_download(inst, fname, local_dir="models/spark")
            hb = hashlib.md5(open(pb, "rb").read()).hexdigest()
            hi = hashlib.md5(open(pi, "rb").read()).hexdigest()
            comp[fname] = {"identical": hb == hi, "base_md5": hb, "inst_md5": hi}
        except Exception as e:
            comp[fname] = {"error": str(e)[:200]}
    # config diff detail
    try:
        cb = json.load(open("models/base_meta/config.json"))
        ci = json.load(open("models/spark/config.json"))
        comp["config_diff"] = {k: {"base": cb.get(k), "inst": ci.get(k)}
                               for k in set(cb) | set(ci) if cb.get(k) != ci.get(k)}
        comp["base_vocab_size"] = cb.get("vocab_size")
        comp["base_tie"] = cb.get("tie_word_embeddings")
    except Exception as e:
        comp["config_diff"] = str(e)[:200]
    report["comparison"] = comp

# License text (head) + look for extra restrictions
try:
    lic = hf_hub_download(inst, "LICENSE", local_dir="models/spark")
    text = open(lic, encoding="utf-8").read()
    report["license"] = {
        "length_chars": len(text),
        "is_apache2_boilerplate": "Apache License" in text and "Version 2.0" in text,
        "head": text[:600],
        "mentions_restrictions": any(
            w in text.lower() for w in ["may not use", "acceptable use", "prohibited"]
        ),
    }
except Exception as e:
    report["license"] = {"error": str(e)[:200]}

os.makedirs("results", exist_ok=True)
with open("results/probe1_repo_facts.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str)[:6000])
