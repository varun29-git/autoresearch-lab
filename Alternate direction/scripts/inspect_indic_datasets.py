"""Inspect schemas of Indic eval datasets before writing the runner."""
import json

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

TARGETS = [
    ("ai4bharat/MILU-1.5", "Hindi", "test"),
    ("sarvamai/mmlu-indic", "hi", "test"),
    ("sarvamai/arc-challenge-indic", "hi", "test"),
    ("sarvamai/gsm8k-indic", "hi", "test"),
    ("ai4bharat/indic-mmlu-pro", None, "test"),
    ("bharatgenai/BhashaBench-Multi", "BBA", "Hindi"),
    ("facebook/belebele", "hin_Deva", "test"),
]

for repo, cfg, split in TARGETS:
    print("=" * 70)
    print(f"{repo} cfg={cfg} split={split}")
    try:
        cfgs = get_dataset_config_names(repo)
        print("configs:", cfgs[:20], f"({len(cfgs)} total)")
        ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
        print("features:", ds.features)
        print("n_rows:", len(ds))
        ex = ds[0]
        print("example:", json.dumps(ex, ensure_ascii=False, default=str)[:900])
    except Exception as e:
        print("ERROR:", str(e)[:400])
