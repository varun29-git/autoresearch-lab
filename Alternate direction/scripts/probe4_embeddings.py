"""Probe 4: embedding-table forensics on Spark-X2.6-1.7B.

Row-norm distribution by script category; detection of never-trained / near-init rows
(glitch-token candidates); state of Indic vs CJK vs ASCII rows; the 86 unused tail slots.
"""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (INDIC_SCRIPTS, BYTE_DECODER, classify, decode_token,
                    load_vocab)

INIT_RANGE = 0.0221  # config.initializer_range; expected row norm if Normal(0, r): r*sqrt(2048) ~ 1.0

tj, vocab, _ = load_vocab("models/spark/tokenizer.json")
added = tj.get("added_tokens", [])
n_base = len(vocab)

# locate embedding tensor across shards
emb = None
for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    p = os.path.join("models/spark", shard)
    if not os.path.exists(p):
        continue
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if "embedding" in k:
                emb = f.get_tensor(k)
                emb_key = k
                break
    if emb is not None:
        break
assert emb is not None, "embedding tensor not found"
print(f"embedding key={emb_key} shape={tuple(emb.shape)} dtype={emb.dtype}")

W = emb.float()
norms = W.norm(dim=1)

# classify every base-vocab row
id2tok = {i: t for t, i in vocab.items()}
cats = {}
for i in range(n_base):
    s, valid = decode_token(id2tok[i])
    if not valid:
        c = "byte_fragment"
    else:
        cs = classify(s)
        if cs <= {"ascii"}:
            c = "ascii"
        elif cs <= {"cjk", "cjk_ext_a"}:
            c = "cjk_single" if len(s) == 1 else "cjk_multi"
        elif cs & INDIC_SCRIPTS and not (cs - INDIC_SCRIPTS - {"ascii"}):
            c = "indic_" + sorted(cs & INDIC_SCRIPTS)[0]
        elif cs <= {"kana"}:
            c = "kana"
        elif cs <= {"hangul"}:
            c = "hangul"
        elif cs <= {"cyrillic"}:
            c = "cyrillic"
        elif cs <= {"arabic"}:
            c = "arabic"
        else:
            c = "other_mixed"
    cats.setdefault(c, []).append(i)

# added/special rows
special_ids = [a["id"] for a in added]
unused_ids = [a["id"] for a in added if "unused" in a["content"]]
cats["special_nontool"] = [i for i in special_ids if i not in unused_ids]
cats["unused_tail"] = unused_ids


def summarize(ids):
    n = norms[torch.tensor(ids)]
    q = torch.quantile(n, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "count": len(ids),
        "norm_mean": round(n.mean().item(), 3),
        "norm_p05": round(q[0].item(), 3),
        "norm_median": round(q[2].item(), 3),
        "norm_p95": round(q[4].item(), 3),
        "pct_below_1.05": round(100 * (n < 1.05).float().mean().item(), 2),
    }


report = {"embedding_key": emb_key, "shape": list(emb.shape), "init_range": INIT_RANGE,
          "expected_init_row_norm": round(INIT_RANGE * (2048 ** 0.5), 3),
          "by_category": {c: summarize(ids) for c, ids in sorted(cats.items())}}

# near-init / dead rows: bottom 0.5% of norms overall
thr = torch.quantile(norms, 0.005).item()
dead = (norms < thr).nonzero().flatten().tolist()
report["dead_row_threshold_norm"] = round(thr, 3)
report["dead_rows_count"] = len(dead)
dead_examples = []
for i in dead[:60]:
    if i < n_base:
        s, valid = decode_token(id2tok[i])
        dead_examples.append({"id": i, "token": s if valid else "<byte-frag>", "norm": round(norms[i].item(), 3)})
    else:
        content = next((a["content"] for a in added if a["id"] == i), "?")
        dead_examples.append({"id": i, "token": content, "norm": round(norms[i].item(), 3)})
report["dead_row_examples"] = dead_examples

# do Indic rows look trained? fraction of Indic rows below the ASCII p05
ascii_p05 = report["by_category"]["ascii"]["norm_p05"]
report["indic_rows_below_ascii_p05_pct"] = {
    c: round(100 * (norms[torch.tensor(ids)] < ascii_p05).float().mean().item(), 2)
    for c, ids in sorted(cats.items()) if c.startswith("indic_")
}
report["ascii_p05_ref"] = ascii_p05

# unused tail rows: are they exactly at init scale?
if unused_ids:
    un = norms[torch.tensor(unused_ids)]
    report["unused_tail_norms"] = {"mean": round(un.mean().item(), 4),
                                   "min": round(un.min().item(), 4),
                                   "max": round(un.max().item(), 4)}

os.makedirs("results", exist_ok=True)
with open("results/probe4_embeddings.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps({k: v for k, v in report.items() if k != "dead_row_examples"}, indent=2)[:4000])
print(f"\ndead rows (< p0.5 norm {thr:.3f}): {len(dead)}; examples: {[e['token'] for e in dead_examples[:25]]}")
