"""Probe 3: vocabulary-surgery dry run.

Rebuilds the Spark tokenizer without (A) multi-char CJK tokens or (B) all CJK-ideograph
tokens, cascading merge deletions, then checks tokenization equivalence on English/Indic/code
corpora and measures Chinese degradation.
"""
import json
import os
import sys

from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BYTE_DECODER, classify, decode_token, fetch_flores,
                    load_vocab, parse_merges)

CODE_SAMPLE = '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
''' * 20

tj, vocab, merges_raw = load_vocab("models/spark/tokenizer.json")
merges = parse_merges(merges_raw)
print(f"vocab={len(vocab)} merges={len(merges)}")

VARIANTS = {
    "A_multichar_cjk": lambda s: sum(1 for ch in s if 0x4E00 <= ord(ch) <= 0x9FFF) >= 1
                                 and len(s) >= 2 and classify(s) <= {"cjk", "cjk_ext_a"},
    "B_all_cjk": lambda s: classify(s) <= {"cjk", "cjk_ext_a"} and len(s) >= 1,
}


def rebuild(prune_pred):
    pruned = set()
    for tok in vocab:
        if len(tok) == 1:
            continue  # byte alphabet, never pruned
        s, valid = decode_token(tok)
        if valid and prune_pred(s):
            pruned.add(tok)
    kept_merges = []
    removed = set(pruned)
    produced = set()
    n_merges_removed = 0
    for a, b in merges:
        out = a + b
        if a in removed or b in removed or out in removed:
            removed.add(out)
            n_merges_removed += 1
            continue
        kept_merges.append([a, b])
        produced.add(out)
    # byte alphabet + added/special tokens (chars outside the byte map) are always kept
    always_keep = {t for t in vocab
                   if len(t) == 1 or any(c not in BYTE_DECODER for c in t)}
    kept_tokens = {t for t in vocab
                   if t not in removed and (t in always_keep or t in produced)}
    # reindex in original id order
    ordered = sorted(kept_tokens, key=lambda t: vocab[t])
    new_vocab = {t: i for i, t in enumerate(ordered)}
    new_tj = dict(tj)
    new_tj["model"] = dict(tj["model"])
    new_tj["model"]["vocab"] = new_vocab
    new_tj["model"]["merges"] = kept_merges
    # specials already live inside model.vocab; leave added_tokens untouched
    return new_tj, {
        "tokens_pruned_by_rule": len(pruned),
        "tokens_removed_total": len(vocab) - len(new_vocab),
        "merges_removed": n_merges_removed,
        "merges_kept": len(kept_merges),
        "new_vocab_size": len(new_vocab),
    }


corpus = fetch_flores("devtest", max_lines=300)
corpus["code_python"] = [CODE_SAMPLE]

orig = Tokenizer.from_str(json.dumps(tj))


def tok_seq(tok, texts):
    # compare token STRINGS, not ids — reindexing makes id comparison meaningless
    return [tok.encode(t, add_special_tokens=False).tokens for t in texts]


report = {"variants": {}}
for name, pred in VARIANTS.items():
    new_tj, stats = rebuild(pred)
    try:
        new = Tokenizer.from_str(json.dumps(new_tj))
    except Exception as e:
        stats["error"] = str(e)[:300]
        report["variants"][name] = stats
        print(f"{name}: BUILD FAILED: {e}")
        continue
    lang_results = {}
    for lang, lines in corpus.items():
        old_ids = tok_seq(orig, lines)
        new_ids = tok_seq(new, lines)
        identical = sum(1 for a, b in zip(old_ids, new_ids) if a == b)
        old_len = sum(len(x) for x in old_ids)
        new_len = sum(len(x) for x in new_ids)
        lang_results[lang] = {
            "pct_identical": round(100 * identical / len(lines), 2),
            "old_tokens": old_len,
            "new_tokens": new_len,
            "token_growth_pct": round(100 * (new_len - old_len) / max(old_len, 1), 2),
        }
    stats["equivalence"] = lang_results
    report["variants"][name] = stats
    print(f"{name}: pruned={stats['tokens_pruned_by_rule']} removed_total={stats['tokens_removed_total']} "
          f"merges_removed={stats['merges_removed']}")

os.makedirs("results", exist_ok=True)
with open("results/probe3_prune_dryrun.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n=== pct_identical (old vs new tokenizer) ===")
for name in VARIANTS:
    eq = report["variants"][name].get("equivalence", {})
    print(f"\n[{name}]")
    for lang, r in eq.items():
        print(f"  {lang:12s} identical={r['pct_identical']:6.2f}%  growth={r['token_growth_pct']:+7.2f}%")
