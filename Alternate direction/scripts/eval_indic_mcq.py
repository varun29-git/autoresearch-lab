"""Unified Indic MCQ evaluator (letter-loglikelihood protocol).

One forward pass per item: logits at the final position of the few-shot context,
compare logprobs of the option-letter tokens (" A".." J"). Matches the base-model
MCQ protocol used by MILU / ARC-Challenge-Indic publications.
"""
import json
import os
import random
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "ABCDEFGHIJ"
INDIC10 = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"]
SEED = 42

MODELS = {
    "spark_instruct": "XHToken/Spark-X2.6-1.7B",
    "spark_base": "XHToken/Spark-X2.6-1.7B-Base",
    "sarvam1": "sarvamai/sarvam-1",
}


def fmt_item(q, options, answer_letter=None):
    s = q.strip() + "\n"
    for i, o in enumerate(options):
        s += f"{LETTERS[i]}. {str(o).strip()}\n"
    s += "Answer:"
    if answer_letter is not None:
        s += f" {answer_letter}"
    return s


def fewshot_prefix(val_rows, k):
    parts = []
    for r in val_rows[:k]:
        parts.append(fmt_item(r["q"], r["opts"], LETTERS[r["ans"]]))
    return "\n\n".join(parts) + "\n\n" if parts else ""


def norm_rows(rows):
    return [{"q": q, "opts": o, "ans": a} for q, o, a in rows]


# ---------------- dataset loaders -> {lang: {"test": rows, "val": rows}} ----------
def load_milu():
    out = {}
    name_map = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
                "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
                "ml": "Malayalam", "pa": "Punjabi", "or": "Odia"}
    for code, cfg in name_map.items():
        test = load_dataset("ai4bharat/MILU-1.5", cfg, split="test")
        val = load_dataset("ai4bharat/MILU-1.5", cfg, split="validation")
        mk = lambda r: (r["question"], [r["option1"], r["option2"], r["option3"], r["option4"]],
                        int(r["target"][-1]) - 1)
        out[code] = {"test": norm_rows([mk(r) for r in test]),
                     "val": norm_rows([mk(r) for r in val])}
    return out


def load_mmlu_indic():
    out = {}
    for code in INDIC10 + ["en"]:
        test = load_dataset("sarvamai/mmlu-indic", code, split="test")
        try:
            val = load_dataset("sarvamai/mmlu-indic", code, split="validation")
        except Exception:
            val = []
        mk = lambda r: (r["question"], list(r["choices"]), int(r["answer"]))
        out[code] = {"test": norm_rows([mk(r) for r in test]),
                     "val": norm_rows([mk(r) for r in val])}
    return out


def load_arc_indic():
    out = {}
    for code in INDIC10 + ["en"]:
        test = load_dataset("sarvamai/arc-challenge-indic", code, split="test")
        val = load_dataset("sarvamai/arc-challenge-indic", code, split="validation")
        def mk(r):
            labels = list(r["choices"]["label"])
            texts = list(r["choices"]["text"])
            return (r["question"], texts, labels.index(r["answerKey"]))
        out[code] = {"test": norm_rows([mk(r) for r in test]),
                     "val": norm_rows([mk(r) for r in val])}
    return out


def load_mmlu_pro_indic():
    out = {c: {"test": [], "val": []} for c in INDIC10 + ["en"]}
    test = load_dataset("ai4bharat/indic-mmlu-pro", split="test")
    lang_col = {"en": ("question", "options")}
    for code, name in [("hi", "Hindi"), ("bn", "Bengali"), ("ta", "Tamil"), ("te", "Telugu"),
                       ("mr", "Marathi"), ("gu", "Gujarati"), ("kn", "Kannada"),
                       ("ml", "Malayalam"), ("pa", "Punjabi"), ("or", "Odia")]:
        lang_col[code] = (f"question_{name}_translation", f"options_{name}_translation")
    for r in test:
        for code, (qc, oc) in lang_col.items():
            q, opts = r[qc], r[oc]
            if not q or not opts or any(not str(o).strip() for o in opts):
                continue
            out[code]["test"].append({"q": q, "opts": list(opts), "ans": int(r["answer_index"])})
    return out


def load_bhashabench():
    out = {c: {"test": [], "val": []} for c in INDIC10}
    split_map = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
                 "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
                 "pa": "Punjabi", "or": "Oriya"}
    for cfg in ["BBA", "BBK", "BBL", "BBF"]:
        for code, split in split_map.items():
            ds = load_dataset("bharatgenai/BhashaBench-Multi", cfg, split=split)
            for r in ds:
                q = r["question_translated"]
                opts = [r["option_a_translated"], r["option_b_translated"],
                        r["option_c_translated"], r["option_d_translated"]]
                ca = (r["correct_answer"] or "").strip().upper()
                if not q or ca not in "ABCD" or any(not str(o).strip() for o in opts):
                    continue
                out[code]["test"].append({"q": q, "opts": opts, "ans": "ABCD".index(ca)})
    return out


def load_belebele():
    out = {}
    cfg_map = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml",
               "te": "tel_Telu", "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda",
               "ml": "mal_Mlym", "pa": "pan_Guru", "or": "ory_Orya"}
    for code, cfg in cfg_map.items():
        ds = load_dataset("facebook/belebele", cfg, split="test")
        mk = lambda r: (r["flores_passage"].strip() + "\n" + r["question"],
                        [r["mc_answer1"], r["mc_answer2"], r["mc_answer3"], r["mc_answer4"]],
                        int(r["correct_answer_num"]) - 1)
        out[code] = {"test": norm_rows([mk(r) for r in ds]), "val": []}
    return out


TASKS = {
    "milu": {"loader": load_milu, "n": 1000, "shots": 5},
    "mmlu_indic": {"loader": load_mmlu_indic, "n": 1000, "shots": 5},
    "arc_indic": {"loader": load_arc_indic, "n": None, "shots": 5},
    "mmlu_pro_indic": {"loader": load_mmlu_pro_indic, "n": 1000, "shots": 5},
    "bhashabench": {"loader": load_bhashabench, "n": 500, "shots": 0},
    "belebele_indic": {"loader": load_belebele, "n": None, "shots": 0},
}

# sarvam1 runs only the Nemotron paper's two Indic metrics
MODEL_TASKS = {
    "spark_instruct": list(TASKS),
    "spark_base": list(TASKS),
    "sarvam1": ["milu", "arc_indic"],
}

BATCH = 32
device = "cuda"


def score_lang(model, tok, letter_ids, rows, val_rows, shots, n, batch=BATCH):
    random.Random(SEED).shuffle(rows)
    if n and len(rows) > n:
        rows = rows[:n]
    prefix = fewshot_prefix(val_rows, shots) if shots and val_rows else ""
    correct = 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    i = 0
    cur_batch = batch
    while i < len(rows):
        chunk = rows[i:i + cur_batch]
        try:
            ctxs = [prefix + fmt_item(r["q"], r["opts"]) for r in chunk]
            encs = [tok.encode(c, add_special_tokens=False, return_tensors="pt") for c in ctxs]
            maxlen = max(e.shape[1] for e in encs)
            input_ids = torch.full((len(encs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(encs), maxlen), dtype=torch.long)
            for j, e in enumerate(encs):
                L = e.shape[1]
                input_ids[j, :L] = e[0]
                attn[j, :L] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn).logits
            last = (attn.sum(1) - 1).tolist()
            for j, r in enumerate(chunk):
                row_logits = logits[j, last[j]].float()
                n_opts = len(r["opts"])
                scores = [row_logits[letter_ids[k]].item() for k in range(n_opts)]
                pred = max(range(n_opts), key=lambda k: scores[k])
                correct += int(pred == r["ans"])
            del logits, input_ids, attn
            i += cur_batch
            if cur_batch < batch:
                cur_batch = min(batch, cur_batch * 2)  # recover
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur_batch == 1:
                raise
            cur_batch = max(1, cur_batch // 2)
            print(f"  OOM -> batch {cur_batch}", flush=True)
    return {"n": len(rows), "acc": round(100 * correct / max(len(rows), 1), 2)}


def main():
    only_tasks = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    only_models = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    os.makedirs("results/evals", exist_ok=True)
    data_cache = {}
    for tname, tcfg in TASKS.items():
        if only_tasks and tname not in only_tasks:
            continue
        print(f"\n##### loading {tname}", flush=True)
        data_cache[tname] = tcfg["loader"]()
    for mname, repo in MODELS.items():
        if only_models and mname not in only_models:
            continue
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
        letter_ids = {}
        for L in LETTERS:
            ids = tok.encode(" " + L, add_special_tokens=False)
            letter_ids[LETTERS.index(L)] = ids[-1]
        print(f"{mname} letter ids: {letter_ids}", flush=True)
        for tname in MODEL_TASKS[mname]:
            if tname not in data_cache:
                continue
            tcfg = TASKS[tname]
            out = f"results/evals/{tname}__{mname}.json"
            res = {}
            if os.path.exists(out):  # fill-missing: reuse completed cells
                res = json.load(open(out))
            for code, d in sorted(data_cache[tname].items()):
                if isinstance(res.get(code), dict) and "acc" in res[code]:
                    continue
                test_rows = [dict(r) for r in d["test"]]
                try:
                    res[code] = score_lang(model, tok, letter_ids, test_rows,
                                           d["val"], tcfg["shots"], tcfg["n"])
                except Exception as e:
                    res[code] = {"error": str(e)[:200]}
                print(f"{mname} {tname} {code}: {res[code]}", flush=True)
            out = f"results/evals/{tname}__{mname}.json"
            json.dump(res, open(out, "w"), indent=1)
            print(f"saved {out}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
