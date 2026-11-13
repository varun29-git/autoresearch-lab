"""CPU keep/drop report. Writes no checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
REPO = FOLDER.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from terminus.tokenizer65k import Frozen65kTokenizer  # noqa: E402

from .indic import BYTE_ALPHABET, has_target_script  # noqa: E402
from .model import PINNED_IDS, Cut, decode_token, has_dropped_script  # noqa: E402

REPORT_PATH = FOLDER / "research" / "closure_report.json"
ID_MAP_PATH = FOLDER / "research" / "id_map.json"

ENGLISH = {
    "prose": "The quick brown fox jumps over the lazy dog.",
    "code": "def f(x):\n    return x * 2  # double\n",
    "math": "If $x^2 + y^2 = z^2$ then \\sqrt{x} \\leq \\frac{z}{\\pi} for x, y, z > 0.",
    "newline": "\n",
    "newline_spaces": "\n    ",
    "newline_tab": "\n\t",
    "newline_return": "\n    return",
    "leading_spaces": "  leading",
    "accents": "café naïve",
    "greek": "α ≤ β",
    "specials": "<think> reason </think>",
}

INDIC = {
    "hi": "सूर्योदय के समय गाँव के लोग खेतों में काम करने जाते हैं।",
    "mr": "महाराष्ट्रातील ग्रामीण भागात शेती हा प्रमुख व्यवसाय आहे.",
    "kn": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಭಾರತದ ಮುಖ್ಯ ನಗರಗಳಲ್ಲಿ ಒಂದು.",
    "te": "తెలంగాణ రాష్ట్ర రాజధాని హైదరాబాద్ చారిత్రక నగరం.",
    "ta": "தமிழ்நாட்டின் தலைநகரம் சென்னை, இது ஒரு பெரிய நகரம்.",
    "gu": "ગુજરાતની સંસ્કૃતિ અને પરંપરા ખૂબ જ સમૃદ્ધ છે.",
    "bn": "পশ্চিমবঙ্গের রাজধানী কলকাতা একটি বড় শহর।",
}

DONOR_GATES = {
    "नमस्ते": ["Ġनम", "स्ते"],
    "हिन्दी": ["Ġह", "िन्", "दी"],
    "ಕನ್ನಡ": ["Ġಕ", "ನ್ನ", "ಡ"],
}


def _words(text):
    parts = text.split()
    return len(parts) if parts else 1


def _first_surface(cut, predicate, *, kept):
    for old in range(len(cut.token_at)):
        surface = decode_token(cut.token_at[old])
        is_kept = old in cut.old_to_new
        if surface and predicate(surface) and is_kept == kept:
            return {"old_id": old, "surface": surface, "kept": is_kept}
    return None


def evaluate(cut, oracle):
    identity = {}
    for name, text in ENGLISH.items():
        if has_target_script(text):
            raise ValueError(f"English sample {name} contains a target script")
        stock = cut.stock.encode(text)
        compact = cut.compact.encode(text)
        mapped = [cut.old_to_new[index] for index in stock.ids]
        identity[name] = {
            "tokens_identical": stock.tokens == compact.tokens,
            "ids_match_map": compact.ids == mapped,
            "tokens": len(stock.tokens),
        }
    donor_gates = {}
    for text, expected in DONOR_GATES.items():
        pieces = [piece for _kind, piece in cut.encode_tagged(text)]
        donor = [oracle.pieces[index] for index in oracle.encode(text)]
        donor_gates[text] = {
            "pieces": pieces,
            "matches_donor": pieces == expected and pieces == donor,
        }
    mixed_text = "main जा रहा hoon"
    mixed = cut.encode_tagged(mixed_text)
    mixed_indic = [piece for kind, piece in mixed if kind == "indic"]
    donor_mixed = [oracle.pieces[index] for index in oracle.encode(mixed_text)]
    donor_indic = [piece for piece in donor_mixed if piece in cut.indic_region]
    fertility = {}
    for name, text in INDIC.items():
        stock_n = len(cut.stock.encode(text).tokens)
        ours_n = len(cut.encode_ids(text))
        words = _words(text)
        fertility[name] = {
            "stock_tokens": stock_n,
            "ours_tokens": ours_n,
            "stock_per_word": round(stock_n / words, 4),
            "ours_per_word": round(ours_n / words, 4),
            "lower": ours_n < stock_n,
        }
    pure = {}
    for name, text in INDIC.items():
        ours = [piece for _kind, piece in cut.encode_tagged(text)]
        donor = [oracle.pieces[index] for index in oracle.encode(text)]
        pure[name] = ours == donor
    return {
        "spark_sha256": cut.spark_sha256,
        "donor_sha256": cut.donor_sha256,
        "counts": cut.counts(),
        "closure_ok": cut.closure_ok,
        "byte_symbols_present": sum(char in cut.token_at.values() for char in BYTE_ALPHABET),
        "byte_alphabet_kept": all(
            char not in cut.token_at.values() or char in cut.keep for char in BYTE_ALPHABET),
        "pinned_ids_fixed": all(cut.old_to_new[index] == index for index in range(PINNED_IDS)),
        "identity": identity,
        "donor_gates": {text: row["matches_donor"] for text, row in donor_gates.items()},
        "donor_gate_pieces": {text: row["pieces"] for text, row in donor_gates.items()},
        "mixed": {
            "indic_pieces": mixed_indic,
            "matches_donor_indic": mixed_indic == donor_indic,
            "english_main": [piece for kind, piece in mixed if kind == "english"][:1],
            "english_hoon": [piece for kind, piece in mixed if kind == "english"][-2:],
        },
        "pure_indic_matches_donor": pure,
        "fertility": fertility,
        "kept_with_dropped_script": sum(
            1 for old, token in cut.token_at.items()
            if old in cut.old_to_new
            and (surface := decode_token(token)) is not None
            and has_dropped_script(surface)),
        "examples": {
            "devanagari": _first_surface(
                cut, lambda text: "\u0900" <= text[0] <= "\u097f", kept=True),
            "cjk": _first_surface(
                cut, lambda text: "\u4e00" <= text[0] <= "\u9fff", kept=False),
        },
    }


def write_id_map(cut, path):
    spark_rows = [
        {"new": cut.old_to_new[old], "old": old}
        for old in sorted(cut.old_to_new)
    ]
    added = [
        {"new": cut.new_piece_to_id[piece], "piece": piece}
        for piece in cut.new_pieces
    ]
    path.write_text(json.dumps({"spark": spark_rows, "added": added}, ensure_ascii=False),
                    encoding="utf-8")


def passed(report):
    if not report["closure_ok"] or not report["byte_alphabet_kept"] or not report["pinned_ids_fixed"]:
        return False
    if not all(row["tokens_identical"] and row["ids_match_map"] for row in report["identity"].values()):
        return False
    if not all(report["donor_gates"].values()):
        return False
    if not report["mixed"]["matches_donor_indic"]:
        return False
    if not all(report["pure_indic_matches_donor"].values()):
        return False
    if not all(row["lower"] for row in report["fertility"].values()):
        return False
    examples = report["examples"]
    if examples["devanagari"] is None or not examples["devanagari"]["kept"]:
        return False
    if examples["cjk"] is None or examples["cjk"]["kept"]:
        return False
    return True


def main():
    cut = Cut()
    oracle = Frozen65kTokenizer(cut.donor_path)
    report = evaluate(cut, oracle)
    report["pass"] = passed(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_id_map(cut, ID_MAP_PATH)
    counts = report["counts"]
    print(
        f"kept {counts['spark_kept']} dropped {counts['spark_dropped']} "
        f"reused {counts['sarvam_reused']} added {counts['sarvam_added']} "
        f"vocab {counts['vocab_size']} pass {report['pass']}"
    )
    print(f"report {REPORT_PATH}")
    print(f"id map {ID_MAP_PATH}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
