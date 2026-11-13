"""Shared helpers for Spark-X2.6-1.7B probes: FLORES fetch, token decoding, Unicode classification."""
import json
import os
import urllib.request

LANGS = [
    "eng_Latn", "zho_Hans",
    "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "mar_Deva", "guj_Gujr",
    "kan_Knda", "mal_Mlym", "pan_Guru", "ory_Orya", "asm_Beng", "nep_Deva",
    "urd_Arab", "san_Deva",
]

INDIC_LANGS = [l for l in LANGS if l not in ("eng_Latn", "zho_Hans")]

WIKI_CODES = {
    "eng_Latn": "en", "zho_Hans": "zh", "hin_Deva": "hi", "ben_Beng": "bn",
    "tam_Taml": "ta", "tel_Telu": "te", "mar_Deva": "mr", "guj_Gujr": "gu",
    "kan_Knda": "kn", "mal_Mlym": "ml", "pan_Guru": "pa", "ory_Orya": "or",
    "asm_Beng": "as", "nep_Deva": "ne", "urd_Arab": "ur", "san_Deva": "sa",
}


def fetch_corpus(langs=LANGS, n_articles=250, max_chars=1200, cache="data/wiki_corpus.json"):
    """Wikipedia (20231101 dump) article openings per language, cached to JSON."""
    if os.path.exists(cache):
        return json.load(open(cache))
    from datasets import load_dataset
    out = {}
    for lang in langs:
        cfg = f"20231101.{WIKI_CODES[lang]}"
        try:
            ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            texts = []
            for ex in ds:
                if len(texts) >= n_articles:
                    break
                t = " ".join(ex["text"].split())
                if len(t) > 50:
                    texts.append(t[:max_chars])
            out[lang] = texts
            print(f"corpus {lang}: {len(texts)} articles")
        except Exception as e:
            print(f"FAILED corpus {lang}: {str(e)[:200]}")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"), ensure_ascii=False)
    return out


def fetch_flores(split="devtest", langs=LANGS, cache="data/wiki_corpus.json", max_lines=None):
    """Back-compat shim: returns the Wikipedia corpus (FLORES CDN unreachable)."""
    return fetch_corpus(langs=langs, n_articles=max_lines or 250, cache=cache)


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_DECODER = {v: k for k, v in _bytes_to_unicode().items()}


def decode_token(tok):
    """GPT-2 byte-map token -> (decoded string, is_valid_utf8).

    Returns ("", False) for added/special tokens whose chars are outside the byte map
    (e.g. fullwidth bars in '<｜start▁of▁sentence｜>').
    """
    try:
        raw = bytes([BYTE_DECODER[c] for c in tok])
    except KeyError:
        return "", False
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), False


SCRIPT_RANGES = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("sinhala", 0x0D80, 0x0DFF),
    ("cjk", 0x4E00, 0x9FFF),
    ("cjk_ext_a", 0x3400, 0x4DBF),
    ("kana", 0x3040, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
    ("cyrillic", 0x0400, 0x04FF),
    ("arabic", 0x0600, 0x06FF),
]

INDIC_SCRIPTS = {"devanagari", "bengali", "gurmukhi", "gujarati", "oriya",
                 "tamil", "telugu", "kannada", "malayalam"}


def classify(s):
    cats = set()
    for ch in s:
        o = ord(ch)
        if o < 128:
            cats.add("ascii")
            continue
        for name, lo, hi in SCRIPT_RANGES:
            if lo <= o <= hi:
                cats.add(name)
                break
        else:
            cats.add("other")
    return cats


def load_vocab(path="models/spark/tokenizer.json"):
    tj = json.load(open(path))
    return tj, tj["model"]["vocab"], tj["model"]["merges"]


def parse_merges(merges):
    out = []
    for m in merges:
        if isinstance(m, list):
            out.append((m[0], m[1]))
        else:
            a, b = m.split(" ")
            out.append((a, b))
    return out
