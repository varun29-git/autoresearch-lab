"""Keep Spark's English and seven-script rows. Drop every other script.

Ids 0-13 stay. Every other surviving Spark row is packed in original order.
Sarvam pieces whose surface is not already a Spark piece are appended after that.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

from .indic import (
    BYTE_ALPHABET,
    BYTE_OF,
    ENGLISH_TAIL_BASE,
    INDIC_BASE,
    SPACE_MARK,
    _BYTE_FALLBACK,
    encode_tagged,
    has_target_script,
)

FOLDER = Path(__file__).resolve().parents[1]
REPO = FOLDER.parent
SPARK_PATH = FOLDER / "research" / "spark_tokenizer" / "tokenizer.json"
DONOR_PATH = REPO / "tokenizer" / "tokenizer-65k.json"
SPARK_SHA256 = "710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c"
DONOR_SHA256 = "c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29"
PINNED_IDS = 14

# Scripts the device cut removes. The seven target scripts are absent on purpose.
DROP_RANGES = (
    (0x0400, 0x052F),  # Cyrillic
    (0x0530, 0x058F),  # Armenian
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0B00, 0x0B7F),  # Odia
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x10A0, 0x10FF),  # Georgian
    (0x1100, 0x11FF),  # Hangul jamo
    (0x1200, 0x137F),  # Ethiopic
    (0x1780, 0x17FF),  # Khmer
    (0x1800, 0x18AF),  # Mongolian
    (0x3040, 0x30FF),  # Kana
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK
    (0xAC00, 0xD7AF),  # Hangul
    (0xF900, 0xFAFF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
    (0xFF65, 0xFF9F),
    (0x20000, 0x2EBEF),
)

BYTE_DECODER = {char: byte for byte, char in enumerate(BYTE_ALPHABET)}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(path, expected):
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"{path} sha256 {digest} does not match {expected}")
    return digest


def dropped_char(char):
    codepoint = ord(char)
    return any(lo <= codepoint <= hi for lo, hi in DROP_RANGES)


def has_dropped_script(text):
    return any(dropped_char(char) for char in text)


def decode_token(token):
    """Return the UTF-8 surface of a byte-level vocab key, or None."""
    try:
        raw = bytes(BYTE_DECODER[char] for char in token)
    except KeyError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _parents(merges):
    parents = {}
    missing = 0
    for pair in merges:
        if isinstance(pair, str):
            left, right = pair.split(" ")
        else:
            left, right = pair
        produced = left + right
        if produced not in parents:
            parents[produced] = (left, right)
    return parents, missing


def keep_tokens(vocab, merges, added):
    """Token strings that survive. Ancestors of a kept piece are kept."""
    by_id = {index: token for token, index in vocab.items()}
    keep = {by_id[index] for index in range(PINNED_IDS)}
    keep.update(item["content"] for item in added)
    for token in vocab:
        if token in keep:
            continue
        surface = decode_token(token)
        if surface is not None and not has_dropped_script(surface):
            keep.add(token)
        elif surface is None and len(token) == 1 and token in BYTE_OF:
            # Atomic byte symbols stay. Longer invalid sequences stay only
            # when a kept piece needs them as a merge ancestor.
            keep.add(token)
    parents, _missing = _parents(merges)
    stack = list(keep)
    while stack:
        token = stack.pop()
        pair = parents.get(token)
        if pair is None:
            continue
        for part in pair:
            if part not in keep:
                keep.add(part)
                stack.append(part)
    return keep


def _surface_index(vocab, keep):
    """Decoded surface -> original id. Prefer the canonical byte-level key."""
    groups = defaultdict(list)
    for token, old in vocab.items():
        if token not in keep:
            continue
        surface = decode_token(token)
        if surface is None:
            continue
        groups[surface].append(old)
    chosen = {}
    collisions = 0
    for surface, olds in groups.items():
        if len(olds) > 1:
            collisions += 1
        key = "".join(BYTE_ALPHABET[byte] for byte in surface.encode("utf-8"))
        canonical = vocab.get(key)
        if canonical in olds:
            chosen[surface] = canonical
        else:
            chosen[surface] = min(olds)
    return chosen, collisions


def _tokenizer_from_spec(spec):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(spec, handle, ensure_ascii=False)
    handle.close()
    try:
        return Tokenizer.from_file(handle.name)
    finally:
        os.unlink(handle.name)


def _compact_spec(raw, vocab, keep, old_to_new):
    spec = json.loads(json.dumps(raw))
    ordered = sorted((index for token, index in vocab.items() if token in keep))
    # Ids 0-13 are in `ordered` and stay at those ids because nothing before
    # them was dropped. Packing is enumerate over original-id order.
    token_at = {index: token for token, index in vocab.items()}
    spec["model"]["vocab"] = {token_at[old]: old_to_new[old] for old in ordered}
    kept_merges = []
    for pair in raw["model"]["merges"]:
        if isinstance(pair, str):
            left, right = pair.split(" ")
        else:
            left, right = pair
        if left in keep and right in keep and (left + right) in keep:
            kept_merges.append([left, right])
    spec["model"]["merges"] = kept_merges
    spec["added_tokens"] = []
    for item in raw["added_tokens"]:
        if item["id"] not in old_to_new:
            continue
        copied = dict(item)
        copied["id"] = old_to_new[item["id"]]
        spec["added_tokens"].append(copied)
    return spec, len(kept_merges)


class Cut:
    def __init__(self, spark_path=SPARK_PATH, donor_path=DONOR_PATH, keep_ids=None):
        self.spark_path = Path(spark_path)
        self.donor_path = Path(donor_path)
        if not self.spark_path.is_file():
            raise FileNotFoundError(
                f"stock Spark tokenizer is not at {self.spark_path}. "
                "Download tokenizer.json from XHToken/Spark-X2.6-1.7B-Base. "
                "Do not substitute a pruned tokenizer.")
        if not self.donor_path.is_file():
            raise FileNotFoundError(
                f"donor table is not at {self.donor_path}. "
                "The Sarvam piece table is not reconstructed from memory.")
        self.spark_sha256 = _check_sha(self.spark_path, SPARK_SHA256)
        self.donor_sha256 = _check_sha(self.donor_path, DONOR_SHA256)
        with self.spark_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        vocab = raw["model"]["vocab"]
        merges = raw["model"]["merges"]
        self.token_at = {index: token for token, index in vocab.items()}
        if keep_ids is None:
            self.keep = keep_tokens(vocab, merges, raw["added_tokens"])
        else:
            unknown = [index for index in keep_ids if index not in self.token_at]
            if unknown:
                raise ValueError(f"{len(unknown)} keep ids are not Spark vocab ids")
            self.keep = {self.token_at[index] for index in keep_ids}
            for index in range(PINNED_IDS):
                self.keep.add(self.token_at[index])
        self.closure_ok = closure_holds(merges, self.keep)
        ordered = sorted(index for token, index in vocab.items() if token in self.keep)
        self.old_to_new = {old: new for new, old in enumerate(ordered)}
        if any(self.old_to_new[index] != index for index in range(PINNED_IDS)):
            raise RuntimeError("ids 0-13 moved during packing")
        spec, self.merges_kept = _compact_spec(raw, vocab, self.keep, self.old_to_new)
        self.stock = Tokenizer.from_file(str(self.spark_path))
        self.compact = _tokenizer_from_spec(spec)
        self.compact_ids = self.compact.get_vocab()
        self.surface_to_old, self.surface_collisions = _surface_index(vocab, self.keep)
        self.surface_to_new = {
            surface: self.old_to_new[old] for surface, old in self.surface_to_old.items()
        }
        scored, indic_region, indic_order = _load_donor(self.donor_path)
        self.scored = scored
        self.indic_region = indic_region
        self.new_pieces = []
        for piece in indic_order:
            surface = piece.replace(SPACE_MARK, " ")
            if surface not in self.surface_to_old:
                self.new_pieces.append(piece)
        base = len(self.old_to_new)
        self.new_piece_to_id = {
            piece: base + index for index, piece in enumerate(self.new_pieces)
        }
        self.reused_sarvam = len(indic_order) - len(self.new_pieces)
        self.vocab_size = base + len(self.new_pieces)
        self._constituents = {}

    def english_tokens(self, text):
        if text == "":
            return []
        return self.compact.encode(text).tokens

    def encode_tagged(self, text):
        if not has_target_script(text):
            return [("english", token) for token in self.english_tokens(text)]
        return encode_tagged(
            text, self.english_tokens, self.scored, self.indic_region)

    def piece_id(self, piece):
        fallback = _BYTE_FALLBACK.fullmatch(piece)
        if fallback:
            byte = int(piece[3:5], 16)
            return self.compact_ids[BYTE_ALPHABET[byte]]
        surface = piece.replace(SPACE_MARK, " ")
        found = self.surface_to_new.get(surface)
        if found is not None:
            return found
        found = self.new_piece_to_id.get(piece)
        if found is not None:
            return found
        found = self.compact_ids.get(piece)
        if found is None:
            raise KeyError(piece)
        return found

    def encode_ids(self, text):
        return [self.piece_id(piece) for _kind, piece in self.encode_tagged(text)]

    def kept_old_ids(self):
        """Original Spark ids in packed order. Compact id i is this list at i."""
        return sorted(self.old_to_new)

    def decode_ids(self, ids):
        """Unicode string of compact ids, including appended Sarvam pieces."""
        if not hasattr(self, "_id_to_token"):
            self._id_to_token = {index: token for token, index in self.compact_ids.items()}
            self._new_surface = {
                index: piece.replace(SPACE_MARK, " ")
                for piece, index in self.new_piece_to_id.items()
            }
        raw = bytearray()
        parts = []

        def flush():
            if raw:
                parts.append(bytes(raw).decode("utf-8", errors="replace"))
                raw.clear()

        for index in ids:
            surface = self._new_surface.get(index)
            if surface is not None:
                flush()
                parts.append(surface)
                continue
            token = self._id_to_token[index]
            try:
                raw.extend(BYTE_DECODER[char] for char in token)
            except KeyError:
                flush()
                parts.append(token)
        flush()
        return "".join(parts)

    def stock_constituents(self, surface):
        """Original Spark ids of `surface`, for FVT. Cached. Not the compact ids."""
        cached = self._constituents.get(surface)
        if cached is None:
            cached = tuple(self.stock.encode(surface).ids)
            self._constituents[surface] = cached
        return cached

    def counts(self):
        dropped = len(self.token_at) - len(self.keep)
        return {
            "spark_vocab": len(self.token_at),
            "spark_kept": len(self.keep),
            "spark_dropped": dropped,
            "merges_kept": self.merges_kept,
            "sarvam_reused": self.reused_sarvam,
            "sarvam_added": len(self.new_pieces),
            "vocab_size": self.vocab_size,
            "surface_collisions": self.surface_collisions,
            "pinned_ids": PINNED_IDS,
        }


def _load_donor(path):
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    pieces = artifact["pieces"]
    scores = artifact["indic_scores"]
    scored = {}
    indic_region = set()
    indic_order = []
    for index, (piece, score) in enumerate(zip(pieces, scores)):
        if score is None or str(piece).startswith("<reserved_space_"):
            continue
        scored[piece] = float(score)
        if INDIC_BASE <= index < ENGLISH_TAIL_BASE:
            indic_region.add(piece)
            indic_order.append(piece)
    return scored, indic_region, indic_order


def closure_holds(merges, keep):
    parents, _missing = _parents(merges)
    for token in keep:
        pair = parents.get(token)
        if pair is not None and not set(pair) <= keep:
            return False
    return True
