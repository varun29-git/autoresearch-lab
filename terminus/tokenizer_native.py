"""CPU native execution of the frozen production tokenizer's NFC contract.

The C++ extension executes routing, BPE, decoding and round-trip verification.
Unicode normalization uses compiled CPython/unicodedata2; the scissors engine
uses C++ with exact property tables from pinned regex. Process-level parallelism avoids GIL contention; the
batch API returns little-endian uint16 buffers without a Python integer per ID.
No training framework is imported by this module.
"""
from __future__ import annotations

import hashlib
import json
import platform
from functools import lru_cache
from pathlib import Path

import regex

from . import unicode_contract as unicodedata
from .tokenizer65k import Frozen65kTokenizer, TOKENIZER_SHA256, _translate_scissors


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _latin_codepoints():
    # Match the reference's NAME + general-category rule, not a different
    # library's Script=Latin property. Pin the Unicode database in each receipt.
    categories = {"Lu", "Ll", "Lt", "Lm", "Lo"}
    return tuple(c for c in range(0x110000)
                 if unicodedata.category(chr(c)) in categories
                 and unicodedata.name(chr(c), "").startswith("LATIN "))


@lru_cache(maxsize=1)
def _regex_properties():
    unicodedata.require_regex_version(regex)
    text = "".join(map(chr, range(0x110000)))
    table = bytearray(0x110000)
    for pattern, bit in ((r"\p{L}+", 1), (r"\p{M}+", 2), (r"\p{N}+", 4), (r"\s+", 8)):
        for match in regex.finditer(pattern, text):
            table[match.start():match.end()] = bytes([bit]) * (match.end() - match.start())
    return bytes(table)


class Native65kTokenizer:
    def __init__(self, path, *, expected_sha256=TOKENIZER_SHA256,
                 english_cache_bytes=64 << 20, indic_cache_bytes=16 << 20):
        import _terminus_tokenizer_native as native

        if native.ABI_VERSION != 3:
            raise ValueError("unsupported native tokenizer ABI")
        unicodedata.require_regex_version(regex)
        if any(type(n) is not int or n < 0 for n in (english_cache_bytes, indic_cache_bytes)):
            raise ValueError("cache budgets must be nonnegative integers")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("an explicit reviewed tokenizer SHA-256 is required")
        self.path = str(Path(path).resolve())
        self.sha256 = _sha(self.path)
        if self.sha256 != expected_sha256:
            raise ValueError("tokenizer artifact does not match the reviewed SHA-256")
        artifact = json.loads(Path(self.path).read_text(encoding="utf-8"))
        pattern = (r"(?:\r\n|\n|\r)[ \t]*|\\[A-Za-z]+|[0-9]|"
                   r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(?:['\x{2019}][\p{L}\p{M}]+)*|"
                   r"[^\s\p{L}\p{M}\p{N}]+|[^\S\r\n]+|[\s\S]")
        if artifact["english_scissors"] != pattern:
            raise ValueError("native scissors require the exact reviewed pattern")
        # Reuse the reference's structural validator without its unrelated
        # training-integrity imports. Do not copy or weaken the format checks.
        validator = object.__new__(Frozen65kTokenizer)
        validator._validate_artifact(artifact)
        self.pieces = artifact["pieces"]
        self.scores = artifact["indic_scores"]
        self.unused_token_ids = tuple(i for i, p in enumerate(self.pieces)
                                     if p.startswith("<reserved"))
        self.noncontent_token_ids = tuple(range(32)) + tuple(
            i for i in self.unused_token_ids if i >= 32)
        self._core = native.NativeEncoder(
            artifact, regex.compile(_translate_scissors(artifact["english_scissors"])),
            _latin_codepoints(), _regex_properties(), english_cache_bytes, indic_cache_bytes)
        self._binary_path = native.__file__

    @staticmethod
    def _normalized(text):
        if not isinstance(text, str):
            raise ValueError("encode expects Unicode text")
        normalized = unicodedata.normalize("NFC", text)
        normalized.encode("utf-8", errors="strict")
        return normalized

    def encode(self, text, out_type=int):
        if out_type is not int:
            raise ValueError("tokenization returns integer IDs")
        return self._core.encode(self._normalized(text), True)

    def encode_bytes(self, text):
        """Little-endian uint16 content IDs, with an exact NFC round trip."""
        return self._core.encode_bytes(self._normalized(text), True)

    def encode_batch_bytes(self, texts):
        """One verified uint16 byte buffer per original document, in order."""
        return self._core.encode_batch_bytes([self._normalized(t) for t in texts], True)

    def decode(self, tokens, *, render_specials=False):
        values = list(tokens)
        if any(type(t) is not int or not 0 <= t < 65536 for t in values):
            raise ValueError("token ID is outside the vocabulary")
        return bytes(self._core.decode_bytes(values, render_specials)).decode("utf-8", errors="strict")

    def identity(self):
        root = Path(__file__).resolve().parent
        record = {
            "schema": 1, "encoder": "Native65kTokenizer", "normalization": "NFC",
            "native_abi": 3,
            "special_policy": "literal_text_no_automatic_special_ids",
            "input_policy": "accept_all_Unicode_scalars_reject_unpaired_surrogates",
            "tokenizer_sha256": self.sha256,
            "python_version": platform.python_version(),
            "unicode_database": unicodedata.unidata_version,
            "regex_version": regex.__version__,
            "implementation_sha256": _sha(__file__),
            "reference_implementation_sha256": _sha(root / "tokenizer65k.py"),
            "unicode_contract_sha256": _sha(root / "unicode_contract.py"),
            "native_binary_sha256": _sha(self._binary_path),
            "latin_classification_sha256": hashlib.sha256(
                json.dumps(_latin_codepoints(), separators=(",", ":")).encode()).hexdigest(),
            "regex_classification_sha256": hashlib.sha256(_regex_properties()).hexdigest(),
            "unused_token_ids": list(self.unused_token_ids),
            "roundtrip_checked_per_document": True,
        }
        record["identity_sha256"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return record

    def statistics(self):
        return dict(self._core.statistics())
