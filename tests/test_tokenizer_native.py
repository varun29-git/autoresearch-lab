"""Native/reference parity gates; run with the compiled extension installed."""
import importlib.util
import random
import struct
import unittest
from pathlib import Path
import regex

from terminus.tokenizer65k import Frozen65kTokenizer, _translate_scissors
from terminus import unicode_contract as unicodedata

ROOT = Path(__file__).resolve().parents[1]
HAS_NATIVE = importlib.util.find_spec("_terminus_tokenizer_native") is not None


@unittest.skipUnless(HAS_NATIVE, "build native/tokenizer65k before native qualification")
class NativeTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminus.tokenizer_native import Native65kTokenizer
        cls.ref = Frozen65kTokenizer(ROOT / "tokenizer/tokenizer-65k.json")
        cls.native = Native65kTokenizer(ROOT / "tokenizer/tokenizer-65k.json",
                                       english_cache_bytes=8192, indic_cache_bytes=8192)

    def check(self, text):
        normalized = unicodedata.normalize("NFC", text)
        expected = self.ref.encode(normalized)
        self.assertEqual(self.native.encode(text), expected, repr(text))
        self.assertEqual(self.native.encode_bytes(text), struct.pack("<" + "H" * len(expected), *expected))
        self.assertEqual(self.native.decode(expected), normalized)
        self.assertFalse(set(expected) & set(self.native.noncontent_token_ids))

    def test_boundary_regressions(self):
        for text in ["", " ", "  ", "\r\n\t", "café", "cafÃ©", "e\u0301", "Ġ",
                     "नमस्ते Ġ दुनिया", "क 2 क", "क a2 क", "a2क", "क्‍क", "क्‌क",
                     "main जा रहा hoon", "ঐতিহাসিক", "ಠಾಕೂರ್", "\x01",
                     "def f():\n    return 'café'\n", "\\frac{x}{2} + \\pi",
                     "<s> </s> <pad> <eod> <reserved_0005> <reserved_space_545>",
                     "😀 中文 العربية", "2. తెలుగు", " जा", "जा "]:
            with self.subTest(text=text): self.check(text)

    def test_seeded_mixed_scripts_and_unicode(self):
        rng = random.Random(20261107)
        atoms = ["नमस्ते", "বাংলা", "ಕನ್ನಡ", "தமிழ்", "తెలుగు", "ગુજરાતી", "café",
                 "Ġ", "<eod>", "क्\u200dष", "\u200c", "e\u0301", "😀", "\u200b", "\u00a0",
                 "\ufeff", "\u0301", "०", "১", "(", ")", "’", "\\frac", " ", "  ", "\n", "\t"]
        for _ in range(2000): self.check("".join(rng.choices(atoms, k=rng.randint(1, 15))))

    def test_long_chunks_use_same_merge_order(self):
        # Pair ties, stale heap entries, uncommon Unicode bytes and long runs.
        for text in ["ab" * 2048, "x" * 8192, "a\u200d" * 1000,
                     "न" * 4096, "नमस्ते " * 512, "\\" + "a" * 4096,
                     "1 " * 300 + "क", "क " + ", " * 300 + "क"]:
            self.check(text)

    def test_native_scissors_match_ordered_regex(self):
        import json
        artifact = json.loads((ROOT / "tokenizer/tokenizer-65k.json").read_text())
        pattern = regex.compile(_translate_scissors(artifact["english_scissors"]))
        atoms = ["a", "Z", "é", "\u0301", "\u093f", "中", "१", "²", "0", "9",
                 " ", "\t", "\n", "\r", "\u0085", "\u2028", "\u2007", "\v", "\f",
                 "'", "’", "\\", "-", "😀", "\u200c", "\ufeff"]
        examples = [a + b + c for a in atoms for b in atoms for c in ("a", "\u0301", "'", "\n")]
        rng = random.Random(2087)
        examples += ["".join(rng.choices(atoms, k=rng.randint(0, 40))) for _ in range(5000)]
        for text in examples:
            self.assertEqual(self.native._core.scissors_words(text), pattern.findall(text), repr(text))

    def test_batch_order_empty_documents_and_normalization(self):
        texts = ["", "café", "e\u0301", "தமிழ்", "a\n    b", "<eod>"]
        actual = self.native.encode_batch_bytes(texts)
        self.assertEqual(actual, [self.native.encode_bytes(text) for text in texts])

    def test_unused_ids_and_strict_inputs(self):
        self.assertEqual(len(self.native.unused_token_ids), 39)
        for text in (None, b"hello", 42, "\ud800", "\x00"):
            with self.assertRaises((ValueError, UnicodeError)): self.native.encode(text)
        for ids in ([-1], [65536], [1.5], [True], [4]):
            with self.assertRaises(ValueError): self.native.decode(ids)
        self.assertEqual(self.native.decode([4], render_specials=True), "<eod>")
        with self.assertRaises(UnicodeError): self.native.decode([32 + 0xc3])

    def test_cache_memory_is_bounded(self):
        for i in range(1000): self.native.encode(f"word{i} নমস্কার {i}")
        stats = self.native.statistics()
        self.assertLessEqual(stats["english_cache_bytes"], 8192)
        self.assertLessEqual(stats["indic_cache_bytes"], 8192)

    def test_identity_pins_runtime(self):
        identity = self.native.identity()
        self.assertEqual(identity["normalization"], "NFC")
        self.assertTrue(identity["roundtrip_checked_per_document"])
        self.assertEqual(identity["unicode_database"], unicodedata.unidata_version)
        self.assertEqual(len(identity["native_binary_sha256"]), 64)
