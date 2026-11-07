import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

from production.tokenizer_revision import build_candidate, expand_atom_ids, ATOM_CANDIDATE_SHA256, ATOM_REPLACEMENTS
from terminus import unicode_contract as unicode
from terminus.tokenizer65k import Frozen65kTokenizer

ROOT = Path(__file__).resolve().parents[1]


class AtomRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "tokenizer.json"
        cls.base = ROOT / "tokenizer/tokenizer-65k.json"
        payload, cls.changes = build_candidate(cls.base)
        cls.path.write_bytes(payload)
        cls.original = Frozen65kTokenizer(cls.base)
        cls.candidate = Frozen65kTokenizer(cls.path, expected_sha256=ATOM_CANDIDATE_SHA256)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_only_retired_slots_changed(self):
        old = json.loads(self.base.read_text()); new = json.loads(self.path.read_text())
        changed = {i for i, (a, b) in enumerate(zip(old["pieces"], new["pieces"])) if a != b}
        self.assertEqual(changed, {i for i, _ in ATOM_REPLACEMENTS})
        self.assertEqual(old["english_merges"], new["english_merges"])
        self.assertEqual(old["tokenizer_spec"], new["tokenizer_spec"])
        self.assertEqual(old["pieces"][:32], new["pieces"][:32])
        self.assertEqual(len(set(new["pieces"])), 65536)

    def test_contexts_expand_exactly_to_original(self):
        for _, char in ATOM_REPLACEMENTS:
            for context in [char, "नमस्ते " + char, "a " + char + " b", "1 " + char + " 2",
                            "क्" + char + "क", char + "\n" + char, "क " + char + " ಕ"]:
                text = unicode.normalize("NFC", context)
                new = self.candidate.encode(text); old = self.original.encode(text)
                self.assertEqual(expand_atom_ids(new), old, repr(text))
                self.assertLessEqual(len(new), len(old))
                self.assertEqual(self.candidate.decode(new), text)

    def test_seeded_atom_and_script_boundaries(self):
        rng = random.Random(12)
        atoms = [c for _, c in ATOM_REPLACEMENTS] + ["नमस्ते", "বাংলা", "hello", "1", " ", "\n", "\t", "Ġ", "<eod>", "\u200d"]
        for _ in range(2000):
            text = unicode.normalize("NFC", "".join(rng.choices(atoms, k=rng.randint(1, 20))))
            new = self.candidate.encode(text)
            self.assertEqual(expand_atom_ids(new), self.original.encode(text), repr(text))
            self.assertEqual(self.candidate.decode(new), text)

    @unittest.skipUnless(importlib.util.find_spec("_terminus_tokenizer_native"), "native extension required")
    def test_native_candidate_agrees(self):
        from terminus.tokenizer_native import Native65kTokenizer
        tok = Native65kTokenizer(self.path, expected_sha256=ATOM_CANDIDATE_SHA256)
        self.assertEqual(tok.unused_token_ids, tuple(range(5, 32)))
        for _, char in ATOM_REPLACEMENTS:
            for text in [char, "क " + char + " ಕ", "1. " + char]:
                self.assertEqual(tok.encode(text), self.candidate.encode(unicode.normalize("NFC", text)))
