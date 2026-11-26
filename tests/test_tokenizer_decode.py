import unittest
from pathlib import Path

import torch

from terminus.tokenizer65k import Frozen65kTokenizer
from terminus.tokenizer_decode import FrozenDecodeStream, decode_continuation, unused_token_ids
from terminus.generate import sample_next_token

ROOT = Path(__file__).resolve().parents[1]


class ContextualTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Frozen65kTokenizer(ROOT / "tokenizer/tokenizer-65k.json")

    def test_every_token_boundary_keeps_prefix_and_utf8(self):
        for text in ["नमस्ते दुनिया", "क 2 क", "नमस्ते hello café दुनिया", "👨‍👩‍👧",
                     "\n    def f():\n        return 'தமிழ்'", "<eod> Ġ Ġ हिंदी", "ॲ‌ಠ"]:
            ids = self.tok.encode(text)
            stream = FrozenDecodeStream(self.tok)
            self.assertEqual("".join(stream.step(i) for i in ids) + stream.finish(), text)
            for split in range(len(ids) + 1):
                try: prefix = self.tok.decode(ids[:split])
                except UnicodeError: continue
                continuation, dropped = decode_continuation(self.tok, ids[:split], ids[split:] + [4], stop_ids=(2, 4))
                self.assertEqual(prefix + continuation, text)
                self.assertEqual(dropped, 0)

    def test_only_explicit_cutoff_can_drop_incomplete_suffix(self):
        with self.assertRaises(UnicodeError):
            decode_continuation(self.tok, [], [32 + 0xc3])
        text, dropped = decode_continuation(self.tok, [], [32 + 0xc3], max_token_cutoff=True)
        self.assertEqual((text, dropped), ("", 1))
        with self.assertRaises(UnicodeError):
            decode_continuation(self.tok, [], [32 + 0xc3, 4], stop_ids=(4,), max_token_cutoff=True)
        with self.assertRaises(UnicodeError):
            decode_continuation(self.tok, [], [32 + 0xff], max_token_cutoff=True)

    def test_specials_are_not_silently_removed(self):
        for token in (0, 1, 3, 5, 545):
            with self.assertRaises(ValueError): FrozenDecodeStream(self.tok).step(token)
        with self.assertRaises(ValueError):
            decode_continuation(self.tok, [], [4, 100], stop_ids=(4,))

    def test_sampling_suppresses_reserved_before_greedy_and_topk(self):
        logits = torch.zeros(65536); logits[5] = 100; logits[545] = 99; logits[100] = 98
        forbidden = unused_token_ids(self.tok)
        self.assertEqual(sample_next_token(logits, temperature=0, suppress_token_ids=forbidden), 100)
        self.assertEqual(sample_next_token(logits, top_k=1, suppress_token_ids=forbidden), 100)
        self.assertEqual(logits[5], 100)
        with self.assertRaises(ValueError): sample_next_token(torch.ones(2), suppress_token_ids=(0, 1))
