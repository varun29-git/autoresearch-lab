"""CPU gates for the device vocabulary cut. No checkpoint is loaded."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FOLDER))

from terminus.tokenizer65k import Frozen65kTokenizer  # noqa: E402
from vocab.alignment import (english_kl, indic_cross_entropy,  # noqa: E402
                             restore_kept_rows)
from vocab.indic import BYTE_ALPHABET  # noqa: E402
from vocab.model import PINNED_IDS, Cut  # noqa: E402
from vocab.report import evaluate, passed  # noqa: E402
from vocab.transplant import assemble, constituent_row, kept_rows_match  # noqa: E402


class VocabularyCutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cut = Cut()
        cls.oracle = Frozen65kTokenizer(cls.cut.donor_path)
        cls.report = evaluate(cls.cut, cls.oracle)

    def test_report_passes(self):
        self.assertTrue(passed(self.report), self.report)

    def test_seven_script_row_stays_and_cjk_goes(self):
        self.assertTrue(self.report["examples"]["devanagari"]["kept"])
        self.assertFalse(self.report["examples"]["cjk"]["kept"])

    def test_pinned_ids_and_bytes(self):
        for index in range(PINNED_IDS):
            token = self.cut.token_at[index]
            self.assertEqual(self.cut.old_to_new[index], index)
            self.assertEqual(self.cut.compact_ids[token], index)
        present = [char for char in BYTE_ALPHABET if char in self.cut.token_at.values()]
        self.assertGreater(len(present), 240)
        self.assertTrue(all(char in self.cut.keep for char in present))
        self.assertTrue(self.cut.closure_ok)

    def test_new_pieces_do_not_reuse_a_spark_surface(self):
        for piece in self.cut.new_pieces:
            surface = piece.replace("\u0120", " ")
            self.assertNotIn(surface, self.cut.surface_to_old)
        self.assertGreater(self.cut.reused_sarvam, 0)
        self.assertGreater(len(self.cut.new_pieces), 0)
        self.assertLess(self.cut.vocab_size, len(self.cut.token_at))

    def test_explicit_keep_ids_append_indic_and_roundtrip_english(self):
        from vocab.indic import is_target_char
        from vocab.model import decode_token
        indic_ids = []
        for old, token in self.cut.token_at.items():
            surface = decode_token(token)
            if surface and any(is_target_char(char) for char in surface):
                indic_ids.append(old)
        self.assertGreater(len(indic_ids), 1000)
        narrowed = Cut(keep_ids=set(range(8000)) | set(indic_ids))
        self.assertEqual(narrowed.old_to_new[0], 0)
        self.assertGreater(narrowed.reused_sarvam, 0)
        self.assertGreater(len(narrowed.new_pieces), 0)
        text = "The quick brown fox."
        ids = narrowed.encode_ids(text)
        self.assertGreater(len(ids), 1)
        self.assertEqual(narrowed.compact.decode(ids), text)
        hindi = narrowed.encode_ids("नमस्ते")
        self.assertGreater(len(hindi), 0)
        self.assertIn("नम", narrowed.decode_ids(hindi))


class TransplantTests(unittest.TestCase):
    def test_kept_rows_are_bitwise_and_new_row_is_scaled_mean(self):
        weight = torch.randn(8, 4)
        kept = [0, 1, 2, 4]
        constituents = [[1, 4, 4]]
        assembled = assemble(weight, kept, constituents)
        self.assertTrue(kept_rows_match(assembled, weight, kept))
        self.assertTrue(torch.equal(assembled[4], constituent_row(weight, constituents[0])))
        self.assertEqual(tuple(assembled.shape), (5, 4))

    def test_bf16_kept_rows_keep_their_bits(self):
        weight = torch.randn(6, 3, dtype=torch.bfloat16)
        kept = [5, 1, 0]
        assembled = assemble(weight, kept, [[2, 3]])
        self.assertTrue(kept_rows_match(assembled, weight, kept))


class AlignmentTests(unittest.TestCase):
    def test_quiet_new_row_matches_renormalized_teacher(self):
        teacher = torch.tensor([[0.0, 2.0, -1.0]])
        quiet = torch.tensor([[0.0, 2.0, -1.0, -80.0]])
        loud = torch.tensor([[0.0, 2.0, -1.0, 20.0]])
        quiet_kl = float(english_kl(quiet, teacher))
        loud_kl = float(english_kl(loud, teacher))
        self.assertLess(quiet_kl, 1e-4)
        self.assertGreater(loud_kl, quiet_kl)

    def test_indic_cross_entropy_hits_the_target_id(self):
        logits = torch.tensor([[0.0, 5.0, 0.0]])
        low = float(indic_cross_entropy(logits, torch.tensor([1])))
        high = float(indic_cross_entropy(logits, torch.tensor([0])))
        self.assertLess(low, high)

    def test_adamw_decay_is_undone_on_kept_rows(self):
        torch.manual_seed(0)
        parameter = torch.nn.Parameter(torch.randn(6, 3))
        snapshot = parameter.detach().clone()
        optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.1)
        loss = parameter.sum()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(parameter[:4], snapshot[:4]))
        restore_kept_rows(parameter, snapshot, n_kept=4)
        self.assertTrue(torch.equal(parameter[:4], snapshot[:4]))
        self.assertFalse(torch.equal(parameter[4:], snapshot[4:]))


if __name__ == "__main__":
    unittest.main()
