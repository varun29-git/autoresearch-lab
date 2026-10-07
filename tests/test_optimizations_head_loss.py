"""Fused head loss: exact full-vocabulary parity with plain cross entropy."""

import unittest

import torch
import torch.nn.functional as functional

from optimizations.fused_head_loss import fused_head_loss


class TestFusedHeadLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.hidden = torch.randn(2, 7, 24)
        self.weight = torch.randn(40, 24)
        self.targets = torch.randint(0, 40, (2, 7))

    def reference(self, hidden, weight, targets):
        logits = functional.linear(hidden, weight).float()
        return functional.cross_entropy(logits.reshape(-1, 40), targets.reshape(-1),
                                        ignore_index=-100, reduction="sum")

    def test_loss_matches_plain_cross_entropy(self):
        loss = fused_head_loss(self.hidden, self.weight, self.targets, tile_size=16)
        torch.testing.assert_close(loss, self.reference(self.hidden, self.weight, self.targets),
                                   atol=1e-5, rtol=1e-5)

    def test_gradients_match_plain_cross_entropy(self):
        hidden = self.hidden.clone().requires_grad_()
        weight = self.weight.clone().requires_grad_()
        fused_head_loss(hidden, weight, self.targets, tile_size=16).backward()
        reference_hidden = self.hidden.clone().requires_grad_()
        reference_weight = self.weight.clone().requires_grad_()
        self.reference(reference_hidden, reference_weight, self.targets).backward()
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-5, rtol=1e-4)

    def test_ignore_index_rows_contribute_nothing(self):
        targets = self.targets.clone()
        targets[:, :3] = -100
        loss = fused_head_loss(self.hidden, self.weight, targets, tile_size=16)
        torch.testing.assert_close(loss, self.reference(self.hidden, self.weight, targets),
                                   atol=1e-5, rtol=1e-5)

    def test_tile_size_does_not_change_result(self):
        coarse = fused_head_loss(self.hidden, self.weight, self.targets, tile_size=64)
        fine = fused_head_loss(self.hidden, self.weight, self.targets, tile_size=3)
        torch.testing.assert_close(coarse, fine, atol=1e-5, rtol=1e-5)

    def test_bf16_inputs(self):
        hidden = self.hidden.bfloat16().requires_grad_()
        loss = fused_head_loss(hidden, self.weight, self.targets, tile_size=16)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(hidden.grad.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(hidden.grad.float()).all())

    def test_fp8_head_changes_numerics_but_stays_sane(self):
        hidden = self.hidden.clone().requires_grad_()
        weight = self.weight.clone().requires_grad_()
        loss_fp8 = fused_head_loss(hidden, weight, self.targets, tile_size=16, fp8_head=True)
        self.assertTrue(torch.isfinite(loss_fp8))
        loss_fp8.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())
        reference = fused_head_loss(self.hidden, self.weight, self.targets, tile_size=16)
        self.assertLess(abs(loss_fp8 - reference).item(), 0.05 * abs(reference).item() + 1e-3)

    def test_validation(self):
        with self.assertRaises(ValueError):
            fused_head_loss(self.hidden, self.weight, self.targets, tile_size=0)
        with self.assertRaises(ValueError):
            fused_head_loss(self.hidden, self.weight, self.targets[:, :5], tile_size=16)
        with self.assertRaises(ValueError):
            fused_head_loss(self.hidden, self.weight.float().T, self.targets, tile_size=16)


if __name__ == "__main__":
    unittest.main()
