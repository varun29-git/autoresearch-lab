"""Update->FP8 pipeline: operand parity with terminus.quantization, batched Muon parity."""

import unittest

import torch
from muon import muon_update

from optimizations.activation_cache import ActivationQuantCache
from optimizations.update_fp8_pipeline import batched_muon_update, operand_nbytes, produce_weight_operands
from terminus.quantization import (quantize_weight_dense_fp8_e4m3_2d, quantize_weight_fp8_e4m3_2d)


class TestProduceWeightOperands(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.weight = torch.randn(64, 128)

    def test_ternary_matches_terminus_reference(self):
        codes, scales = quantize_weight_fp8_e4m3_2d(self.weight)
        bundle = produce_weight_operands(self.weight, weight_mode="ternary")
        forward_codes, forward_scales = bundle["forward"]
        self.assertTrue(torch.equal(forward_codes, codes))
        self.assertTrue(torch.equal(forward_scales, scales))
        backward_codes, backward_scales = bundle["backward"]
        self.assertTrue(torch.equal(backward_codes, codes.T.contiguous()))
        self.assertTrue(torch.equal(backward_scales, scales.T.contiguous()))

    def test_dense_matches_terminus_reference(self):
        codes, scales = quantize_weight_dense_fp8_e4m3_2d(self.weight)
        bundle = produce_weight_operands(self.weight, weight_mode="dense", transposed=False)
        self.assertIsNone(bundle["backward"])
        self.assertTrue(torch.equal(bundle["forward"][0], codes))
        self.assertTrue(torch.equal(bundle["forward"][1], scales))

    def test_operand_nbytes_counts_both_orientations(self):
        bundle = produce_weight_operands(self.weight, weight_mode="ternary")
        codes, scales = bundle["forward"]
        single = codes.numel() + scales.numel() * 4
        self.assertEqual(operand_nbytes(bundle), 2 * single)

    def test_validation(self):
        with self.assertRaises(ValueError):
            produce_weight_operands(torch.randn(4), weight_mode="ternary")
        with self.assertRaises(ValueError):
            produce_weight_operands(self.weight, weight_mode="fp16")


class TestBatchedMuonUpdate(unittest.TestCase):
    def test_matches_per_matrix_pinned_muon(self):
        torch.manual_seed(37)
        shapes = [(8, 16), (8, 16), (8, 16), (16, 8)]
        grads = [torch.randn(shape) for shape in shapes]
        momentums = [torch.zeros(shape) for shape in shapes]
        updates = batched_muon_update(grads, momentums, beta=0.95)
        for grad, shape, update in zip(grads, shapes, updates):
            reference_momentum = torch.zeros(shape)
            reference = muon_update(grad.clone(), reference_momentum, beta=0.95)
            torch.testing.assert_close(update, reference, atol=1e-5, rtol=1e-4)

    def test_momentum_updated_in_place(self):
        torch.manual_seed(41)
        grads = [torch.randn(8, 16), torch.randn(8, 16)]
        momentums = [torch.zeros(8, 16), torch.zeros(8, 16)]
        batched_muon_update(grads, momentums, beta=0.95)
        for grad, momentum in zip(grads, momentums):
            torch.testing.assert_close(momentum, grad * (1 - 0.95), atol=1e-6, rtol=1e-5)

    def test_caller_gradients_not_mutated(self):
        torch.manual_seed(43)
        grads = [torch.randn(8, 16), torch.randn(8, 16)]
        snapshots = [grad.clone() for grad in grads]
        batched_muon_update(grads, [torch.zeros(8, 16), torch.zeros(8, 16)], beta=0.95)
        for grad, snapshot in zip(grads, snapshots):
            self.assertTrue(torch.equal(grad, snapshot))

    def test_validation(self):
        with self.assertRaises(ValueError):
            batched_muon_update([torch.randn(8, 16)], [], beta=0.95)
        with self.assertRaises(ValueError):
            batched_muon_update([torch.randn(8)], [torch.randn(8)], beta=0.95)


class TestActivationQuantCache(unittest.TestCase):
    def test_same_tensor_hits_cache(self):
        torch.manual_seed(47)
        cache = ActivationQuantCache()
        values = torch.randn(16, 64)
        first = cache.directions(values)
        second = cache.directions(values)
        self.assertIs(first, second)
        self.assertEqual((cache.hits, cache.misses), (1, 1))

    def test_mutation_invalidates(self):
        cache = ActivationQuantCache()
        values = torch.randn(16, 64)
        first = cache.directions(values)
        values.add_(1.0)
        second = cache.directions(values)
        self.assertIsNot(first, second)
        self.assertEqual(cache.misses, 2)

    def test_orientation_keys_are_distinct(self):
        cache = ActivationQuantCache()
        values = torch.randn(16, 64)
        rowwise = cache.directions(values)
        both = cache.directions(values, transposed=True)
        self.assertIsNot(rowwise, both)
        self.assertIsNone(rowwise[1])
        self.assertIsNotNone(both[1])

    def test_clear(self):
        cache = ActivationQuantCache()
        cache.directions(torch.randn(16, 64))
        cache.clear()
        self.assertEqual(cache._entries, {})


if __name__ == "__main__":
    unittest.main()
