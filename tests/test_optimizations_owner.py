"""Owner optimizer (idea 1): layout, parity with pinned Muon, clipping, replicas."""

import math
import unittest

import torch
from muon import muon_update

from optimizations.owner_optimizer import (OwnerLayout, OwnerMuonOptimizer, SingleProcessCollectives,
                                           replicated_state_bytes, ring_volume_bytes)
from terminus.optim import muon_update_scale_for_param


SHAPES = {"a": (8, 16), "b": (8, 16), "c": (16, 8), "d": (32, 16)}


class TestOwnerLayout(unittest.TestCase):
    def test_covers_each_projection_once(self):
        layout = OwnerLayout(SHAPES, 3)
        self.assertEqual(sorted(layout.owner_of), sorted(SHAPES))
        self.assertTrue(all(0 <= owner < 3 for owner in layout.owner_of.values()))

    def test_balances_by_parameter_count(self):
        layout = OwnerLayout(SHAPES, 2)
        # d (512) alone on one rank; a+b+c (128*3) on the other.
        self.assertEqual(sorted(layout.loads), [384, 512])
        self.assertEqual(layout.owned_by(layout.owner_of["d"]), ["d"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            OwnerLayout(SHAPES, 0)
        with self.assertRaises(ValueError):
            OwnerLayout({}, 2)
        with self.assertRaises(ValueError):
            OwnerLayout({"bad": (4, 4, 4)}, 1)


def reference_step(masters, momentums, grads, *, muon_lr, weight_decay, momentum):
    for name in masters:
        scale = muon_update_scale_for_param(masters[name], rule="sqrt_fan_in")
        lr, decay = muon_lr * scale, weight_decay / scale
        update = muon_update(grads[name].clone(), momentums[name], beta=momentum)
        masters[name].mul_(1 - lr * decay)
        masters[name].add_(update, alpha=-lr)


class TestOwnerMuonOptimizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.shapes = {"a": (8, 16), "b": (8, 16), "c": (16, 8)}
        self.initial = {name: torch.randn(shape) for name, shape in self.shapes.items()}
        self.grads = {name: torch.randn(shape) for name, shape in self.shapes.items()}

    def make_optimizer(self, **kwargs):
        optimizer = OwnerMuonOptimizer(self.shapes, comm=SingleProcessCollectives(),
                                       weight_mode="ternary", **kwargs)
        optimizer.initialize_masters(self.initial)
        return optimizer

    def test_step_matches_pinned_muon(self):
        muon_lr, weight_decay, momentum = 3e-3, 0.1, 0.95
        optimizer = self.make_optimizer(muon_lr=muon_lr, weight_decay=weight_decay, momentum=momentum)
        masters = {name: weight.clone() for name, weight in self.initial.items()}
        momentums = {name: torch.zeros(shape) for name, shape in self.shapes.items()}
        reference_step(masters, momentums, self.grads,
                       muon_lr=muon_lr, weight_decay=weight_decay, momentum=momentum)
        optimizer.step(self.grads)
        for name in self.shapes:
            torch.testing.assert_close(optimizer.slots[name]["master"], masters[name],
                                       atol=1e-6, rtol=1e-5)

    def test_multi_step_parity(self):
        optimizer = self.make_optimizer()
        masters = {name: weight.clone() for name, weight in self.initial.items()}
        momentums = {name: torch.zeros(shape) for name, shape in self.shapes.items()}
        for _ in range(3):
            torch.manual_seed(59)
            grads = {name: torch.randn(shape) for name, shape in self.shapes.items()}
            reference_step(masters, momentums, grads, muon_lr=3e-3, weight_decay=0.1, momentum=0.95)
            optimizer.step(grads)
        for name in self.shapes:
            torch.testing.assert_close(optimizer.slots[name]["master"], masters[name],
                                       atol=1e-5, rtol=1e-4)

    def test_clipping_scales_gradients_globally(self):
        optimizer = self.make_optimizer()
        clipped = self.make_optimizer()
        # At world_size=1 the clip coefficient is max_norm / global_grad_norm.
        global_norm = math.sqrt(sum(grad.double().square().sum().item() for grad in self.grads.values()))
        clipped.step(self.grads, max_norm=global_norm / 2)
        optimizer.step({name: grad / 2 for name, grad in self.grads.items()})
        for name in self.shapes:
            torch.testing.assert_close(clipped.slots[name]["master"],
                                       optimizer.slots[name]["master"], atol=1e-6, rtol=1e-5)

    def test_replicas_match_master_quantization(self):
        from optimizations.update_fp8_pipeline import produce_weight_operands

        optimizer = self.make_optimizer()
        optimizer.step(self.grads)
        for name, slot in optimizer.slots.items():
            reference = produce_weight_operands(slot["master"], weight_mode="ternary")
            for key in ("forward", "backward"):
                self.assertTrue(torch.equal(slot["replica"][key][0], reference[key][0]))
                self.assertTrue(torch.equal(slot["replica"][key][1], reference[key][1]))

    def test_state_dict_round_trip(self):
        optimizer = self.make_optimizer()
        optimizer.step(self.grads)
        state = optimizer.state_dict()
        self.assertEqual(sorted(state), sorted(self.shapes))
        restored = self.make_optimizer()
        restored.load_state_dict(state)
        for name in self.shapes:
            torch.testing.assert_close(restored.slots[name]["master"],
                                       optimizer.slots[name]["master"], atol=0, rtol=0)

    def test_gradient_shape_validation(self):
        optimizer = self.make_optimizer()
        with self.assertRaises(ValueError):
            optimizer.step({"a": torch.randn(4, 4)})


class TestAccounting(unittest.TestCase):
    def test_owner_ring_volume_is_smaller(self):
        params = 1_200_000_000
        owner = ring_volume_bytes(params, 8, scheme="owner")
        baseline = ring_volume_bytes(params, 8, scheme="allreduce_fp32")
        self.assertLess(owner, baseline)
        self.assertAlmostEqual(owner / baseline, (4 + 2 * (1 + 4 / 128**2)) / 8, places=3)

    def test_owner_state_drops_replicated_masters(self):
        params = 1_200_000_000
        owner = replicated_state_bytes(params, 8, scheme="owner")
        baseline = replicated_state_bytes(params, 8, scheme="allreduce_fp32")
        self.assertGreater(baseline - owner, 10 * (1 << 30))  # > 10 GiB saved per GPU

    def test_accounting_validation(self):
        with self.assertRaises(ValueError):
            ring_volume_bytes(100, 0)
        with self.assertRaises(ValueError):
            ring_volume_bytes(100, 8, scheme="fp16")


if __name__ == "__main__":
    unittest.main()
