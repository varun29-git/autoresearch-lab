import copy
import unittest

import torch
from muon import zeropower_via_newtonschulz5

from terminus.logical_muon import LogicalMuonWithAuxAdam
from terminus.model import BitNetLlama
from terminus.optim import OptimizerConfig, build_optimizer
from tests.helpers import tiny_config


class MuonHyperballTests(unittest.TestCase):
    def test_logical_matrix_update_matches_hyperball_equation_and_resume(self):
        torch.manual_seed(71)
        # Production shape ratios: Q/K/V, gate/up and down.
        for batched in (False, True):
            for rows, cols in (((24, 8, 8), 24), ((60, 60), 24), ((24,), 60)):
                with self.subTest(batched=batched, rows=rows, cols=cols):
                    original = torch.randn(sum(rows), cols)
                    parameter = torch.nn.Parameter(original.clone())
                    expected = original.clone()
                    momentum = torch.zeros_like(original)
                    radii = [part.norm() for part in original.split(rows)]
                    group = {"params": [parameter], "lr": .02, "weight_decay": 0.,
                             "momentum": .95, "use_muon": True}
                    optimizer = LogicalMuonWithAuxAdam([group], [rows], batched=batched,
                                                        adjust_lr_fn="hyperball")
                    for index, lr in enumerate((.02, 0., .007)):
                        gradient = torch.randn_like(original)
                        parameter.grad = gradient.clone()
                        optimizer.param_groups[0]["lr"] = lr
                        momentum.lerp_(gradient, .05)
                        direction = gradient.lerp(momentum, .95)
                        for weight, source, radius in zip(expected.split(rows), direction.split(rows), radii):
                            raw = zeropower_via_newtonschulz5(source, steps=5).float()
                            trial = weight - lr * radius * raw / raw.norm()
                            weight.copy_(radius * trial / trial.norm())
                        optimizer.step()
                        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], momentum,
                                                   rtol=1e-6, atol=1e-7)
                        torch.testing.assert_close(optimizer.state[parameter]["hyperball_radii"],
                                                   torch.stack(radii), rtol=0, atol=0)
                        for part, radius in zip(parameter.split(rows), radii):
                            torch.testing.assert_close(part.norm(), radius, rtol=1e-6, atol=1e-6)
                        if index == 1:
                            resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
                            # The pinned constructor mutates the original group
                            # with metadata; rebuild only its accepted input keys.
                            resumed_group = {"params": [resumed_parameter], "lr": .02,
                                             "weight_decay": 0., "momentum": .95, "use_muon": True}
                            resumed = LogicalMuonWithAuxAdam([resumed_group], [rows], batched=batched,
                                                               adjust_lr_fn="hyperball")
                            resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                        elif index == 2:
                            resumed.param_groups[0]["lr"] = lr
                            resumed_parameter.grad = gradient.clone()
                            resumed.step()
                            torch.testing.assert_close(resumed_parameter, parameter, rtol=0, atol=0)

    def test_model_groups_and_auxiliary_optimizer(self):
        torch.manual_seed(73)
        model = BitNetLlama(tiny_config(fuse_projections=True, qk_norm_mode="learned_rms"))
        config = OptimizerConfig(kind="muon", muon_lr_rule="hyperball", muon_lr=.01,
                                 adamw_lr=.003, weight_decay=.1)
        optimizer = build_optimizer(model, config)
        groups = optimizer.param_groups
        self.assertTrue(any(group["use_muon"] for group in groups))
        self.assertTrue(any(not group["use_muon"] and group["weight_decay"] == .1 for group in groups))
        for group in groups:
            if group["use_muon"]:
                self.assertEqual(group["weight_decay"], 0.)
                self.assertEqual(group["lr"], .01)
                self.assertEqual(group["adjust_lr_fn"], "hyperball")
        ids = torch.randint(0, model.config.vocab_size, (2, 16))
        model(ids, ids)["loss"].backward()
        optimizer.step()
        for group in groups:
            if not group["use_muon"]:
                continue
            parameter = group["params"][0]
            radii = optimizer.state[parameter]["hyperball_radii"]
            self.assertEqual(len(radii), len(group.get("logical_rows", (parameter.shape[0],))))
            for part, radius in zip(parameter.split(group.get("logical_rows", (parameter.shape[0],))), radii):
                torch.testing.assert_close(part.norm(), radius, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.isfinite(model.tok_embeddings.weight).all())

    def test_hyperball_checkpoint_cannot_load_as_spectral(self):
        def make(rule):
            parameter = torch.nn.Parameter(torch.randn(8, 16))
            group = {"params": [parameter], "lr": .01, "weight_decay": 0.,
                     "momentum": .95, "use_muon": True}
            return LogicalMuonWithAuxAdam([group], [None], adjust_lr_fn=rule)

        spectral, hyperball = make("spectral_unclamped"), make("hyperball")
        for target, source in ((spectral, hyperball), (hyperball, spectral)):
            with self.assertRaisesRegex(ValueError, "scaling differs"):
                target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    unittest.main()
