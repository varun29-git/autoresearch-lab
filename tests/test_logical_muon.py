import copy
import math
import unittest

import torch

from terminus.model import BitNetLlama
from terminus.optim import OptimizerConfig, build_optimizer
from tests.helpers import tiny_config


class LogicalMuonTests(unittest.TestCase):
    def test_fused_projection_groups_preserve_logical_matrices(self):
        model = BitNetLlama(tiny_config(fuse_projections=True))
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        groups = {id(group["params"][0]): group for group in optimizer.param_groups if group["use_muon"]}
        for layer in model.layers:
            qkv = groups[id(layer.attn.qkv_proj.weight)]
            gate_up = groups[id(layer.ffn.gate_up_proj.weight)]
            self.assertEqual(qkv["logical_rows"], (32, 16, 16))
            self.assertEqual(gate_up["logical_rows"], (64, 64))
            self.assertAlmostEqual(qkv["lr"], .003 * .2 * math.sqrt(32))
            self.assertAlmostEqual(qkv["lr"] * qkv["weight_decay"], .003 * .1)
        self.assertEqual(sum(len(group["params"]) for group in optimizer.param_groups),
                         sum(parameter.requires_grad for parameter in model.parameters()))

    def test_fused_update_matches_archived_separate_matrix_muon(self):
        from muon import SingleDeviceMuonWithAuxAdam
        from terminus.logical_muon import LogicalMuonWithAuxAdam

        torch.manual_seed(19)
        for rows in ((32, 16, 16), (64, 64)):
            with self.subTest(rows=rows):
                weight = torch.nn.Parameter(torch.randn(sum(rows), 32))
                separate = [torch.nn.Parameter(chunk.clone()) for chunk in weight.detach().split(rows)]
                scale = .2 * math.sqrt(weight.shape[1])
                options = {"lr": .003 * scale, "weight_decay": .1 / scale, "momentum": .95, "use_muon": True}
                fused = LogicalMuonWithAuxAdam([{"params": [weight], **options}], [rows])
                reference = SingleDeviceMuonWithAuxAdam([{"params": [part], **options} for part in separate])
                resumed = None
                for index in range(2):
                    gradient = torch.randn_like(weight)
                    weight.grad = gradient.clone()
                    for part, chunk in zip(separate, gradient.split(rows)):
                        part.grad = chunk.clone()
                    fused.step()
                    reference.step()
                    torch.testing.assert_close(weight, torch.cat(separate), rtol=0, atol=0)
                    torch.testing.assert_close(fused.state[weight]["momentum_buffer"],
                                               torch.cat([reference.state[part]["momentum_buffer"] for part in separate]),
                                               rtol=0, atol=0)
                    if index == 0:
                        resumed_weight = torch.nn.Parameter(weight.detach().clone())
                        resumed = LogicalMuonWithAuxAdam([{"params": [resumed_weight], **options}], [rows])
                        resumed.load_state_dict(copy.deepcopy(fused.state_dict()))
                        self.assertEqual(resumed.param_groups[0]["logical_rows"], rows)
                    else:
                        resumed_weight.grad = gradient.clone()
                        resumed.step()
                        torch.testing.assert_close(resumed_weight, weight, rtol=0, atol=0)


    def test_batched_step_matches_reference_loop(self):
        # Class-E parity: the batched launch path must reproduce the reference
        # loop's parameter and optimizer-state evolution (Newton-Schulz runs as
        # bmm vs mm, so only reduction-order noise is tolerated).
        from terminus.logical_muon import LogicalMuonWithAuxAdam

        torch.manual_seed(23)
        for rows in ((32, 16, 16), (64, 64)):
            with self.subTest(rows=rows):
                initial = torch.randn(sum(rows), 32)
                weight_a = torch.nn.Parameter(initial.clone())
                weight_b = torch.nn.Parameter(initial.clone())
                options = {"lr": .003 * .2 * math.sqrt(32), "weight_decay": .1 / (.2 * math.sqrt(32)),
                           "momentum": .95, "use_muon": True}
                batched = LogicalMuonWithAuxAdam([{"params": [weight_a], **options}], [rows], batched=True)
                reference = LogicalMuonWithAuxAdam([{"params": [weight_b], **options}], [rows], batched=False)
                for _ in range(3):
                    gradient = torch.randn_like(initial)
                    weight_a.grad = gradient.clone()
                    weight_b.grad = gradient.clone()
                    batched.step()
                    reference.step()
                    torch.testing.assert_close(weight_a, weight_b, rtol=1e-3, atol=1e-5)
                    torch.testing.assert_close(batched.state[weight_a]["momentum_buffer"],
                                               reference.state[weight_b]["momentum_buffer"],
                                               rtol=0, atol=0)

    def test_batched_step_matches_reference_adam_groups(self):
        from terminus.logical_muon import LogicalMuonWithAuxAdam

        torch.manual_seed(29)
        specs = [torch.randn(64, 32), torch.randn(48), torch.randn(16)]
        weights_a = [torch.nn.Parameter(value.clone()) for value in specs]
        weights_b = [torch.nn.Parameter(value.clone()) for value in specs]
        groups = lambda ws: [{"params": ws, "lr": .003, "weight_decay": .1,
                              "betas": (.9, .95), "eps": 1e-10, "use_muon": False}]
        batched = LogicalMuonWithAuxAdam(groups(weights_a), [None], batched=True)
        reference = LogicalMuonWithAuxAdam(groups(weights_b), [None], batched=False)
        for _ in range(3):
            for wa, wb in zip(weights_a, weights_b):
                gradient = torch.randn_like(wa)
                wa.grad = gradient.clone()
                wb.grad = gradient.clone()
            batched.step()
            reference.step()
        for wa, wb in zip(weights_a, weights_b):
            torch.testing.assert_close(wa, wb, rtol=0, atol=0)
            torch.testing.assert_close(batched.state[wa]["exp_avg"], reference.state[wb]["exp_avg"], rtol=0, atol=0)
            torch.testing.assert_close(batched.state[wa]["exp_avg_sq"], reference.state[wb]["exp_avg_sq"], rtol=0, atol=0)

    def test_learned_qk_norm_group_builds_and_steps(self):
        # Regression: the named qk_norm aux group must not reach the pinned Muon
        # constructor with extra keys (baseline campaign config uses learned_rms).
        model = BitNetLlama(tiny_config(qk_norm_mode="learned_rms"))
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        named = {group.get("name"): group for group in optimizer.param_groups}
        qk_group = named["qk_norm"]
        self.assertFalse(qk_group["use_muon"])
        self.assertEqual(qk_group["lr"], OptimizerConfig().qk_norm_lr)
        self.assertEqual(qk_group["weight_decay"], 0.0)
        self.assertEqual({id(parameter) for parameter in qk_group["params"]},
                         {id(layer.attn.q_norm.weight) for layer in model.layers} |
                         {id(layer.attn.k_norm.weight) for layer in model.layers})
        ids = torch.randint(0, 32, (2, 16))
        for _ in range(2):
            optimizer.zero_grad()
            model(ids, torch.randint(0, 32, (2, 16)))["loss"].backward()
            optimizer.step()
        for layer in model.layers:
            self.assertTrue(torch.isfinite(layer.attn.q_norm.weight).all())


if __name__ == "__main__":
    unittest.main()
