import copy
import unittest

import torch

from terminus.hpo_optimizer import build_hpo_optimizer
from terminus.model import BitNetLlama, RMSNorm
from terminus.optim import OptimizerConfig, build_optimizer, muon_update_scale_for_param
from tests.helpers import tiny_config


def stack_role_fixture():
    """Exercise the selected stack's extra parameter names without changing base model code."""
    model = BitNetLlama(tiny_config(qk_norm_mode="learned_rms", fuse_projections=True))
    for index, layer in enumerate(model.layers):
        layer.attn_out_norm = RMSNorm(model.config.dim, model.config.rms_norm_eps)
        layer.ffn_out_norm = RMSNorm(model.config.dim, model.config.rms_norm_eps)
        if index:
            layer.attn.value_lambda = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    return model


def architecture_rates():
    return OptimizerConfig(kind="muon", muon_lr=.003, adamw_lr=.003,
                           norm_lr=.0002, qk_norm_lr=.0001, eps=1e-10)


class HPOOptimizerTests(unittest.TestCase):
    def test_selected_stack_roles_and_exact_parameter_coverage(self):
        model = stack_role_fixture()
        optimizer = build_hpo_optimizer(model, architecture_rates())
        by_role = {}
        for group in optimizer.param_groups:
            by_role.setdefault(group["role"], []).extend(group["param_names"])
            self.assertEqual(len(group["params"]), len(group["param_names"]))
            self.assertEqual(group["base_lr"], group["lr"])
        self.assertEqual(set(by_role), {
            "embedding_readout", "block_norm", "peri_norm", "final_norm",
            "qk_gain", "value_residual", "sub_norm",
            "matrix.attn_qkv", "matrix.attn_o", "matrix.ffn_gate_up", "matrix.ffn_down",
        })
        self.assertEqual(by_role["embedding_readout"], ["tok_embeddings.weight"])
        self.assertEqual(by_role["final_norm"], ["norm.weight"])
        self.assertEqual(by_role["value_residual"], ["layers.1.attn.value_lambda"])
        self.assertEqual(len(by_role["peri_norm"]), 4)
        self.assertEqual(len(by_role["qk_gain"]), 4)
        self.assertEqual(set(name for names in by_role.values() for name in names),
                         set(dict(model.named_parameters())))

    def test_same_rates_match_architecture_optimizer_and_resume(self):
        torch.manual_seed(181)
        baseline_model = stack_role_fixture()
        hpo_model = copy.deepcopy(baseline_model)
        config = architecture_rates()
        baseline = build_optimizer(baseline_model, config)
        hpo = build_hpo_optimizer(hpo_model, config)

        for step in range(3):
            gradients = {name: torch.randn_like(parameter)
                         for name, parameter in baseline_model.named_parameters()}
            for model in (baseline_model, hpo_model):
                for name, parameter in model.named_parameters():
                    parameter.grad = gradients[name].clone()
            baseline.step()
            hpo.step()
            for (name, old), (same_name, new) in zip(baseline_model.named_parameters(),
                                                       hpo_model.named_parameters()):
                self.assertEqual(name, same_name)
                torch.testing.assert_close(old, new, rtol=0, atol=0)
                self.assertEqual(set(baseline.state[old]), set(hpo.state[new]))
                for key, value in baseline.state[old].items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, hpo.state[new][key], rtol=0, atol=0)
                    else:
                        self.assertEqual(value, hpo.state[new][key])
            if step == 1:
                resumed_model = copy.deepcopy(hpo_model)
                resumed = build_hpo_optimizer(resumed_model, config)
                resumed.load_state_dict(copy.deepcopy(hpo.state_dict()))
            if step == 2:
                for name, parameter in resumed_model.named_parameters():
                    parameter.grad = gradients[name].clone()
                resumed.step()
                for p, q in zip(hpo_model.parameters(), resumed_model.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_role_rate_override_and_validation(self):
        model = stack_role_fixture()
        config = architecture_rates()
        optimizer = build_hpo_optimizer(model, config, role_lrs={
            "embedding_readout": .01, "value_residual": .0008, "matrix.attn_o": .012,
        })
        for group in optimizer.param_groups:
            if group["role"] == "embedding_readout":
                self.assertEqual(group["lr"], .01)
                self.assertEqual(group["weight_decay"], .1)
            elif group["role"] == "value_residual":
                self.assertEqual(group["lr"], .0008)
                self.assertEqual(group["weight_decay"], 0.0)
            elif group["role"] == "matrix.attn_o":
                parameter = group["params"][0]
                scale = muon_update_scale_for_param(parameter)
                self.assertAlmostEqual(group["lr"], .012 * scale)
                self.assertAlmostEqual(group["lr"] * group["weight_decay"], .012 * .1)
            elif group["role"] == "matrix.attn_qkv":
                parameter = group["params"][0]
                self.assertAlmostEqual(group["lr"], .003 * muon_update_scale_for_param(parameter))
        for overrides in ({"unknown": .01}, {"peri_norm": float("nan")},
                          {"peri_norm": 0}, {"peri_norm": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                build_hpo_optimizer(model, config, role_lrs=overrides)
        plain = BitNetLlama(tiny_config())
        with self.assertRaisesRegex(ValueError, "absent from model"):
            build_hpo_optimizer(plain, config, role_lrs={"peri_norm": .0002})
        plain.register_parameter("unclassified_gain", torch.nn.Parameter(torch.ones(2)))
        with self.assertRaisesRegex(ValueError, "unclassified AdamW parameter"):
            build_hpo_optimizer(plain, config)
        with self.assertRaisesRegex(ValueError, "pinned Muon"):
            build_hpo_optimizer(model, OptimizerConfig(kind="adamw"))


if __name__ == "__main__":
    unittest.main()
