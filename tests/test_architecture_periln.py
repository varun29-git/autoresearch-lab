"""Peri-LN branch output normalization (trial-001): contract and numerics tests."""

import torch

from terminus.model import BitNetLlama, BitNetLlamaConfig, RMSNorm

from tests.helpers import tiny_config


def make_model(**changes):
    torch.manual_seed(7)
    return BitNetLlama(tiny_config(**changes))


class TestPeriLNConfig:
    def test_default_off(self):
        config = tiny_config()
        assert config.branch_norm == "none"
        model = BitNetLlama(config)
        for layer in model.layers:
            assert isinstance(layer.attn_out_norm, torch.nn.Identity)
            assert isinstance(layer.ffn_out_norm, torch.nn.Identity)

    def test_invalid_mode_rejected(self):
        try:
            tiny_config(branch_norm="subln")
        except ValueError:
            return
        raise AssertionError("unknown branch_norm must be rejected")

    def test_peri_adds_channel_gains(self):
        model = make_model(branch_norm="peri")
        for layer in model.layers:
            assert isinstance(layer.attn_out_norm, RMSNorm)
            assert isinstance(layer.ffn_out_norm, RMSNorm)
            assert layer.attn_out_norm.weight.shape == (model.config.dim,)
            assert layer.ffn_out_norm.weight.shape == (model.config.dim,)


class TestPeriLNForward:
    def test_shapes_and_loss(self):
        model = make_model(branch_norm="peri")
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        out = model(ids, targets)
        assert out["logits"].shape == (2, 16, 32)
        assert torch.isfinite(out["loss"])

    def test_causality(self):
        model = make_model(branch_norm="peri").eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            reference = model(ids)["logits"]
            changed = ids.clone()
            changed[0, 8:] = (changed[0, 8:] + 1) % 32
            altered = model(changed)["logits"]
        assert torch.allclose(reference[:, :8], altered[:, :8], atol=1e-5)

    def test_gradients_reach_gains_and_projections(self):
        model = make_model(branch_norm="peri")
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        model(ids, targets)["loss"].backward()
        for layer in model.layers:
            assert layer.attn_out_norm.weight.grad is not None
            assert torch.isfinite(layer.attn_out_norm.weight.grad).all()
            assert layer.ffn_out_norm.weight.grad.abs().sum() > 0
            assert layer.attn.o_proj.weight.grad is not None

    def test_cached_matches_full_forward(self):
        model = make_model(branch_norm="peri").eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            full = model(ids)["logits"]
            _, cache = model.forward_cached(ids[:, :9])
            streamed, _ = model.forward_cached(ids[:, 9:], cache=cache)
        assert torch.allclose(full[:, 9:], streamed["logits"], atol=1e-4)

    def test_output_norm_bounds_branch_scale(self):
        model = make_model(branch_norm="peri")
        layer = model.layers[0]
        hidden = torch.randn(2, 16, 32) * 5.0
        branch = layer.attn(layer.attn_norm(hidden), *model.rotary(16, device=hidden.device, dtype=hidden.dtype))
        normed = layer.attn_out_norm(branch)
        rms = normed.float().square().mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.05)
