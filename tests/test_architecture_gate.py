"""Head-specific sigmoid output gate on attention (trial-003): contract and numerics tests."""

import torch

from terminus.model import BitNetLlama, BitNetLlamaConfig

from tests.helpers import tiny_config


def make_model(**changes):
    torch.manual_seed(7)
    return BitNetLlama(tiny_config(**changes))


class TestAttentionGateConfig:
    def test_default_off(self):
        config = tiny_config()
        assert config.attention_gate is False
        model = BitNetLlama(config)
        for layer in model.layers:
            assert layer.attn.gate_proj is None

    def test_non_bool_rejected(self):
        for value in (1, "true", None):
            try:
                tiny_config(attention_gate=value)
            except ValueError:
                continue
            raise AssertionError(f"attention_gate={value!r} must be rejected")

    def test_gate_on_adds_bitlinear_projection(self):
        model = make_model(attention_gate=True)
        for layer in model.layers:
            gate = layer.attn.gate_proj
            assert gate is not None
            assert gate.weight.shape == (model.config.dim, model.config.dim)


class TestAttentionGateOff:
    def test_bit_identical_to_baseline(self):
        torch.manual_seed(11)
        baseline = BitNetLlama(tiny_config())
        torch.manual_seed(11)
        gated_off = BitNetLlama(tiny_config(attention_gate=False))
        assert set(dict(baseline.named_parameters())) == set(dict(gated_off.named_parameters()))
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        with torch.no_grad():
            reference = baseline(ids, targets)
            candidate = gated_off(ids, targets)
        assert torch.equal(reference["logits"], candidate["logits"])
        assert torch.equal(reference["loss"], candidate["loss"])


class TestAttentionGateForward:
    def test_shapes_and_loss(self):
        model = make_model(attention_gate=True)
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        out = model(ids, targets)
        assert out["logits"].shape == (2, 16, 32)
        assert torch.isfinite(out["loss"])

    def test_gate_changes_output(self):
        torch.manual_seed(11)
        baseline = BitNetLlama(tiny_config())
        torch.manual_seed(11)
        gated = BitNetLlama(tiny_config(attention_gate=True))
        ids = torch.randint(0, 32, (2, 16))
        with torch.no_grad():
            reference = baseline(ids)["logits"]
            candidate = gated(ids)["logits"]
        assert not torch.allclose(reference, candidate)

    def test_causality(self):
        model = make_model(attention_gate=True).eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            reference = model(ids)["logits"]
            changed = ids.clone()
            changed[0, 8:] = (changed[0, 8:] + 1) % 32
            altered = model(changed)["logits"]
        assert torch.allclose(reference[:, :8], altered[:, :8], atol=1e-5)

    def test_gradients_reach_gate_and_o_proj(self):
        model = make_model(attention_gate=True)
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        model(ids, targets)["loss"].backward()
        for layer in model.layers:
            assert layer.attn.gate_proj.weight.grad is not None
            assert torch.isfinite(layer.attn.gate_proj.weight.grad).all()
            assert layer.attn.gate_proj.weight.grad.abs().sum() > 0
            assert layer.attn.o_proj.weight.grad is not None
            assert torch.isfinite(layer.attn.o_proj.weight.grad).all()

    def test_gate_statistics_at_init(self):
        model = make_model(attention_gate=True).eval()
        for layer in model.layers:
            hidden = torch.randn(2, 16, 32)
            with torch.no_grad():
                gate = torch.sigmoid(layer.attn.gate_proj(hidden))
            assert ((gate > 0) & (gate < 1)).all()
            assert abs(gate.mean().item() - 0.5) < 0.05

    def test_cached_matches_full_forward(self):
        model = make_model(attention_gate=True).eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            full = model(ids)["logits"]
            _, cache = model.forward_cached(ids[:, :9])
            streamed, _ = model.forward_cached(ids[:, 9:], cache=cache)
        assert streamed["logits"].shape == (1, 7, 32)
        assert torch.allclose(full[:, 9:], streamed["logits"], atol=1e-4)


class TestProtocolEnvelope:
    def test_candidate_count_in_envelope(self):
        from research.protocol import count_candidate, load_candidate, policy, validate_candidate

        rules = policy()
        model = load_candidate(rules)
        validate_candidate(model, rules)
        assert model["attention_gate"] is True
        assert model["ffn_dim"] == 1664
        count = count_candidate(model, rules)
        assert rules["min_parameters"] <= count <= rules["max_parameters"]
        assert abs(count - 200.5e6) < 1e6


class TestMuonGrouping:
    def test_gate_proj_in_muon_group_and_full_coverage(self):
        from terminus.optim import OptimizerConfig, build_optimizer

        model = make_model(attention_gate=True)
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        gate_weights = {id(layer.attn.gate_proj.weight) for layer in model.layers}
        muon_ids, covered = set(), []
        for group in optimizer.param_groups:
            covered.extend(group["params"])
            if group.get("use_muon"):
                muon_ids.update(id(parameter) for parameter in group["params"])
        assert gate_weights <= muon_ids
        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        assert len(covered) == len({id(parameter) for parameter in covered}) == len(trainable)
