"""Value residual learning (trial-002, ResFormer V1 skip): contract and numerics tests."""

import torch

from terminus.model import BitNetLlama

from tests.helpers import tiny_config


def make_model(**changes):
    torch.manual_seed(7)
    return BitNetLlama(tiny_config(**changes))


class TestValueResidualConfig:
    def test_default_off(self):
        config = tiny_config()
        assert config.value_residual == "none"
        model = BitNetLlama(config)
        for layer in model.layers:
            assert not hasattr(layer.attn, "value_lambda")
            hidden = torch.randn(1, 4, config.dim)
            output = layer.attn(layer.attn_norm(hidden),
                                *model.rotary(4, device=hidden.device, dtype=hidden.dtype))
            assert isinstance(output, torch.Tensor)  # no tuple when disabled

    def test_invalid_mode_rejected(self):
        try:
            tiny_config(value_residual="fixed")
        except ValueError:
            return
        raise AssertionError("unknown value_residual must be rejected")

    def test_learned_adds_lambda_from_layer_one(self):
        model = make_model(value_residual="learned")
        assert not hasattr(model.layers[0].attn, "value_lambda")
        names = dict(model.named_parameters())
        for index, layer in enumerate(model.layers):
            if index == 0:
                continue
            assert layer.attn.value_lambda.shape == (2,)
            assert torch.allclose(layer.attn.value_lambda.detach(), torch.tensor([0.5, 0.5]))
            assert f"layers.{index}.attn.value_lambda" in names

    def test_none_matches_baseline_bit_identical(self):
        torch.manual_seed(11)
        reference = BitNetLlama(tiny_config())
        torch.manual_seed(11)
        explicit = BitNetLlama(tiny_config(value_residual="none"))
        assert list(reference.state_dict()) == list(explicit.state_dict())
        ids = torch.randint(0, 32, (2, 16))
        with torch.no_grad():
            assert torch.equal(reference(ids)["logits"], explicit(ids)["logits"])


class TestValueResidualForward:
    def test_shapes_and_loss(self):
        model = make_model(value_residual="learned")
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        out = model(ids, targets)
        assert out["logits"].shape == (2, 16, 32)
        assert torch.isfinite(out["loss"])

    def test_causality(self):
        model = make_model(value_residual="learned").eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            reference = model(ids)["logits"]
            changed = ids.clone()
            changed[0, 8:] = (changed[0, 8:] + 1) % 32
            altered = model(changed)["logits"]
        assert torch.allclose(reference[:, :8], altered[:, :8], atol=1e-5)

    def test_gradients_reach_lambda_and_both_value_projections(self):
        model = make_model(value_residual="learned")
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        model(ids, targets)["loss"].backward()
        lambda_grad = model.layers[1].attn.value_lambda.grad
        assert lambda_grad is not None and torch.isfinite(lambda_grad).all()
        assert lambda_grad.abs().sum() > 0
        # Multi-path: every layer's loss must reach layer 0's v_proj (no detach).
        first = model.layers[0].attn.v_proj.weight.grad
        later = model.layers[1].attn.v_proj.weight.grad
        assert first is not None and first.abs().sum() > 0
        assert later is not None and later.abs().sum() > 0

    def test_lambda_zero_one_recovers_baseline(self):
        model = make_model(value_residual="learned")
        with torch.no_grad():
            for layer in model.layers[1:]:
                layer.attn.value_lambda.copy_(torch.tensor([0.0, 1.0]))
        baseline = make_model()
        ids = torch.randint(0, 32, (2, 16))
        with torch.no_grad():
            assert torch.equal(model(ids)["logits"], baseline(ids)["logits"])

    def test_cached_matches_full_forward(self):
        model = make_model(value_residual="learned").eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            full = model(ids)["logits"]
            _, cache = model.forward_cached(ids[:, :9])
            streamed, _ = model.forward_cached(ids[:, 9:], cache=cache)
        assert torch.allclose(full[:, 9:], streamed["logits"], atol=1e-4)

    def test_two_step_cached_decode_prefill9_decode7(self):
        model = make_model(value_residual="learned").eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            full = model(ids)["logits"]
            prefill, cache = model.forward_cached(ids[:, :9])
            decode, cache = model.forward_cached(ids[:, 9:], cache=cache)
        assert torch.allclose(full[:, :9], prefill["logits"], atol=1e-4)
        assert torch.allclose(full[:, 9:], decode["logits"], atol=1e-4)
        # Rolling V_1 buffer: capacity max_seq_len, raw layer-0 values, exact tail alignment.
        assert cache.value_residual_buffer.shape == (1, 2, 16, 8)
        assert cache.position == 16

    def test_cached_without_value_residual_keeps_no_buffer(self):
        model = make_model().eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            _, cache = model.forward_cached(ids[:, :9])
        assert cache.value_residual_buffer is None

    def test_gradient_checkpointing_parity(self):
        # Mirrors the native preflight gate: checkpointed and retained executions
        # of the same arithmetic must agree on loss and every parameter gradient.
        # Layer 0 runs uncheckpointed by design so V1 keeps its gradient path.
        from dataclasses import replace

        torch.manual_seed(23)
        retained = BitNetLlama(tiny_config(value_residual="learned"))
        torch.manual_seed(23)
        checkpointed = BitNetLlama(replace(tiny_config(value_residual="learned"),
                                           gradient_checkpointing=True))
        checkpointed.load_state_dict(retained.state_dict())
        ids = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))
        retained_loss = retained(ids, targets)["loss"]
        checkpointed.train()
        checkpointed_loss = checkpointed(ids, targets)["loss"]
        assert torch.allclose(retained_loss, checkpointed_loss, atol=1e-6)
        retained_loss.backward()
        checkpointed_loss.backward()
        for (name, parameter), (_, checkpointed_parameter) in zip(
                retained.named_parameters(), checkpointed.named_parameters()):
            assert parameter.grad is not None, name
            assert checkpointed_parameter.grad is not None, name
            assert torch.allclose(parameter.grad, checkpointed_parameter.grad, atol=1e-5), name
        # The multi-path gradient must reach layer 0's value projection.
        assert retained.layers[0].attn.v_proj.weight.grad.abs().sum() > 0

    def test_value_residual_handoff_releases_activation_after_backward(self):
        import gc
        import weakref

        for checkpointing in (False, True):
            model = make_model(value_residual="learned", n_layers=4,
                               fuse_projections=True, gradient_checkpointing=checkpointing)
            references = []
            hook = model.layers[0].attn.register_forward_hook(
                lambda module, inputs, output: references.append(weakref.ref(module.values_used)))
            ids = torch.randint(0, 32, (2, 16))
            output = model(ids, ids)
            output["loss"].backward()
            hook.remove()
            del output
            gc.collect()
            assert references and all(reference() is None for reference in references)
            assert all(getattr(layer.attn, "values_used", None) is None for layer in model.layers)
            assert model.layers[0].attn.qkv_proj.weight.grad.abs().sum() > 0

    def test_cached_values_belong_to_cache_and_not_model(self):
        model = make_model(value_residual="learned", n_layers=4, fuse_projections=True).eval()
        ids = torch.randint(0, 32, (1, 16))
        _, cache = model.forward_cached(ids[:, :9])
        result, cache = model.forward_cached(ids[:, 9:], cache=cache)
        assert cache.value_residual_buffer.shape == (1, 2, 16, 8)
        assert all(getattr(layer.attn, "values_used", None) is None for layer in model.layers)
        assert torch.allclose(model(ids)["logits"][:, 9:], result["logits"], atol=1e-4)

    def test_later_layers_share_one_contiguous_gradient_carrying_v1(self):
        model = make_model(value_residual="learned", n_layers=4, fuse_projections=True)
        sources = []

        def capture_source(_module, _args, kwargs):
            sources.append(kwargs["value_source"])

        hooks = [layer.attn.register_forward_pre_hook(capture_source, with_kwargs=True)
                 for layer in model.layers[1:]]
        ids = torch.randint(0, 32, (2, 16))
        model(ids, ids)["loss"].backward()
        for hook in hooks:
            hook.remove()
        assert len(sources) == 3
        assert all(source.is_contiguous() and source.requires_grad for source in sources)
        assert len({source.contiguous().data_ptr() for source in sources}) == 1
        kv = model.config.n_kv_heads * model.config.head_dim
        assert model.layers[0].attn.qkv_proj.weight.grad[-kv:].abs().sum() > 0

    def test_training_loop_with_stability_monitor(self):
        # Regression for the trial-002 screen crash: the protected StabilityMonitor
        # hooks each attention module's output and requires a plain tensor return.
        import json
        import tempfile
        from pathlib import Path

        from terminus.train import run_training

        from tests.helpers import fixture

        with tempfile.TemporaryDirectory() as directory:
            fixture(directory)
            config_path = Path(directory) / "config.json"
            config = json.loads(config_path.read_text())
            config["model"]["value_residual"] = "learned"
            config["optimizer"]["kind"] = "muon"
            config["training"]["stability_log_interval"] = 1
            config_path.write_text(json.dumps(config))
            run_training(config_path, Path(directory) / "manifest.json", directory,
                         Path(directory) / "out")
            stability = (Path(directory) / "out" / "stability.jsonl").read_text().strip()
            assert stability, "stability monitor must have recorded probes"
            assert "attention_output" in stability
