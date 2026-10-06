"""Stack candidate (trial-004): Peri-LN branch output norms + value residual combined."""

import torch

from terminus.model import BitNetLlama, RMSNorm

from tests.helpers import tiny_config


def make_model(**changes):
    torch.manual_seed(7)
    return BitNetLlama(tiny_config(branch_norm="peri", value_residual="learned", **changes))


class TestStack:
    def test_both_mechanisms_present(self):
        model = make_model()
        for index, layer in enumerate(model.layers):
            assert isinstance(layer.attn_out_norm, RMSNorm)
            assert isinstance(layer.ffn_out_norm, RMSNorm)
            if index >= 1:
                assert layer.attn.value_lambda.shape == (2,)

    def test_shapes_and_loss(self):
        model = make_model()
        ids = torch.randint(0, 32, (2, 16))
        out = model(ids, torch.randint(0, 32, (2, 16)))
        assert out["logits"].shape == (2, 16, 32)
        assert torch.isfinite(out["loss"])

    def test_causality(self):
        model = make_model().eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            reference = model(ids)["logits"]
            changed = ids.clone()
            changed[0, 8:] = (changed[0, 8:] + 1) % 32
            altered = model(changed)["logits"]
        assert torch.allclose(reference[:, :8], altered[:, :8], atol=1e-5)

    def test_gradients_reach_all_new_parameters(self):
        model = make_model()
        ids = torch.randint(0, 32, (2, 16))
        model(ids, torch.randint(0, 32, (2, 16)))["loss"].backward()
        for index, layer in enumerate(model.layers):
            assert layer.attn_out_norm.weight.grad is not None
            assert layer.ffn_out_norm.weight.grad is not None
            if index >= 1:
                assert layer.attn.value_lambda.grad is not None
                assert torch.isfinite(layer.attn.value_lambda.grad).all()
        assert model.layers[0].attn.v_proj.weight.grad is not None  # multi-path V1 gradient

    def test_cached_matches_full_forward(self):
        model = make_model().eval()
        ids = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            full = model(ids)["logits"]
            _, cache = model.forward_cached(ids[:, :9])
            streamed, _ = model.forward_cached(ids[:, 9:], cache=cache)
        assert torch.allclose(full[:, 9:], streamed["logits"], atol=1e-4)

    def test_gradient_checkpointing_parity(self):
        from dataclasses import replace

        torch.manual_seed(23)
        retained = make_model()
        torch.manual_seed(23)
        checkpointed = BitNetLlama(replace(tiny_config(branch_norm="peri", value_residual="learned"),
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
            assert torch.allclose(parameter.grad, checkpointed_parameter.grad, atol=1e-5), name

    def test_training_loop_with_stability_monitor(self):
        # Regression for the trial-002 screen crash, with both mechanisms on.
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
            config["model"]["branch_norm"] = "peri"
            config["optimizer"]["kind"] = "muon"
            config["training"]["stability_log_interval"] = 1
            config_path.write_text(json.dumps(config))
            run_training(config_path, Path(directory) / "manifest.json", directory,
                         Path(directory) / "out")
            stability = (Path(directory) / "out" / "stability.jsonl").read_text().strip()
            assert stability, "stability monitor must have recorded probes"
            assert "attention_output" in stability
