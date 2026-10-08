"""The H100 ablation harness must not leak a candidate into its control."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from optimizations.h100_stack_benchmark import _patch_fused, parse_variants, run_benchmark
from terminus import loss as loss_module
from terminus import model as model_module
from terminus.logical_muon import LogicalMuonWithAuxAdam
from tests.helpers import tiny_config


def test_variant_patches_are_isolated_and_restored():
    original = (model_module.BitNetAttention.forward, model_module.BitNetBlock.forward,
                loss_module.streaming_cross_entropy_sum, LogicalMuonWithAuxAdam.step)
    with _patch_fused(components={"value_mix"}):
        assert model_module.BitNetAttention.forward is not original[0]
        assert (model_module.BitNetBlock.forward, loss_module.streaming_cross_entropy_sum,
                LogicalMuonWithAuxAdam.step) == original[1:]
    assert (model_module.BitNetAttention.forward, model_module.BitNetBlock.forward,
            loss_module.streaming_cross_entropy_sum, LogicalMuonWithAuxAdam.step) == original
    with _patch_fused(components={"output_norm"}):
        assert model_module.BitNetBlock.forward.__name__ == "output_norm_block_forward"
    with _patch_fused(components={"block_boundary"}):
        assert model_module.BitNetBlock.forward.__name__ == "boundary_block_forward"
    with _patch_fused(components={"output_norm", "block_boundary"}):
        assert model_module.BitNetBlock.forward.__name__ == "fused_block_forward"
    assert model_module.BitNetBlock.forward is original[1]


def test_invalid_variant_selection_fails_before_timing():
    with pytest.raises(ValueError, match="baseline"):
        parse_variants(("value_mix",))
    with pytest.raises(ValueError, match="unknown"):
        parse_variants(("baseline", "no_such_variant"))
    with pytest.raises(ValueError, match="duplicates"):
        parse_variants(("baseline", "baseline"))
    with pytest.raises(ValueError, match="graph"):
        run_benchmark(batch_size=1, warmup_steps=0, measured_steps=1, skip_gates=True,
                      variants=("baseline", "graphs"), event_profile=True)


def test_benchmark_runs_each_variant_from_a_clean_baseline():
    original = (model_module.BitNetAttention.forward, model_module.BitNetBlock.forward,
                loss_module.streaming_cross_entropy_sum, LogicalMuonWithAuxAdam.step)
    seen = []

    def fake_benchmark(path, **kwargs):
        config = json.loads(Path(path).read_text())
        state = (model_module.BitNetAttention.forward is not original[0],
                 model_module.BitNetBlock.forward is not original[1],
                 loss_module.streaming_cross_entropy_sum is not original[2],
                 LogicalMuonWithAuxAdam.step is not original[3])
        seen.append((state, config["training"].get("cuda_graphs", False), kwargs["profile_dir"]))
        return {"tokens_per_second": 10.0, "median_step_seconds": 1.0,
                "peak_allocated_bytes": 100, "peak_reserved_bytes": 120,
                "step_seconds": [1.0], "losses": [2.0], "gradient_norms": [1.0],
                "profile_dir": None, "event_profile": None, "runtime": {}, "physics": None}

    with patch("terminus.benchmark.training_benchmark", side_effect=fake_benchmark):
        report = run_benchmark(batch_size=1, warmup_steps=0, measured_steps=1, skip_gates=True,
                               variants=("baseline", "value_mix", "head_loss", "batched_muon", "graphs", "fused"))
    assert [state for state, _, _ in seen] == [
        (False, False, False, False), (True, False, False, False),
        (False, False, True, False), (False, False, False, True),
        (False, False, False, False), (True, True, True, True)]
    assert [graphs for _, graphs, _ in seen] == [False, False, False, False, True, False]
    assert report["variant_order"] == ("baseline", "value_mix", "head_loss", "batched_muon", "graphs", "fused")
    assert (model_module.BitNetAttention.forward, model_module.BitNetBlock.forward,
            loss_module.streaming_cross_entropy_sum, LogicalMuonWithAuxAdam.step) == original


def test_single_stack_fusions_match_reference_on_cpu():
    torch.manual_seed(43)
    model = model_module.BitNetLlama(tiny_config(branch_norm="peri", value_residual="learned"))
    ids = torch.randint(0, 32, (2, 16))
    targets = torch.randint(0, 32, (2, 16))
    baseline_loss = model(ids, targets, return_logits=False)["loss"]
    baseline_loss.backward()
    baseline_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    for component in ("output_norm", "block_boundary", "value_mix", "head_loss"):
        model.zero_grad(set_to_none=True)
        with _patch_fused(components={component}):
            candidate_loss = model(ids, targets, return_logits=False)["loss"]
            candidate_loss.backward()
        torch.testing.assert_close(candidate_loss, baseline_loss, rtol=1e-4, atol=1e-5)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, baseline_grads[name], rtol=1e-4, atol=1e-5,
                                       msg=f"{component}: {name}")
