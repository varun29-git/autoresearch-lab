import json
from dataclasses import replace

import pytest

from production.build_dense_config import build_config
from production.inspect_run import inspect_run
from terminus.data import TokenDataConfig, TokenManifest
from terminus.train import load_config


def inputs(tmp_path, *, total_tokens=10_000_123):
    manifest = TokenManifest(TokenDataConfig(batch_size=4, total_train_tokens=total_tokens), [])
    config = build_config(manifest=manifest, world_size=32, global_batch_tokens=4_194_304,
                          warmup_tokens=1_000_000, decay_tokens=2_000_000)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return (*load_config(path), manifest)


def test_inspector_accounts_for_partial_update_without_certifying_data_or_gpu(tmp_path):
    result = inspect_run(*inputs(tmp_path), world_size=32, expected_global_batch_tokens=4_194_304)
    assert result["parameters"] == 1_500_456_496
    assert result["parallelism"]["accumulation_steps"] == 8
    assert result["parallelism"]["tokens_per_rank_per_full_update"] == 131_072
    assert result["parallelism"]["ring_allreduce_bytes_per_rank_per_full_update"] == 11_628_537_844
    schedule = result["schedule"]
    assert schedule["total_updates"] == 3
    assert schedule["full_updates"] == 2
    assert schedule["usable_train_tokens"] == 9_998_336
    assert schedule["partial_final_update_tokens"] == 1_609_728
    assert schedule["unused_tail_tokens"] == 1787
    assert result["data"]["status"] == "not_checked"
    assert result["unverified"]


def test_inspector_rejects_wrong_world_size_even_with_valid_config(tmp_path):
    with pytest.raises(ValueError, match="global batch"):
        inspect_run(*inputs(tmp_path), world_size=16, expected_global_batch_tokens=4_194_304)


@pytest.mark.parametrize("section,changes,message", [
    (0, {"value_residual": "none"}, "model differs"),
    (1, {"development_mode": True}, "training semantics"),
    (1, {"decay_tokens": 10_000_000}, "within budget"),
    (1, {"cuda_graphs": True}, "eager execution"),
    (2, {"muon_lr_rule": "sqrt_fan_in"}, "optimizer differs"),
])
def test_inspector_rejects_baseline_and_schedule_drift(tmp_path, section, changes, message):
    args = list(inputs(tmp_path))
    args[section] = replace(args[section], **changes)
    with pytest.raises(ValueError, match=message):
        inspect_run(*args, world_size=32, expected_global_batch_tokens=4_194_304)


def test_requested_data_verification_cannot_pass_an_empty_manifest(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        inspect_run(*inputs(tmp_path), world_size=32, expected_global_batch_tokens=4_194_304,
                    data_root=tmp_path)
