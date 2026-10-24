import copy
from dataclasses import asdict, replace
import json

import pytest
import torch

from production.amd.budget import forecast
from production.amd.build_config import build_config
from production.amd.measurement import summarize
from production.amd.recipe import CANDIDATES, changes, model_for, validate_production
from production.amd.runtime import require_gfx942
from production.amd.staging import create, verify
from production.recipe import SELECTED_MODEL, SELECTED_OPTIMIZER
from terminus.data import TokenManifest, TokenDataConfig
from terminus.integrity import fingerprint
from terminus.model import BitNetLlama, BitNetLlamaConfig
from terminus.optim import OptimizerConfig
from terminus.preflight import count_model
from terminus.train import TrainConfig
from tests.helpers import fixture


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_candidate_preserves_geometry_optimizer_and_batch(candidate):
    manifest = TokenManifest(TokenDataConfig(batch_size=16, seq_len=4096, total_train_tokens=10**9), [])
    config = build_config(manifest=manifest, world_size=8, global_batch_tokens=4_194_304,
                          warmup_tokens=20_000_000, decay_tokens=100_000_000, candidate=candidate)
    model = BitNetLlamaConfig(**config["model"])
    training, optimizer = TrainConfig(**config["training"]), OptimizerConfig(**config["optimizer"])
    assert validate_production(model, training, optimizer) == candidate
    assert training.gradient_accumulation_steps == 8
    assert count_model(model)["total"] == 1_446_478_126
    assert optimizer == OptimizerConfig(**SELECTED_OPTIMIZER)
    allowed = {"attention_backend", "linear_backend", "fp8_wgrad_splits", "quantize_activations",
               "fp8_backward", "fp8_quantizer", "fp8_cache_weights"}
    assert set(changes(model)) <= allowed
    with torch.device("meta"):
        instance = BitNetLlama(model)
    assert [i + 1 for i, layer in enumerate(instance.layers) if layer.attn.is_global_layer] == [6, 12, 18, 24]


def test_no_silent_amd_qat_or_precision_fallback():
    for options in ({"weight_mode": "ternary"}, {"fp8_backward": "bf16"}, {"fused_fp8_producers": True},
                    {"fp8_quantizer": "torch"}, {"weight_scale_layout": "rowwise"}):
        with pytest.raises(ValueError):
            replace(model_for(), **options)
    assert BitNetLlamaConfig(**SELECTED_MODEL).linear_backend == "deepgemm"
    with pytest.raises(RuntimeError, match="gfx942"):
        require_gfx942(torch.device("cpu"))
    for candidate in ("fp8_te", "bf16"):
        with pytest.raises(ValueError, match="split"):
            model_for(candidate, wgrad_splits=8)


def test_mismatched_global_batch_rejected():
    manifest = TokenManifest(TokenDataConfig(batch_size=24, seq_len=4096, total_train_tokens=10**9), [])
    with pytest.raises(ValueError, match="multiple"):
        build_config(manifest=manifest, world_size=8, global_batch_tokens=4_194_304,
                     warmup_tokens=1, decay_tokens=100)
    with pytest.raises(ValueError, match="ddp_bucket"):
        TrainConfig(ddp_bucket_cap_mb=0)


def test_budget_counts_all_gpu_hours_and_phase_costs():
    gross = forecast(dense_rate=50_000)
    assert gross["allocation_wall_days"] == 30
    assert gross["projected_dense_tokens_after_qat"] == 1_036_800_000_000
    assert gross["gross_required_average_rate_per_gpu"] == pytest.approx(72_337.96296296296)
    assert not gross["fits_assumptions"]
    mixed = forecast(dense_rate=80_000, qat_tokens=100_000_000_000, qat_rate=40_000,
                     reserve_gpu_hours=320, duty_cycle=.9)
    assert mixed["gpu_hours_required"] == pytest.approx(320 + (1.4e12 / 80_000 + 1e11 / 40_000) / 3600 / .9)
    assert forecast(dense_rate=50_000, gpus=32)["gpu_hours_required"] == gross["gpu_hours_required"]
    for kwargs in ({"dense_rate": float("nan")}, {"reserve_gpu_hours": 5760},
                   {"qat_tokens": 10}, {"duty_cycle": 1.1}):
        with pytest.raises(ValueError):
            forecast(**({"dense_rate": 50_000} | kwargs))


def test_slowest_rank_rate_does_not_multiply_global_tokens_twice():
    ranks = [{"rank": r, "records": [{"seconds": t, "global_tokens": 800, "loss": 3.}
                                     for t in ([1., 2.] if r == 0 else [2., 1.])],
              "peak_allocated_bytes": 100, "peak_reserved_bytes": 120} for r in range(2)]
    result = summarize(ranks, world_size=2)
    assert result["tokens_per_second_total"] == 400
    assert result["tokens_per_second_per_gpu"] == 200
    bad = copy.deepcopy(ranks)
    bad[1]["records"][0]["global_tokens"] = 801
    with pytest.raises(ValueError, match="token"):
        summarize(bad, world_size=2)
    with pytest.raises(ValueError, match="missing"):
        summarize(ranks[:1], world_size=2)


def test_staging_detects_changes_and_writable_files(tmp_path):
    fixture(tmp_path, examples=8)
    manifest = TokenManifest.load(tmp_path / "manifest.json")
    config = json.loads((tmp_path / "config.json").read_text())
    model = BitNetLlamaConfig(**config["model"])
    with pytest.raises(ValueError, match="read-only"):
        create(manifest, tmp_path, model, development_mode=True)
    for spec in manifest.shards:
        (tmp_path / spec.path).chmod(0o444)
    receipt = create(manifest, tmp_path, model, development_mode=True)
    assert verify(receipt, manifest, tmp_path, model, development_mode=True)["validated_shards"] == len(manifest.shards)
    with pytest.raises(ValueError, match="stale"):
        verify(receipt, manifest, tmp_path, replace(model, rope_theta=9999.), development_mode=True)
    shard = tmp_path / manifest.shards[0].path
    shard.chmod(0o644)
    original = shard.read_bytes()
    shard.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    shard.chmod(0o444)
    with pytest.raises(ValueError, match="stale"):
        verify(receipt, manifest, tmp_path, model, development_mode=True)


def test_amd_receipt_cannot_use_h100_or_incomplete_evidence(tmp_path, monkeypatch):
    from production.amd.preflight import validate_report

    monkeypatch.setattr("production.amd.preflight.identity", lambda *a: {"test": "identity"})
    path = tmp_path / "preflight.json"
    report = {"kind": "ci1_amd_preflight", "schema": 1, "status": "passed", "identity": {"test": "identity"}}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evidence"):
        validate_report(path, model_for(), torch.device("cpu"))
    report.update({k: [True] for k in ("attention", "model_numerics", "optimized_kernels", "packed_learned_qkv", "optimizer", "fp8")})
    report["report_sha256"] = fingerprint(report)
    path.write_text(json.dumps(report))
    validate_report(path, model_for(), torch.device("cpu"))
    report["fp8"] = []
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_report(path, model_for(), torch.device("cpu"))
