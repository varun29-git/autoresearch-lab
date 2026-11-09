import copy
from dataclasses import asdict, replace
import json

import pytest
import torch

from production.amd.budget import forecast
from production.amd.build_config import build_config
from production.amd.measurement import summarize
from production.amd.recipe import CANDIDATES, SELECTED_DENSE_OPTIMIZER, SELECTED_QAT_OPTIMIZER, changes, model_for, validate_production
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
    assert optimizer == OptimizerConfig(**SELECTED_DENSE_OPTIMIZER)
    allowed = {"attention_backend", "linear_backend", "fp8_wgrad_splits", "quantize_activations",
               "fp8_backward", "fp8_quantizer", "fp8_cache_weights"}
    assert set(changes(model)) <= allowed
    with torch.device("meta"):
        instance = BitNetLlama(model)
    assert [i + 1 for i, layer in enumerate(instance.layers) if layer.attn.is_global_layer] == [6, 12, 18, 24]


@pytest.mark.parametrize("old_rate", [.002, .003])
def test_dense_selection_changes_only_embedding_rate_and_rejects_old_config(old_rate):
    assert SELECTED_DENSE_OPTIMIZER == SELECTED_OPTIMIZER | {"adamw_lr": .001}
    assert SELECTED_DENSE_OPTIMIZER["adamw_lr"] == .003 * 768 / 2304
    assert SELECTED_DENSE_OPTIMIZER["muon_lr"] == .017
    assert SELECTED_DENSE_OPTIMIZER["muon_lr_rule"] == "spectral_unclamped"
    assert SELECTED_QAT_OPTIMIZER["adamw_lr"] == .003
    with pytest.raises(ValueError, match="optimizer"):
        validate_production(model_for(), TrainConfig(device="cuda", dtype="bf16"),
                            OptimizerConfig(**(SELECTED_DENSE_OPTIMIZER | {"adamw_lr": old_rate})))


def test_no_silent_amd_precision_fallback():
    for options in ({"fp8_backward": "bf16"},
                    {"fp8_quantizer": "torch"}, {"weight_scale_layout": "rowwise"}):
        with pytest.raises(ValueError):
            replace(model_for(), **options)
    assert BitNetLlamaConfig(**SELECTED_MODEL).linear_backend == "deepgemm"
    with pytest.raises(RuntimeError, match="gfx942"):
        require_gfx942(torch.device("cpu"))
    for candidate in ("fp8_te", "bf16"):
        with pytest.raises(ValueError, match="split"):
            model_for(candidate, wgrad_splits=8)
    with pytest.raises(ValueError, match="QAT requires"):
        model_for("bf16", weight_mode="ternary")


@pytest.mark.parametrize("candidate", CANDIDATES[:-1])
def test_native_amd_qat_rejects_ignored_gradual_strength(candidate):
    with torch.device("meta"):
        model = BitNetLlama(model_for(candidate, weight_mode="ternary"))
    with pytest.raises(ValueError, match="quantization strength"):
        model.set_quantization_strength_(0.5)
    model.set_quantization_strength_(1.0)


@pytest.mark.parametrize("candidate", CANDIDATES[:-1])
def test_amd_qat_config_preserves_fp8_geometry_and_phase_schedule(candidate):
    manifest = TokenManifest(TokenDataConfig(batch_size=16, seq_len=4096, total_train_tokens=10**9), [])
    config = build_config(manifest=manifest, world_size=8, global_batch_tokens=2_097_152,
                          warmup_tokens=50_331_648, decay_tokens=100_663_296,
                          candidate=candidate, weight_mode="ternary", phase_start_tokens=503_316_480)
    model = BitNetLlamaConfig(**config["model"])
    training, optimizer = TrainConfig(**config["training"]), OptimizerConfig(**config["optimizer"])
    assert validate_production(model, training, optimizer) == candidate
    assert training.gradient_accumulation_steps == 4
    assert training.phase_start_tokens == 503_316_480
    assert optimizer == OptimizerConfig(**SELECTED_QAT_OPTIMIZER)
    assert count_model(model)["total"] == 1_446_478_126
    assert changes(model)["weight_mode"] == {"h100": "dense", "amd": "ternary"}
    with pytest.raises(ValueError, match="optimizer"):
        validate_production(model, training, OptimizerConfig(**SELECTED_OPTIMIZER))
    with pytest.raises(ValueError, match="phase_start"):
        validate_production(model, replace(training, phase_start_tokens=None), optimizer)


def test_amd_qat_schedule_must_fit_remaining_manifest_budget():
    manifest = TokenManifest(TokenDataConfig(batch_size=16, seq_len=4096, total_train_tokens=10**9), [])
    common = dict(manifest=manifest, world_size=8, global_batch_tokens=2_097_152,
                  weight_mode="ternary", phase_start_tokens=838_860_800)
    with pytest.raises(ValueError, match="remaining phase"):
        build_config(**common, warmup_tokens=100_000_000, decay_tokens=100_000_000)
    with pytest.raises(ValueError, match="phase_start"):
        build_config(**(common | {"phase_start_tokens": None}),
                     warmup_tokens=1, decay_tokens=2)
    with pytest.raises(ValueError, match="align"):
        build_config(**(common | {"phase_start_tokens": 838_860_801}),
                     warmup_tokens=1, decay_tokens=2)


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


def test_resumed_benchmark_checks_checkpoint_cursor_not_phase_start():
    from production.amd.measurement import validate_window

    # The phase began at 200 tokens, but the resumed checkpoint is at 900.
    # Two 100-token updates fit the phase budget but cannot fit this resume.
    with pytest.raises(ValueError, match="remaining tokens"):
        validate_window(total_tokens=1000, tokens_seen=900, batch_tokens=100,
                        warmup_updates=1, measured_updates=1)
    validate_window(total_tokens=1000, tokens_seen=800, batch_tokens=100,
                    warmup_updates=1, measured_updates=1)


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
    report.update({k: [True] for k in ("attention", "document_attention", "model_numerics", "packed_model_numerics", "optimized_kernels", "packed_learned_qkv", "optimizer", "fp8")})
    report["fp8"] = {"surrogate_gradients": {"cases": [True]}}
    report["report_sha256"] = fingerprint(report)
    path.write_text(json.dumps(report))
    validate_report(path, model_for(), torch.device("cpu"))
    missing = copy.deepcopy(report)
    missing.pop("packed_model_numerics")
    missing["report_sha256"] = fingerprint({k: v for k, v in missing.items() if k != "report_sha256"})
    path.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="evidence"):
        validate_report(path, model_for(), torch.device("cpu"))
    report["fp8"] = []
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_report(path, model_for(), torch.device("cpu"))


def test_amd_qat_receipt_requires_mode_specific_native_evidence(tmp_path, monkeypatch):
    from production.amd.preflight import validate_report

    monkeypatch.setattr("production.amd.preflight.identity", lambda *a: {"test": "identity"})
    model = model_for(weight_mode="ternary")
    report = {"kind": "ci1_amd_preflight", "schema": 1, "status": "passed",
              "identity": {"test": "identity"},
              **{k: [True] for k in ("attention", "document_attention", "model_numerics", "packed_model_numerics", "optimized_kernels",
                                      "packed_learned_qkv", "optimizer")},
              "fp8": {"weight_mode": "ternary", "fp32_wgrad": True,
                      "weight_cache_refresh": True, "surrogate_gradients": {"cases": [True]}},
              "ternary": {"weight_mode": "ternary", "fp8_operand_gate": True,
                          "reduced_model_numerics": True, "optimizer": True}}
    path = tmp_path / "qat-preflight.json"
    for key in ("weight_mode", "fp32_wgrad", "weight_cache_refresh"):
        changed = copy.deepcopy(report)
        changed["fp8"][key] = "dense" if key == "weight_mode" else False
        changed["report_sha256"] = fingerprint(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="ternary"):
            validate_report(path, model, torch.device("cpu"))
    report["report_sha256"] = fingerprint(report)
    path.write_text(json.dumps(report))
    validate_report(path, model, torch.device("cpu"))


def test_fp8_producer_candidate_is_explicit_and_uses_qualified_dense_rates():
    assert not model_for().fused_fp8_producers
    for mode in ('dense','ternary'):
        model=model_for('fp8_te',weight_mode=mode,fused_fp8_producers=True)
        assert model.fused_fp8_producers
    with pytest.raises(ValueError):model_for('bf16',fused_fp8_producers=True)
    unpacked = replace(model_for(), fuse_projections=False, fused_qkv=False, fused_mlp=False)
    with pytest.raises(ValueError, match="packed projections"):
        replace(unpacked, fused_fp8_producers=True)
