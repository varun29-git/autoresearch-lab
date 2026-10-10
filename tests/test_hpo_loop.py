import copy
import json
from pathlib import Path

import pytest

from hpo.loop import (DEFAULT_REFERENCE, candidate_config, load_reference,
                      rank, source_hashes, validation_curve)
from hpo.bridge import bridge_reference


def test_reference_is_fixed_stack4_on_speed_execution_path(tmp_path):
    reference = load_reference(DEFAULT_REFERENCE)
    model = reference["config"]["model"]
    assert model["n_layers"] == 25
    assert (model["dim"], model["ffn_dim"], model["n_heads"], model["n_kv_heads"]) == (768, 1920, 6, 2)
    assert (model["branch_norm"], model["value_residual"], model["fused_qkv"]) == ("peri", "learned", True)
    assert model["weight_mode"] == "dense"
    assert reference["config"]["optimizer"]["muon_lr_rule"] == "spectral_unclamped"
    assert reference["stages"][-2:] == [1250, 6358]
    assert reference["stages"][-2] * 393216 < 500_000_000
    assert reference["stages"][-1] * 393216 == 2_500_067_328
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(reference["config"]))
    from terminus.train import load_config

    load_config(config_path)


def test_candidate_only_changes_allowed_hyperparameters():
    reference = load_reference(DEFAULT_REFERENCE)
    reference_before = copy.deepcopy(reference)
    candidate = {"optimizer": {"muon_lr": 0.012, "adamw_lr": 0.001,
                                "weight_decay": 0.03},
                 "training": {"grad_clip": 0.5}}
    result = candidate_config(reference, candidate)
    assert result["optimizer"]["muon_lr"] == 0.012
    assert result["training"]["grad_clip"] == 0.5
    assert result["model"] == reference["config"]["model"]
    assert reference == reference_before
    for forbidden in ({"optimizer": {"muon_lr_rule": "sqrt_fan_in"}, "training": {}},
                      {"optimizer": {}, "training": {"warmup_tokens": 10}},
                      {"optimizer": {}, "training": {"gradient_accumulation_steps": 1}}):
        with pytest.raises(ValueError):
            candidate_config(reference, forbidden)


def test_curve_requires_matched_tokens_and_rank_requires_same_stage(tmp_path):
    reference = load_reference(DEFAULT_REFERENCE)
    events = tmp_path / "events.jsonl"
    record = {"event": "validation", "updates": 1250, "tokens_seen": 1250 * 393216,
              "languages": {"en": {"loss": 3.2, "tokens": 131072}}}
    events.write_text(json.dumps(record) + "\n")
    assert validation_curve(events, reference["stages"], 393216)[1250]["english_ce"] == 3.2
    record["tokens_seen"] -= 1
    events.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="token count"):
        validation_curve(events, reference["stages"], 393216)

    campaign_root = tmp_path / reference["campaign_id"]
    for name, loss, achieved in (("a", 3.0, 1250), ("b", 2.9, 6358), ("c", 2.8, 750)):
        trial = campaign_root / name
        trial.mkdir(parents=True)
        from hpo.loop import sha256

        result = {"trial": name, "reference_sha256": sha256(DEFAULT_REFERENCE),
                  "manifest_sha256": reference["manifest_sha256"],
                  "candidate": {"optimizer": {"muon_lr": 0.003}, "training": {}},
                  "curve": [{"updates": achieved, "tokens": achieved * 393216, "english_ce": loss}]}
        (trial / "result.json").write_text(json.dumps(result))
    class Args:
        reference = DEFAULT_REFERENCE
        results_root = tmp_path
        updates = 1250

    assert [row["trial"] for row in rank(Args)] == ["a"]


def test_source_fingerprint_covers_loop_and_runtime():
    hashes = source_hashes()
    assert "hpo/loop.py" in hashes
    assert "terminus/model.py" in hashes
    assert "terminus/wgrad_kernels.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_width_bridge_preserves_effective_batch_and_validation_set():
    base = load_reference(DEFAULT_REFERENCE)
    for width, microbatch, heads, kv_heads, ffn in ((1536, 4, 12, 4, 3840),
                                                    (2304, 2, 18, 6, 5760)):
        bridge = bridge_reference(base, width=width, microbatch=microbatch,
                                  manifest_sha256="a" * 64)
        model = bridge["config"]["model"]
        training = bridge["config"]["training"]
        assert (model["dim"], model["n_heads"], model["n_kv_heads"], model["ffn_dim"]) == (width, heads, kv_heads, ffn)
        assert (model["n_layers"], model["branch_norm"], model["value_residual"]) == (25, "peri", "learned")
        assert microbatch * 4096 * training["gradient_accumulation_steps"] == 393216
        assert microbatch * training["validation_batches_per_language"] == 32
        assert bridge["config"]["optimizer"] == base["config"]["optimizer"]
    with pytest.raises(ValueError):
        bridge_reference(base, width=2304, microbatch=3, manifest_sha256="a" * 64)
