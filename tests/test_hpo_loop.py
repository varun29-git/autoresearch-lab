import copy
import json
from pathlib import Path

import pytest

from hpo.loop import (DEFAULT_CANDIDATE, DEFAULT_REFERENCE, candidate_config, load_reference,
                      rank, run, source_hashes, validation_curve)
from hpo.bridge import bridge_reference


def test_reference_is_fixed_stack4_on_speed_execution_path(tmp_path):
    reference = load_reference(DEFAULT_REFERENCE)
    model = reference["config"]["model"]
    assert model["n_layers"] == 25
    assert (model["dim"], model["ffn_dim"], model["n_heads"], model["n_kv_heads"]) == (768, 1920, 6, 2)
    assert (model["branch_norm"], model["value_residual"], model["fused_qkv"]) == ("peri", "learned", True)
    assert model["weight_mode"] == "dense"
    assert reference["config"]["optimizer"]["muon_lr_rule"] == "spectral_unclamped"
    assert reference["stages"][-1] == 1272
    assert reference["stages"][-1] * 393216 == 500_170_752
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(reference["config"]))
    from terminus.train import load_config

    load_config(config_path)
    from terminus.train import schedule_scale

    training = reference["config"]["training"]
    assert schedule_scale(500_170_752, 2_500_067_328,
                          training["warmup_tokens"], training["final_lr_fraction"],
                          schedule=training["schedule"], decay_tokens=training["decay_tokens"],
                          decay_style=training["decay_style"]) == 1.0


def test_candidate_only_changes_allowed_hyperparameters():
    reference = load_reference(DEFAULT_REFERENCE)
    reference_before = copy.deepcopy(reference)
    candidate = {"optimizer": {"muon_lr": 0.012, "adamw_lr": 0.001,
                                "muon_o_lr": 0.006, "muon_down_lr": 0.006,
                                "peri_norm_lr": 0.0003,
                                "embedding_weight_decay": 0.0,
                                "weight_decay": 0.03},
                 "training": {"grad_clip": 0.5}}
    result = candidate_config(reference, candidate)
    assert result["optimizer"]["muon_lr"] == 0.012
    assert result["optimizer"]["muon_o_lr"] == 0.006
    assert result["training"]["grad_clip"] == 0.5
    assert result["model"] == reference["config"]["model"]
    assert reference == reference_before
    with pytest.raises(ValueError, match="explicitly choose"):
        candidate_config(reference, {"optimizer": {"muon_lr": 0.003}, "training": {}})
    with pytest.raises(ValueError, match="explicitly choose"):
        candidate_config(reference, json.loads(DEFAULT_CANDIDATE.read_text()))
    for forbidden in ({"optimizer": {"muon_lr_rule": "sqrt_fan_in"}, "training": {}},
                      {"optimizer": {}, "training": {"warmup_tokens": 10}},
                      {"optimizer": {}, "training": {"gradient_accumulation_steps": 1}}):
        with pytest.raises(ValueError):
            candidate_config(reference, forbidden)


def test_muon_role_rates_are_explicit_and_optional():
    from terminus.optim import OptimizerConfig, muon_role_lr

    config = OptimizerConfig(muon_lr=0.003, muon_qkv_lr=0.002,
                             muon_o_lr=0.005, muon_gate_up_lr=0.004,
                             muon_down_lr=0.006, peri_norm_lr=0.0003,
                             embedding_weight_decay=0.0)
    for suffix, expected in (("attn.qkv_proj.weight", 0.002),
                             ("attn.o_proj.weight", 0.005),
                             ("ffn.gate_up_proj.weight", 0.004),
                             ("ffn.down_proj.weight", 0.006)):
        assert muon_role_lr("layers.0." + suffix, config) == expected
    assert muon_role_lr("embed_rot_in.weight", config) == 0.003
    with pytest.raises(ValueError, match="muon_o_lr"):
        OptimizerConfig(muon_o_lr=0.0)


def test_auxiliary_optimizer_roles_get_distinct_rates_without_dropping_parameters():
    from terminus.model import BitNetLlama
    from terminus.optim import OptimizerConfig, build_optimizer
    from tests.helpers import tiny_config

    model = BitNetLlama(tiny_config(branch_norm="peri", value_residual="learned",
                                    qk_norm_mode="learned_rms"))
    optimizer = build_optimizer(model, OptimizerConfig(
        kind="adamw", adamw_lr=0.001, norm_lr=0.0002,
        peri_norm_lr=0.0003, final_norm_lr=0.0004,
        value_mix_lr=0.0005, embedding_weight_decay=0.0))
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["tied_embedding"]["lr"] == 0.001
    assert groups["tied_embedding"]["weight_decay"] == 0.0
    assert groups["input_norm"]["lr"] == 0.0002
    assert groups["peri_norm"]["lr"] == 0.0003
    assert groups["final_norm"]["lr"] == 0.0004
    assert groups["value_mix"]["lr"] == 0.0005
    assert groups["qk_norm"]["lr"] == 0.0001
    grouped = [id(parameter) for group in groups.values() for parameter in group["params"]]
    assert len(grouped) == len(set(grouped)) == sum(parameter.requires_grad for parameter in model.parameters())


def test_curve_requires_matched_tokens_and_rank_requires_same_stage(tmp_path):
    reference = load_reference(DEFAULT_REFERENCE)
    events = tmp_path / "events.jsonl"
    record = {"event": "validation", "updates": 1272, "tokens_seen": 1272 * 393216,
              "languages": {"en": {"loss": 3.2, "tokens": 131072}}}
    events.write_text(json.dumps(record) + "\n")
    assert validation_curve(events, reference["stages"], 393216)[1272]["english_ce"] == 3.2
    record["tokens_seen"] -= 1
    events.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="token count"):
        validation_curve(events, reference["stages"], 393216)

    campaign_root = tmp_path / reference["campaign_id"]
    for name, loss, achieved in (("a", 3.0, 1272), ("b", 2.9, 1272), ("c", 2.8, 750)):
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
        updates = 1272

    assert [row["trial"] for row in rank(Args)] == ["b", "a"]
    Args.updates = 750
    with pytest.raises(ValueError, match="full 500M"):
        rank(Args)

    class RunArgs:
        trial = "short"
        reference = DEFAULT_REFERENCE
        updates = 750

    with pytest.raises(ValueError, match="full 500M"):
        run(RunArgs)


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
