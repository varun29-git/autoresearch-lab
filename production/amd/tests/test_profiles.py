from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from production.amd.prepare import prepare
from production.amd.profiles import (BRIDGE_END_TOKENS, CADENCE_TOKENS, GLOBAL_BATCH_TOKENS,
                                    PROFILES, VALIDATION_SEQUENCES, WARMUP_TOKENS, select)
from production.amd.recipe import validate_production
from production.amd.recipe import SELECTED_DENSE_OPTIMIZER as SELECTED_OPTIMIZER
from terminus.data import TokenBatcher, TokenDataConfig, TokenManifest, TokenShardSpec
from terminus.optim import OptimizerConfig
from terminus.train import TrainConfig, evaluate, load_config, make_resume_contract, schedule_scale


def manifest(tokens=10**12):
    # Metadata fixture only; no fabricated data is passed through staging.
    validation = VALIDATION_SEQUENCES * 4096 + 1
    return TokenManifest(
        TokenDataConfig(batch_size=7, seq_len=4096, total_train_tokens=tokens,
                        validation_tokens=validation - 1, source_mix={"en": .8, "code": .2}),
        [TokenShardSpec("en.bin", "en", "train", tokens + 1, language="en"),
         TokenShardSpec("code.bin", "code", "train", tokens + 1, language="en"),
         TokenShardSpec("en-val.bin", "en", "val", validation, language="en"),
         TokenShardSpec("code-val.bin", "code", "val", validation, language="en")],
        tokenizer={"sha256": "a" * 64},
        provenance={"scope": {"name": "test-en-code-v1", "languages": ["en"], "domains": ["code"]}})


@pytest.mark.parametrize("name", PROFILES)
def test_selected_profiles_roundtrip_and_comparison_contract(tmp_path, name):
    original = manifest()
    before = asdict(original)
    derived, config, report = select(original, profile=name)
    selected = PROFILES[name]
    # The actual data/phase contract survives; only grouping into batches changes.
    expected = asdict(original)
    expected["config"]["batch_size"] = selected.microbatch
    assert asdict(derived) == expected
    assert asdict(original) == before
    training = config["training"]
    assert selected.world_size * derived.config.tokens_per_step * training["gradient_accumulation_steps"] == selected.global_batch_tokens
    assert 1024 <= training["validation_batches_per_language"] * selected.microbatch * selected.world_size < 1024 + selected.microbatch * selected.world_size
    assert training["validation_sequences_per_language"] == 1024
    assert report["validation_tokens_per_check"] == 2 * 1024 * 4096
    assert report["bridge_stop_after_updates"] * selected.global_batch_tokens == BRIDGE_END_TOKENS
    assert report["cadence_tokens"] == CADENCE_TOKENS
    assert WARMUP_TOKENS < BRIDGE_END_TOKENS < report["decay_start_tokens"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    model, loaded, optimizer = load_config(path)
    assert validate_production(model, loaded, optimizer) == selected.candidate
    assert optimizer == OptimizerConfig(**SELECTED_OPTIMIZER)
    assert report["allocation"]["measured_rate"] is None
    assert report["allocation"]["scope"] == "standalone_dense_only"


def test_schedule_token_alignment_and_manifest_horizon_are_preserved():
    original = manifest(tokens=1_000_000_000_001)
    derived, config, report = select(original)
    assert derived.config.total_train_tokens == 1_000_000_000_001
    total, decay = derived.config.usable_train_tokens, report["decay_tokens"]
    assert 0 <= decay - total / 10 < 4096
    assert report["discarded_subsequence_tail_tokens"] == original.config.total_train_tokens % 4096
    settings = {key: config["training"][key] for key in
                ("warmup_tokens", "final_lr_fraction", "schedule", "decay_tokens", "decay_style")}
    assert schedule_scale(WARMUP_TOKENS // 2, total, **settings) == .5
    assert schedule_scale(WARMUP_TOKENS, total, **settings) == 1
    assert schedule_scale(BRIDGE_END_TOKENS, total, **settings) == 1
    assert schedule_scale(total - decay, total, **settings) == 1
    assert schedule_scale(total, total, **settings) == .1
    assert select(manifest(tokens=10_000_000_000))[2]["bridge_end_tokens"] == BRIDGE_END_TOKENS


def test_fail_closed_on_short_heldouts_or_incompatible_inputs():
    original = manifest()
    short = replace(original.shards[-1], num_tokens=1024 * 4096)  # Needs one next-token target.
    with pytest.raises(ValueError, match="code"):
        select(replace(original, shards=original.shards[:-1] + [short]))
    with pytest.raises(ValueError, match="10B"):
        select(manifest(tokens=10**9))
    with pytest.raises(ValueError, match="26 complete"):
        select(manifest(tokens=25 * GLOBAL_BATCH_TOKENS), purpose="smoke")
    with pytest.raises(ValueError, match="4096-token uint16"):
        select(replace(original, config=replace(original.config, seq_len=2048)))
    with pytest.raises(ValueError, match="unknown"):
        select(original, profile="auto")


def test_smoke_is_explicit_and_does_not_claim_a_quality_bridge():
    _, config, report = select(manifest(tokens=26 * GLOBAL_BATCH_TOKENS), purpose="smoke")
    training = config["training"]
    assert training["warmup_tokens"] == 2 * GLOBAL_BATCH_TOKENS
    assert training["stability_log_interval"] == 1
    assert report["bridge_stop_after_updates"] is None
    assert report["purpose"] == "smoke"


def test_hpo_batch_can_use_one_eight_gpu_microstep_without_changing_the_optimizer():
    original = manifest()
    _, conservative, _ = select(original, profile="batch_bridge")
    _, larger_microbatch, _ = select(original, profile="batch_bridge_mb12")
    assert conservative["training"]["gradient_accumulation_steps"] == 3
    assert larger_microbatch["training"]["gradient_accumulation_steps"] == 1
    assert conservative["optimizer"] == larger_microbatch["optimizer"]
    assert conservative["training"]["warmup_tokens"] == larger_microbatch["training"]["warmup_tokens"]


def test_fused_profile_keeps_the_balanced_batch_and_leaves_the_default_off():
    original = manifest()
    _, balanced, _ = select(original, profile="balanced")
    _, fused, _ = select(original, profile="fused")
    assert balanced["model"]["fused_fp8_producers"] is False
    assert fused["model"]["fused_fp8_producers"] is True
    assert fused["model"]["linear_backend"] == balanced["model"]["linear_backend"] == "rocm_fp8_te"
    assert fused["model"]["fp8_wgrad_splits"] == balanced["model"]["fp8_wgrad_splits"] == 0
    assert fused["training"]["gradient_accumulation_steps"] == balanced["training"]["gradient_accumulation_steps"] == 4
    assert fused["optimizer"] == balanced["optimizer"]
    with pytest.raises(ValueError, match="producer"):
        select(original, profile="fused", candidate="bf16")


def test_explicit_backend_fallback_keeps_selected_batch_and_data_phases():
    original = manifest(tokens=10_000_000_000)
    original = replace(original, config=replace(original.config, source_mix={}, phase_source_mixes=[
        {"name": "first", "tokens": 4_096_000_000, "source_mix": {"en": .8, "code": .2}},
        {"name": "second", "tokens": 5_904_000_000, "source_mix": {"en": .5, "code": .5}},
    ]))
    derived, config, report = select(original, profile="mb8", candidate="fp8")
    assert derived.config.phase_source_mixes == original.config.phase_source_mixes
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["model"]["linear_backend"] == "rocm_fp8"
    assert config["model"]["fp8_wgrad_splits"] == 4
    assert report["candidate_override"] == report["selection"]["candidate"] == "fp8"


def test_external_bundle_is_consumable_and_shell_safe(tmp_path):
    source = tmp_path / "input.json"
    manifest().save(source)
    before = source.read_bytes()
    output = tmp_path / "bundle ' with $(touch BAD)"
    report = prepare(manifest_path=source, output_dir=output)
    assert source.read_bytes() == before
    assert load_config(output / "config.json")[1].gradient_accumulation_steps == 4
    assert TokenManifest.load(output / "manifest.json").config.batch_size == 16
    assert json.loads((output / "selection.json").read_text()) == report
    # Source the generated shell file using positional arguments, never string interpolation.
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s" "$CONFIG"', "bash", str(output / "env.sh")],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout == str(output / "config.json")
    assert not (tmp_path / "BAD").exists()
    with pytest.raises(FileExistsError):
        prepare(manifest_path=source, output_dir=output)
    with pytest.raises(ValueError, match="outside"):
        prepare(manifest_path=source, output_dir=Path(__file__).parent / "generated-do-not-create")


def test_real_loader_preserves_global_update_under_microbatch_and_rank_changes(tmp_path):
    # Exercise actual mmap/rank-striding, not just the batch multiplication.
    tokens = np.random.default_rng(19).integers(0, 65536, 3_145_728 + 1, dtype=np.uint16)
    tokens.tofile(tmp_path / "train.bin")
    tokens[:4097].tofile(tmp_path / "val.bin")
    specs = [TokenShardSpec("train.bin", "en", "train", len(tokens), language="en"),
             TokenShardSpec("val.bin", "en", "val", 4097, language="en")]
    reference_x = torch.from_numpy(tokens[:-1].astype(np.int64)).reshape(-1, 4096)
    reference_y = torch.from_numpy(tokens[1:].astype(np.int64)).reshape(-1, 4096)
    for name in ("balanced", "mb4", "mb8", "mb32", "batch_bridge", "batch_bridge_mb12", "batch_1m_mb16",
                 "batch_3m_mb16", "batch_3m_mb24", "single_gpu"):
        chosen = PROFILES[name]
        data = TokenManifest(TokenDataConfig(batch_size=chosen.microbatch, seq_len=4096,
                             total_train_tokens=chosen.global_batch_tokens, validation_tokens=4096), specs)
        batchers = [TokenBatcher(data, root=tmp_path, rank=rank, world_size=chosen.world_size)
                    for rank in range(chosen.world_size)]
        steps = chosen.global_batch_tokens // (chosen.world_size * data.config.tokens_per_step)
        assert steps * chosen.world_size * data.config.tokens_per_step == chosen.global_batch_tokens
        xs, ys = [], []
        for step in range(steps):
            rank_batches = [batcher.train_batch(step) for batcher in batchers]
            xs.append(torch.stack([pair[0] for pair in rank_batches], dim=1).flatten(0, 1))
            ys.append(torch.stack([pair[1] for pair in rank_batches], dim=1).flatten(0, 1))
        actual_x, actual_y = torch.cat(xs), torch.cat(ys)
        assert torch.equal(actual_x, reference_x[:len(actual_x)]), name
        assert torch.equal(actual_y, reference_y[:len(actual_y)]), name


def test_validation_uses_same_examples_when_microbatch_has_partial_final_batch(tmp_path):
    # The 24-sequence profile needs a partial last validation batch to preserve coverage.
    values = np.arange(8 * 8 + 1, dtype=np.uint16)
    values.tofile(tmp_path / "val.bin")

    class Recorder:
        training = True

        def __init__(self):
            self.starts = []
            self.batch_sizes = []

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

        def __call__(self, inputs, targets, return_logits=False):
            self.starts.extend(inputs[:, 0].tolist())
            self.batch_sizes.append(len(inputs))
            return {"loss_sum": targets.double().sum(), "loss_tokens": torch.tensor(targets.numel())}

    traces = []
    for batch_size in (2, 3):
        data = TokenManifest(TokenDataConfig(seq_len=8, batch_size=batch_size, total_train_tokens=64,
                             validation_tokens=64), [TokenShardSpec("val.bin", "en", "val", len(values), language="en")])
        recorder = Recorder()
        result = evaluate(recorder, data, tmp_path, torch.device("cpu"), torch.float32,
                          max_batches=3, rank=0, world_size=1, sequences_per_language=5)
        assert recorder.training
        assert result["en"]["tokens"] == 40
        traces.append((recorder.starts, recorder.batch_sizes))
    assert traces == [([0, 8, 24, 40, 56], [2, 2, 1]),
                      ([0, 8, 24, 40, 56], [3, 2])]
    assert TrainConfig().validation_sequences_per_language is None
    with pytest.raises(ValueError, match="validation_sequences"):
        TrainConfig(validation_sequences_per_language=0)


def test_validation_sequence_cap_is_operational_for_resume(monkeypatch):
    monkeypatch.setattr("terminus.train.source_hashes", lambda: {})
    monkeypatch.setattr("terminus.train.runtime_contract", lambda _device: {})

    @dataclass
    class StubConfig:
        value: int = 1

    training = TrainConfig()
    args = (StubConfig(), StubConfig(), {}, 1, torch.device("cpu"))
    original = make_resume_contract(args[0], training, *args[1:])
    changed = make_resume_contract(args[0], replace(training, validation_sequences_per_language=1024), *args[1:])
    assert original == changed
