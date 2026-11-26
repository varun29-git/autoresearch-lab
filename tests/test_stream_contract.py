from dataclasses import replace

import numpy as np
import pytest
import torch

from terminus.data import TokenDataConfig, TokenManifest, TokenShardSpec, TrainingTokenRouter
from terminus.integrity import file_sha256
from terminus.stream_contract import capture_training_stream, verify_training_continuation


def routers(tmp_path, parent_config, child_config):
    shards = []
    for source, offset in (("en", 0), ("hi", 20000)):
        path = tmp_path / (source + ".bin")
        np.arange(offset, offset + 12000, dtype=np.uint16).tofile(path)
        shards.append(TokenShardSpec(path.name, source, "train", 12000, sha256=file_sha256(path)))
    return [TrainingTokenRouter(TokenManifest(config, shards), root=tmp_path)
            for config in (parent_config, child_config)]


def verify(parent, child, tokens, policy="continue"):
    return verify_training_continuation(capture_training_stream(parent), capture_training_stream(child),
                                        tokens, policy=policy)


def test_batch_and_budget_change_preserves_actual_prefix(tmp_path):
    config = TokenDataConfig(seq_len=4, total_train_tokens=8000, source_mix={"en": .6, "hi": .4})
    parent, child = routers(tmp_path, config, replace(config, batch_size=12, total_train_tokens=12000))
    proof = verify(parent, child, 5000)
    assert proof["verified_prefix_tokens"] == 8000
    for offset in (0, 4, 3996, 4000, 4996, 5000, 7996):
        for a, b in zip(parent.block_at_token_offset(offset), child.block_at_token_offset(offset)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_phase_mix_changes_only_future_routing(tmp_path):
    config = TokenDataConfig(seq_len=4, total_train_tokens=64, source_mix={"en": .5, "hi": .5})
    child_config = replace(config, total_train_tokens=96, source_mix={}, phase_source_mixes=[
        {"tokens": 32, "source_mix": {"en": .5, "hi": .5}},
        {"tokens": 64, "source_mix": {"en": .75, "hi": .25}},
    ])
    parent, child = routers(tmp_path, config, child_config)
    assert verify(parent, child, 32, "phase_mix")["verified_prefix_tokens"] == 32
    for offset in range(0, 32, 4):
        for a, b in zip(parent.block_at_token_offset(offset), child.block_at_token_offset(offset)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="source order|mixture"):
        verify(parent, child, 32)


@pytest.mark.parametrize("change", ["mixture", "offset", "shard"])
def test_changed_stream_cannot_hide_behind_a_matching_token_counter(tmp_path, change):
    config = TokenDataConfig(seq_len=4, total_train_tokens=8000, source_mix={"en": .6, "hi": .4})
    child_config = config
    if change == "mixture":
        child_config = replace(config, source_mix={"en": .5, "hi": .5})
    elif change == "offset":
        child_config = replace(config, source_start_examples={"en": 1})
    parent, child = routers(tmp_path, config, child_config)
    p, c = capture_training_stream(parent), capture_training_stream(child)
    if change == "shard":
        c["sources"]["en"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data continuation changed"):
        verify_training_continuation(p, c, 4000, policy="phase_mix")


def test_continuation_requires_proof_for_legacy_contract():
    with pytest.raises(ValueError, match="training-stream metadata"):
        verify_training_continuation(None, {}, 32, policy="continue")


@pytest.mark.parametrize("prefix_tokens", [0, 32])
def test_sequential_and_prefix_routes_match_loader(tmp_path, prefix_tokens):
    config = TokenDataConfig(seq_len=4, total_train_tokens=80,
                             cosmopedia_stories_prefix_tokens=prefix_tokens,
                             prefix_source="hi" if prefix_tokens else "")
    parent, child = routers(tmp_path, config, replace(config, total_train_tokens=160))
    assert verify(parent, child, 40)["verified_prefix_tokens"] == 80
    for offset in range(0, 80, 4):
        for a, b in zip(parent.block_at_token_offset(offset), child.block_at_token_offset(offset)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_matching_multiphase_plan_preserves_offsets_and_shard_append(tmp_path):
    config = TokenDataConfig(seq_len=4, total_train_tokens=8000, phase_source_mixes=[
        {"tokens": 4000, "source_mix": {"en": .6, "hi": .4}},
        {"tokens": 4000, "source_mix": {"en": .4, "hi": .6}},
    ])
    child_config = replace(config, total_train_tokens=12000, phase_source_mixes=[
        config.phase_source_mixes[0], {"tokens": 8000, "source_mix": {"en": .4, "hi": .6}},
    ])
    parent, child = routers(tmp_path, config, child_config)
    p, c = capture_training_stream(parent), capture_training_stream(child)
    # Immutable prefixes can be relocated or extended; paths are deliberately
    # absent and only the shards covering the common stream are compared.
    c["sources"]["en"].append({"sha256": "a" * 64, "num_tokens": 1000, "dtype": "uint16"})
    assert verify_training_continuation(p, c, 6000, policy="continue")["verified_prefix_tokens"] == 8000
    for offset in (3996, 4000, 4004, 5996, 6000, 7996):
        for a, b in zip(parent.block_at_token_offset(offset), child.block_at_token_offset(offset)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
