import copy

import numpy as np
import pytest
import torch

from production.document_packing import ShardWriter, fragments
from terminus.model import BitNetLlama
from terminus.packed_data import evaluate_packed, verify_shard
from tests.test_document_training import packed_fixture


def test_reserved_lookahead_label_is_rejected_even_when_payload_is_clean(tmp_path):
    # A full context has a label in its descriptor, outside the uint16 payload.
    item = next(fragments(np.full(9, 40, dtype=np.uint16), {"id": "long"}, length=8))
    item.lookahead = 5
    writer = ShardWriter(tmp_path, "lookahead", length=8, write_offsets=True)
    writer.add([item])
    shard = writer.close()
    shard.update(leaf="en/test", stratum="en/test")
    paths = {}
    for name in ("payload", "index", "index_offsets"):
        shard[name]["uri"] = name
        paths[name] = shard[name]["path"]
    with pytest.raises(ValueError, match="reserved"):
        verify_shard(shard, paths, tmp_path, 64)


def weighted_fixture(tmp_path):
    config, manifest = packed_fixture(tmp_path)
    phases = manifest.provenance["packed_stream"]["val"]
    second = copy.deepcopy(phases[0])
    second["shards"][0].update(leaf="bn/test", stratum="bn/test")
    phases.append(second)
    return BitNetLlama(config), manifest


@pytest.mark.parametrize("weights", [
    {"en": 1}, {"en": 1, "missing": 1}, {"en": -1, "bn": 2},
    {"en": 0, "bn": 1}, {"en": True, "bn": 1},
    {"en": float("nan"), "bn": 1}, {"en": float("inf"), "bn": 1},
])
def test_invalid_heldout_language_weights_fail_closed(tmp_path, weights):
    model, manifest = weighted_fixture(tmp_path)
    manifest.provenance["packed_stream"]["val"][0]["language_loss_weight_tokens"] = weights
    with pytest.raises(ValueError, match="language loss weights"):
        evaluate_packed(model, manifest, tmp_path, torch.device("cpu"), torch.float32, 2, 0, 1)


def test_conflicting_heldout_weights_are_rejected(tmp_path):
    model, manifest = weighted_fixture(tmp_path)
    phases = manifest.provenance["packed_stream"]["val"]
    phases[0]["language_loss_weight_tokens"] = {"en": 1, "bn": 1}
    phases[1]["language_loss_weight_tokens"] = {"en": 2, "bn": 1}
    with pytest.raises(ValueError, match="language loss weights"):
        evaluate_packed(model, manifest, tmp_path, torch.device("cpu"), torch.float32, 2, 0, 1)


def test_language_mixture_does_not_double_count_domain_metrics(tmp_path):
    model, manifest = weighted_fixture(tmp_path)
    phases = manifest.provenance["packed_stream"]["val"]
    domain = copy.deepcopy(phases[0])
    domain["shards"][0]["evaluation_key"] = "code"
    phases.append(domain)
    phases[0]["language_loss_weight_tokens"] = {"en": 3, "bn": 1}
    result = evaluate_packed(model, manifest, tmp_path, torch.device("cpu"), torch.float32, 2, 0, 1)
    assert result["mixture"]["loss"] == pytest.approx((3*result["en"]["loss"]+result["bn"]["loss"])/4)
    assert result["mixture"]["tokens"] == result["en"]["tokens"]+result["bn"]["tokens"]
    assert "code" in result
