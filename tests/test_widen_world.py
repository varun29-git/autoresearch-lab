import json
from pathlib import Path

import torch

from terminus.checkpoint import load_checkpoint
from terminus.widen_world import widen_checkpoint, widen_payload


def _contract(accumulation=8, world=4):
    return {
        "model": {"dim": 4},
        "training": {"seed": 1337, "gradient_accumulation_steps": accumulation, "schedule": "wsd"},
        "optimizer": {"muon_lr": 0.017},
        "data": {"manifest": "packed"},
        "world_size": world,
        "source": {"terminus": "test"},
        "runtime": {"device": "cpu"},
    }


def _payload(**overrides):
    updates = overrides.pop("updates", 3)
    accumulation = overrides.pop("accumulation", 8)
    payload = {
        "schema_version": 1,
        "contract": _contract(accumulation),
        "state": {
            "updates": updates,
            "next_micro_step": updates * accumulation,
            "tokens_seen": updates * 2_097_152,
            "input_slots_seen": updates * 2_097_152,
            "content_tokens_seen": updates * 2_000_000,
        },
        "model": {"weight": torch.ones(4)},
        "optimizer": {"state": {"0": {"exp_avg": torch.zeros(4)}}, "param_groups": []},
        "rank_rng": [{"rank": rank} for rank in range(4)],
    }
    payload.update(overrides)
    return payload


def test_widen_keeps_weights_tokens_and_halves_the_microstep():
    original = _payload()
    widened = widen_payload(original)
    assert original["contract"]["world_size"] == 4
    assert original["state"]["next_micro_step"] == 24
    assert widened["contract"]["world_size"] == 8
    assert widened["contract"]["training"]["gradient_accumulation_steps"] == 4
    assert widened["contract"]["training"]["seed"] == 1337
    assert widened["contract"]["data"] == original["contract"]["data"]
    assert widened["contract"]["runtime"] == original["contract"]["runtime"]
    assert widened["state"]["updates"] == 3
    assert widened["state"]["next_micro_step"] == 12
    assert widened["state"]["tokens_seen"] == original["state"]["tokens_seen"]
    assert widened["state"]["input_slots_seen"] == original["state"]["input_slots_seen"]
    assert widened["state"]["content_tokens_seen"] == original["state"]["content_tokens_seen"]
    assert torch.equal(widened["model"]["weight"], original["model"]["weight"])
    assert torch.equal(widened["optimizer"]["state"]["0"]["exp_avg"], original["optimizer"]["state"]["0"]["exp_avg"])
    assert [item["rank"] for item in widened["rank_rng"][:4]] == [0, 1, 2, 3]
    assert len(widened["rank_rng"]) == 8
    assert all("torch" in item for item in widened["rank_rng"][4:])


def test_microbatch_32_prefix_halves_accumulation_from_4_to_2():
    widened = widen_payload(_payload(accumulation=4, updates=2))
    assert widened["contract"]["training"]["gradient_accumulation_steps"] == 2
    assert widened["state"]["next_micro_step"] == 4
    assert widened["state"]["tokens_seen"] == 2 * 2_097_152


def test_widen_refuses_a_partial_update_or_the_wrong_accumulation():
    partial = _payload()
    partial["state"]["next_micro_step"] -= 1
    try:
        widen_payload(partial)
    except ValueError as error:
        assert "finished update" in str(error)
    else:
        raise AssertionError("partial update was widened")
    try:
        widen_payload(_payload(accumulation=7))
    except ValueError as error:
        assert "even number" in str(error)
    else:
        raise AssertionError("wrong accumulation was widened")


def test_widened_checkpoint_loads_under_the_eight_gpu_contract(tmp_path):
    source = tmp_path / "four"
    # commit_checkpoint_payload is reached through the public writer, which checks integrity.
    from terminus.checkpoint import commit_checkpoint_payload

    saved = _payload()
    # The committed file must carry real generator state; the in-memory fixture uses markers.
    from terminus.widen_world import _fresh_rank_rng

    saved["rank_rng"] = [_fresh_rank_rng(1337, rank) for rank in range(4)]
    commit_checkpoint_payload(source, saved)
    output = tmp_path / "eight"
    path, payload = widen_checkpoint(source, output)
    assert path.name == "step_00000003.pt"
    loaded = load_checkpoint(output, payload["contract"])
    assert loaded["state"]["next_micro_step"] == 12
    assert loaded["contract"]["world_size"] == 8
    receipt = json.loads((output / "latest.json").read_text())
    assert receipt["state"]["tokens_seen"] == 3 * 2_097_152
    try:
        load_checkpoint(output, saved["contract"])
    except ValueError as error:
        assert "resume contract" in str(error)
    else:
        raise AssertionError("eight-GPU file opened under the four-GPU contract")
    assert not (source / "latest.json").read_text() == (output / "latest.json").read_text()
