"""24-hour campaign accounting and prospective checkpoint-fork regressions."""

from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile

import pytest
import torch

from production.h100_24h import campaign
from production.h100_24h.auto import require_campaign_budget
from production.h100_24h.budget import BudgetExhausted, Ledger
from production.h100_24h.fork import fork_dense_checkpoint
from production.h100_24h.prepare24 import prepare
from terminus.data import TokenManifest
from terminus.train import run_training
from tests.helpers import checkpoint_payload, fixture


def _tiny_parent(root: Path):
    data = root / "data"
    fixture(data, examples=24, batch_size=2, accumulation=2, dropout=.1)
    config_path = data / "config.json"
    config = json.loads(config_path.read_text())
    config["model"]["weight_mode"] = "dense"
    config["training"].update(schedule="wsd", warmup_tokens=32, decay_tokens=64)
    config_path.write_text(json.dumps(config))
    parent = root / "parent"
    with contextlib.redirect_stdout(io.StringIO()):
        run_training(config_path, data / "manifest.json", data, parent, stop_after_updates=2)
    events = [json.loads(line) for line in (parent / "events.jsonl").read_text().splitlines()]
    with (parent / "qualification.jsonl").open("w") as handle:
        for event in events:
            if event["event"] == "train":
                handle.write(json.dumps({"event": "qualification_update", "updates": event["updates"],
                                         "finite": {"finite": True},
                                         "required_gradients": {"checked": 1}}) + "\n")
    manifest = TokenManifest.load(data / "manifest.json")
    child_manifest = replace(manifest, config=replace(manifest.config, total_train_tokens=128))
    child_manifest.save(root / "child_manifest.json")
    child_config = json.loads(config_path.read_text())
    child_config["training"]["decay_tokens"] = 32
    (root / "child_config.json").write_text(json.dumps(child_config))
    return data, parent, config_path


def test_fork_preserves_next_update_weights_moments_rng_and_cursor():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data, parent, config = _tiny_parent(root)
        receipt = fork_dense_checkpoint(
            parent, data / "manifest.json", root / "child_config.json",
            root / "child_manifest.json", data, root / "child", expected_parent_updates=2)
        assert receipt["past_lr_identical_updates"] == 2
        assert receipt["stream_continuity"]["verified_prefix_tokens"] == 128
        assert receipt["retained_validations"] == [2]
        with contextlib.redirect_stdout(io.StringIO()):
            run_training(config, data / "manifest.json", data, parent,
                         resume=parent / "checkpoints/step_00000002.pt", stop_after_updates=3)
            run_training(root / "child/config.json", root / "child_manifest.json", data,
                         root / "child", resume=root / "child/checkpoints/step_00000002.pt",
                         stop_after_updates=3)
        expected, actual = checkpoint_payload(parent), checkpoint_payload(root / "child")
        assert expected["state"] == actual["state"]
        for name, value in expected["model"].items():
            torch.testing.assert_close(value, actual["model"][name], rtol=0, atol=0)
        for key, state in expected["optimizer"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, actual["optimizer"]["state"][key][name],
                                           rtol=0, atol=0)


def test_fork_rejects_lr_change_in_consumed_prefix_and_missing_health_record():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data, parent, _ = _tiny_parent(root)
        path = root / "child_config.json"
        config = json.loads(path.read_text())
        config["training"]["warmup_tokens"] = 64
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="future WSD"):
            fork_dense_checkpoint(parent, data / "manifest.json", path,
                                  root / "child_manifest.json", data, root / "child",
                                  expected_parent_updates=2)
        config["training"]["warmup_tokens"] = 32
        path.write_text(json.dumps(config))
        (parent / "qualification.jsonl").write_text("")
        with pytest.raises(ValueError, match="full-state checks"):
            fork_dense_checkpoint(parent, data / "manifest.json", path,
                                  root / "child_manifest.json", data, root / "child",
                                  expected_parent_updates=2)


def test_24_hour_budget_rejects_uncredited_five_arm_campaign():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for trial in campaign.TRIAL_ORDER:
            (root / "trials" / trial).mkdir(parents=True)
        ledger = Ledger(root / "budget.jsonl", clock=lambda: 1000.0)
        ledger.open_allocation()
        receipt = require_campaign_budget(root, ledger, h_fork_credit=True,
                                          proxy_tps=163_248, full_tps=35_000)
        assert receipt["needed_hours"] + receipt["reserve_hours"] < 24
        with pytest.raises(BudgetExhausted):
            require_campaign_budget(root, ledger, h_fork_credit=False,
                                    proxy_tps=163_248, full_tps=35_000)


def test_packet_builds_short_horizons_without_shard_rewrite():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "new"
        parent_env = os.environ.get("CI1_H100_PARENT_PACKET")
        if not parent_env:
            pytest.skip("set CI1_H100_PARENT_PACKET for the real historical packet check")
        parent = Path(parent_env)
        receipt = prepare(parent, root)
        assert receipt["proxy_tokens_each"] == 1_998_585_856
        assert receipt["full_tokens"] == 599_785_472
        assert json.loads((root / "gates/checkpoint_fork.json").read_text())["status"] == "pending_native"
