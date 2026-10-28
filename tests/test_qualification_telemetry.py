"""CPU regression tests for full-state qualification telemetry."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from terminus.model import BitNetLlama
from terminus.optim import OptimizerConfig, build_optimizer
from terminus.train import run_training

from production.kimi_h100_5b import telemetry
from tests.helpers import checkpoint_payload, fixture, tiny_config


def fused_tiny():
    return BitNetLlama(tiny_config(fuse_projections=True, branch_norm="peri",
                                   value_residual="learned", qk_norm_mode="learned_rms"))


def one_step(model, optimizer):
    model.train()
    optimizer.zero_grad()
    inputs = torch.randint(0, model.config.vocab_size, (2, model.config.max_seq_len))
    targets = torch.randint(0, model.config.vocab_size, (2, model.config.max_seq_len))
    loss = model(inputs, targets)["loss_sum"] * (1 / inputs.numel())
    loss.backward()
    optimizer.step()


class GroupRecordTests(unittest.TestCase):
    def test_group_lr_records_cover_every_group(self):
        model = fused_tiny()
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        records = telemetry.group_lr_records(optimizer)
        self.assertEqual(len(records), len(optimizer.param_groups))
        names = [record["name"] for record in records]
        self.assertIn("tied_embedding", names)
        self.assertIn("qk_norm", names)
        qkv = next(record for record in records if record["name"].endswith("qkv_proj.weight"))
        self.assertEqual(qkv["logical_rows"], [32, 16, 16])
        self.assertEqual(qkv["lr"], qkv["base_lr"])


class StateHealthTests(unittest.TestCase):
    def test_required_gradients_detects_missing(self):
        model = fused_tiny()
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        one_step(model, optimizer)
        report = telemetry.required_gradient_report(model)
        self.assertEqual(report["missing"], [])
        self.assertGreater(report["checked"], 0)
        model.layers[0].attn.qkv_proj.weight.grad = None
        report = telemetry.required_gradient_report(model)
        self.assertEqual(report["missing"], ["layers.0.attn.qkv_proj.weight"])
        documented = telemetry.required_gradient_report(
            model, documented_absent=("layers.0.attn.qkv_proj.weight",))
        self.assertEqual(documented["missing"], [])
        with self.assertRaisesRegex(telemetry.QualificationFailure, "missing required"):
            telemetry.require_healthy_state(model, optimizer)

    def test_full_state_finite_complete_coverage(self):
        model = fused_tiny()
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        one_step(model, optimizer)
        report = telemetry.full_state_finite_report(model, optimizer)
        self.assertTrue(report["finite"])
        total = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreaterEqual(report["elements_checked"], 2 * total)
        with torch.no_grad():
            model.tok_embeddings.weight[0, 0] = float("nan")
        report = telemetry.full_state_finite_report(model, optimizer)
        self.assertFalse(report["finite"])
        self.assertIn("parameter:tok_embeddings.weight", report["failures"])
        with self.assertRaisesRegex(telemetry.QualificationFailure, "nonfinite"):
            telemetry.require_healthy_state(model, optimizer)


class RoleDeltaTests(unittest.TestCase):
    def test_role_deltas_cosines_and_decay_ledger(self):
        torch.manual_seed(7)
        model = fused_tiny()
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        tracker = telemetry.RoleDeltaTracker(optimizer)
        roles = [role["role"] for role in tracker.roles]
        self.assertIn("layers.0.attn.qkv_proj.weight:q", roles)
        self.assertIn("layers.0.ffn.gate_up_proj.weight:up", roles)
        self.assertIn("tied_embedding", roles)
        tracker.begin()
        tracker.record_decay(optimizer)
        one_step(model, optimizer)
        first = tracker.finish(1)
        tracker.begin()
        tracker.record_decay(optimizer)
        one_step(model, optimizer)
        second = tracker.finish(2)
        self.assertEqual(first["update_id"], 1)
        q_first = next(role for role in first["roles"] if role["role"].endswith(":q"))
        self.assertGreater(q_first["update_rms"], 0)
        self.assertGreater(q_first["weight_rms_after"], 0)
        self.assertNotIn("direction_cosine", q_first)
        q_second = next(role for role in second["roles"] if role["role"].endswith(":q"))
        self.assertEqual(q_second["compared_updates"], [1, 2])
        self.assertIsNotNone(q_second["direction_cosine"])
        muon_name = next(name for name in second["decay_log_retention"]
                         if name.endswith("qkv_proj.weight"))
        self.assertLess(second["decay_log_retention"][muon_name], 0)

    def test_observation_leaves_state_and_rng_unchanged(self):
        torch.manual_seed(11)
        model = fused_tiny()
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        one_step(model, optimizer)
        weights_before = {key: value.clone() for key, value in model.state_dict().items()}
        rng_before = torch.random.get_rng_state()
        tracker = telemetry.RoleDeltaTracker(optimizer)
        tracker.begin()
        report = tracker.finish(1)
        telemetry.require_healthy_state(model, optimizer)
        telemetry.group_lr_records(optimizer)
        telemetry.native_fp8_cache_report(model)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, weights_before[key]), key)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertGreater(report["seconds"], 0)


class NativeFp8CacheTests(unittest.TestCase):
    def test_absent_cache_is_observed_not_recomputed(self):
        model = fused_tiny()
        report = telemetry.native_fp8_cache_report(model)
        self.assertTrue(report["modules"])
        self.assertTrue(all(entry["cache"] == "absent" for entry in report["modules"].values()))
        for module in model.modules():
            if hasattr(module, "_fp8_weight_cache"):
                self.assertIsNone(module._fp8_weight_cache)


class SnapshotRestoreTests(unittest.TestCase):
    def test_restore_discards_branch_effects(self):
        torch.manual_seed(13)
        model = fused_tiny()
        optimizer = build_optimizer(model, OptimizerConfig(kind="muon"))
        one_step(model, optimizer)
        snapshot = telemetry.snapshot_state(model, optimizer)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        one_step(model, optimizer)
        self.assertGreater(telemetry.max_state_delta(before, model.state_dict()), 0)
        telemetry.restore_state(model, optimizer, snapshot)
        self.assertEqual(telemetry.max_state_delta(before, model.state_dict()), 0.0)


class TrainerCounterTests(unittest.TestCase):
    def test_valid_targets_and_data_wait_are_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "dataset", examples=8, batch_size=2, accumulation=2)
            run_training(root / "dataset/config.json", root / "dataset/manifest.json",
                         root / "dataset", root / "run")
            events = [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]
            trains = [event for event in events if event["event"] == "train"]
            self.assertEqual(len(trains), 2)
            for event in trains:
                self.assertEqual(event["valid_targets"], event["loss_tokens"])
                self.assertIn("data_wait_seconds", event)

    def test_padded_tail_still_reconciles_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "dataset", examples=9, batch_size=2, accumulation=2)
            run_training(root / "dataset/config.json", root / "dataset/manifest.json",
                         root / "dataset", root / "run")
            payload = checkpoint_payload(root / "run")
            self.assertEqual(payload["state"]["updates"], 3)
            events = [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]
            trains = [event for event in events if event["event"] == "train"]
            self.assertEqual([event["loss_tokens"] for event in trains], [32, 32, 8])
            self.assertEqual([event["valid_targets"] for event in trains], [32, 32, 8])


if __name__ == "__main__":
    unittest.main()
