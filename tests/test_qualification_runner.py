"""CPU regression tests for the bounded runner gates, group assertions and drive loop."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from terminus.model import BitNetLlama
from terminus.optim import OptimizerConfig, build_optimizer
from terminus.train import prepare_training

from production.kimi_h100_5b import budget, campaign, telemetry
from production.kimi_h100_5b.builder import build_trial_config
from production.kimi_h100_5b.runner import (assert_effective_groups, drive, gate_offline,
                                            run as runner_run)
from tests.helpers import fixture, tiny_config


def proxy_manifest(root):
    path = Path(root) / "manifest.json"
    path.write_text(json.dumps({
        "config": {"seq_len": 4096, "batch_size": 4, "total_train_tokens": 4_999_610_368,
                   "validation_tokens": 50_000_000},
        "shards": [], "tokenizer": {}, "provenance": {}}))
    return path


class OfflineGateTests(unittest.TestCase):
    def test_gate_offline_receipt_and_budget_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "H.json"
            config.write_text(json.dumps(build_trial_config("H", microbatch=4)))
            manifest = proxy_manifest(root)
            receipt = gate_offline(config, manifest, trial_id="H", confirmation=False)
            self.assertEqual(receipt["token_accounting"]["status"], "aligned_complete_updates")
            clock = [1_000_000.0]
            ledger_path = root / "budget.jsonl"
            ledger = budget.Ledger(ledger_path, clock=lambda: clock[0])
            ledger.open_allocation()
            receipt = gate_offline(config, manifest, trial_id="H", confirmation=False,
                                   ledger_path=ledger_path, stage_forecast_hours=7.0,
                                   now=clock[0])
            self.assertTrue(receipt["budget"]["fits"])
            clock[0] += 45 * 3600
            with self.assertRaises(budget.BudgetExhausted):
                gate_offline(config, manifest, trial_id="H", confirmation=False,
                             ledger_path=ledger_path, stage_forecast_hours=7.0,
                             now=clock[0])

    def test_gate_offline_rejects_stale_source_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "H.json"
            config.write_text(json.dumps(build_trial_config("H", microbatch=4)))
            manifest = proxy_manifest(root)
            stale = root / "source.json"
            stale.write_text(json.dumps({"source": {"terminus/train.py": "00" * 32}}))
            with self.assertRaisesRegex(ValueError, "source changed"):
                gate_offline(config, manifest, trial_id="H", confirmation=False,
                             source_receipt_path=stale)
            fresh = root / "source_ok.json"
            fresh.write_text(json.dumps({"source": campaign.qualification_source_hashes()}))
            receipt = gate_offline(config, manifest, trial_id="H", confirmation=False,
                                   source_receipt_path=fresh)
            self.assertEqual(receipt["source_receipt"], "verified")


class EffectiveGroupTests(unittest.TestCase):
    def fused_optimizer(self, **optimizer_changes):
        config = tiny_config(fuse_projections=True, branch_norm="peri",
                             value_residual="learned", qk_norm_mode="learned_rms")
        model = BitNetLlama(config)
        options = dict(kind="muon", muon_lr=0.017, adamw_lr=0.006, norm_lr=0.0003,
                       qk_norm_lr=0.0001, weight_decay=0.02, embedding_weight_decay=0.0)
        options.update(optimizer_changes)
        optimizer = build_optimizer(model, OptimizerConfig(**options))
        return config, optimizer, options

    def test_effective_groups_match_contract(self):
        config, optimizer, options = self.fused_optimizer()
        result = assert_effective_groups(config, optimizer, expected_optimizer=options)
        self.assertEqual(result["muon_groups"], 8)
        self.assertEqual(result["role_counts"], {"qkv": 2, "o": 2, "gate_up": 2, "down": 2})
        self.assertEqual(set(result["adam_labels"]),
                         {"tied_embedding", "input_norm", "peri_norm", "final_norm",
                          "value_mix", "qk_norm"})

    def test_effective_groups_reject_drift(self):
        config, optimizer, options = self.fused_optimizer()
        with self.assertRaisesRegex(ValueError, "base LR"):
            assert_effective_groups(config, optimizer,
                                    expected_optimizer={**options, "muon_lr": 0.0082})
        config2, optimizer2, options2 = self.fused_optimizer(embedding_weight_decay=0.01)
        with self.assertRaisesRegex(ValueError, "decay"):
            assert_effective_groups(config2, optimizer2,
                                    expected_optimizer={**options2,
                                                        "embedding_weight_decay": 0.0})


class DriveLoopTests(unittest.TestCase):
    def prepare(self, root, **training_changes):
        import json as _json
        from dataclasses import asdict

        fixture(root / "dataset", examples=16, batch_size=2, accumulation=2)
        config_path = root / "dataset/config.json"
        config = _json.loads(config_path.read_text())
        config["training"].update(training_changes)
        config_path.write_text(_json.dumps(config))
        return prepare_training(config_path, root / "dataset/manifest.json",
                                root / "dataset", root / "run")

    def test_drive_records_boundaries_and_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = [1_000_000.0]
            ledger = budget.Ledger(root / "budget.jsonl", clock=lambda: clock[0])
            ledger.open_allocation()
            with self.prepare(root) as workload:
                summary = drive(workload, boundaries=(2, 3, 4),
                                ledger_path=root / "budget.jsonl", final_update=4,
                                output_dir=root / "run")
            self.assertEqual(summary["updates_completed"], 4)
            records = [json.loads(line)
                       for line in (root / "run/qualification.jsonl").read_text().splitlines()]
            updates = [r for r in records if r["event"] == "qualification_update"]
            self.assertEqual(len(updates), 4)
            for record in updates:
                self.assertEqual(record["valid_targets"], record["update_tokens"])
                self.assertTrue(record["finite"]["finite"])
                self.assertGreater(record["finite"]["elements_checked"], 0)
                self.assertEqual(len(record["group_lrs"]),
                                 len(record["group_lrs"]))
                self.assertTrue(all("lr" in group and "name" in group
                                    for group in record["group_lrs"]))
            boundary = next(r for r in updates if r["updates"] == 3)
            self.assertIn("role_diagnostics", boundary)
            self.assertTrue((root / "run/checkpoints/step_00000003.pt").is_file())
            events = [json.loads(line)
                      for line in (root / "run/events.jsonl").read_text().splitlines()]
            validations = [event["updates"] for event in events if event["event"] == "validation"]
            self.assertIn(3, validations)  # forced boundary validation
            self.assertEqual(summary["alarms"], [])

    def test_drive_fails_closed_on_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.prepare(root) as workload:
                original = workload.complete_update

                def tampered():
                    result = original()
                    workload.valid_targets = workload.update_tokens + 1
                    return result

                workload.complete_update = tampered
                with self.assertRaisesRegex(telemetry.QualificationFailure, "valid targets"):
                    drive(workload, boundaries=(2, 3, 4), ledger_path=None,
                          final_update=4, output_dir=root / "run")

    def test_reference_replay_preserves_run_and_excludes_branch_tokens(self):
        from production.kimi_h100_5b import branches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.prepare(root) as workload:
                report = branches.reference_replay(workload)
                self.assertEqual(report["replayed_update"], 1)
                self.assertEqual(report["max_weight_delta"], 0.0)
                self.assertEqual(report["max_moment_delta"], 0.0)
                self.assertEqual(workload.state["updates"], 1)
                branches.require_within_tolerance(
                    report, max_weight_delta=0.0, max_moment_delta=0.0)
                # The main run continues bit-identically after the branch.
                second = workload.complete_update()
                self.assertEqual(workload.state["updates"], 2)
                self.assertEqual(second, workload.update_tokens)
                with self.assertRaisesRegex(branches.BranchFailure, "exceeds"):
                    branches.require_within_tolerance(
                        {**report, "max_weight_delta": 1e-6},
                        max_weight_delta=0.0, max_moment_delta=0.0)


class RunnerGuardTests(unittest.TestCase):
    def args(self, root, **changes):
        values = dict(config="c", manifest="m", data_root="d",
                      output_dir=str(root / "H"), trial="H", confirmation_from=None,
                      probe=None, runtime_report="r", staging_receipt="s",
                      preflight_report="p", source_receipt="src", ledger="l",
                      stage_forecast_hours=7.0, emergency_reserve_hours=0.5,
                      backup_destination=None, backup_max_pending_bytes=1,
                      backup_max_pending_seconds=1.0, min_free_bytes=0,
                      reference_probes=False, replay_max_weight_delta=None,
                      replay_max_moment_delta=None, resume=None, stop_after_updates=None)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_candidate_output_dir_and_truncation_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "named 'H'"):
                runner_run(self.args(root, output_dir=str(root / "elsewhere")))
            with self.assertRaisesRegex(ValueError, "never truncate"):
                runner_run(self.args(root, stop_after_updates=10))
            with self.assertRaisesRegex(ValueError, "requires --stop-after-updates"):
                runner_run(self.args(root, output_dir=str(root / "probe_fit"),
                                     probe="fit"))
            with self.assertRaisesRegex(ValueError, "requires --stop-after-updates"):
                runner_run(self.args(root, output_dir=str(root / "probe_full"),
                                     probe="fit", trial=None, confirmation_from="R0"))
            with self.assertRaisesRegex(ValueError, "candidate output"):
                runner_run(self.args(root, output_dir=str(root / "R0"),
                                     probe="fit", stop_after_updates=2))

    def test_reference_probes_require_frozen_tolerances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "H.json"
            config.write_text(json.dumps(build_trial_config("H", microbatch=4)))
            manifest = proxy_manifest(root)
            args = self.args(root, config=str(config), manifest=str(manifest),
                             reference_probes=True)
            with self.assertRaisesRegex(ValueError, "tolerances"):
                runner_run(args)


if __name__ == "__main__":
    unittest.main()
