"""Campaign invariants that prevent architecture trials from changing the yardstick."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.loop import trial_config, trial_manifest
from research.protocol import (compare_losses, grouped_losses, load_candidate,
                               paired_batch_interval, policy, validate_candidate,
                               validate_task_manifest)


class ResearchProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rules = policy()

    def test_baseline_candidate_and_fixed_training_policy(self):
        model = load_candidate(self.rules)
        self.assertFalse(model["use_subln"])
        rendered = trial_config(model, self.rules)
        self.assertEqual(rendered["model"], model)
        self.assertEqual(rendered["training"]["warmup_tokens"], 250_000_000)
        self.assertEqual(rendered["training"]["decay_tokens"], 500_000_000)
        self.assertEqual(rendered["training"]["gradient_accumulation_steps"], 3)
        self.assertEqual(rendered["optimizer"]["muon_lr"], 0.003)
        self.assertIsNone(rendered["training"]["time_limit_seconds"])

    def test_rejects_repeat_of_gain_free_unit_qk_norm(self):
        model = dict(load_candidate(self.rules), qk_norm_mode="unit_rms")
        with self.assertRaisesRegex(ValueError, "already tested"):
            validate_candidate(model, self.rules)
        del model["qk_norm_mode"]
        with self.assertRaisesRegex(ValueError, "must set qk_norm_mode explicitly"):
            validate_candidate(model, self.rules)

    def test_trial_manifest_preserves_data_and_mix(self):
        source = {"config": {"seq_len": 4096, "batch_size": 48,
                             "total_train_tokens": 10_000_000_000,
                             "source_mix": {"en": 0.5, "hi": 0.5},
                             "phase_source_mixes": []},
                  "shards": [{"path": "a.bin", "sha256": "a" * 64}],
                  "tokenizer": {"sha256": "b" * 64}, "provenance": {"split": "fixed"}}
        narrowed = trial_manifest(source, self.rules)
        self.assertEqual(narrowed["config"]["total_train_tokens"], 2_500_000_000)
        self.assertEqual(narrowed["config"]["batch_size"], 32)
        self.assertEqual(narrowed["config"]["source_mix"], source["config"]["source_mix"])
        self.assertEqual(narrowed["shards"], source["shards"])
        self.assertEqual(source["config"]["batch_size"], 48)

    def test_grouped_score_and_regression_guardrail(self):
        baseline = {domain: 2.0 for domain in self.rules["domains"]}
        candidate = dict(baseline)
        candidate["en"] = 1.8
        candidate["hi"] = 2.1
        result = compare_losses(candidate, baseline, self.rules)
        self.assertGreater(result["relative_improvement"], 0)
        self.assertIn("hi", result["material_regressions"])
        self.assertAlmostEqual(grouped_losses(baseline)["indic"], 2.0)
        with self.assertRaises(ValueError):
            grouped_losses({"en": 2.0})

    def test_tasks_must_cover_ten_domains_and_keep_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for domain in self.rules["domains"]:
                path = root / f"{domain}.jsonl"
                path.write_text(json.dumps({"prompt": "Question", "reference": "Answer"}) + "\n")
                entries.append({"domain": domain, "task": "qa", "path": path.name,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "role": "development", "answer_only": True,
                                "source": "test-source", "split": "dev"})
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps({"version": 1, "tasks": entries}))
            self.assertEqual(len(validate_task_manifest(manifest, self.rules["domains"])), 10)
            (root / "hi.jsonl").write_text(json.dumps({"prompt": "Changed", "reference": "Answer"}) + "\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_task_manifest(manifest, self.rules["domains"])

    def test_paired_interval_uses_the_same_validation_blocks(self):
        baseline = {domain: {"batches": [{"start": 0, "examples": 2, "tokens": 8,
                                          "loss_sum": 16.0},
                                         {"start": 2, "examples": 2, "tokens": 8,
                                          "loss_sum": 16.0}]}
                    for domain in self.rules["domains"]}
        candidate = {domain: {"batches": [{**batch, "loss_sum": 15.2}
                                         for batch in value["batches"]]}
                     for domain, value in baseline.items()}
        interval = paired_batch_interval(candidate, baseline, self.rules, resamples=100)
        self.assertGreater(interval["low"], 0)
        candidate["hi"]["batches"][0]["start"] = 1
        with self.assertRaisesRegex(ValueError, "selection differs"):
            paired_batch_interval(candidate, baseline, self.rules, resamples=10)

    def test_heldout_evaluator_matches_training_evaluator(self):
        import torch
        from research.evaluate import heldout_batches
        from terminus.model import BitNetLlama
        from terminus.train import evaluate
        from tests.helpers import fixture

        with tempfile.TemporaryDirectory() as directory:
            config, manifest = fixture(directory)
            model = BitNetLlama(config)
            model.eval()
            actual = heldout_batches(model, manifest, Path(directory), torch.device("cpu"), 2)
            expected = evaluate(model, manifest, Path(directory), torch.device("cpu"),
                                torch.float32, 2, 0, 1)
            self.assertAlmostEqual(actual["en"]["loss"], expected["en"]["loss"], places=6)
            self.assertEqual(actual["en"]["tokens"], expected["en"]["tokens"])

    def test_archive_verification_catches_missing_or_changed_logs(self):
        from research.artifacts import create, verify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "events.jsonl").write_text("event\n")
            (root / "checkpoint.pt").write_bytes(b"checkpoint")
            receipt = root / "artifact_receipt.json"
            self.assertEqual(create(root, receipt)["files"], 2)
            copy = Path(directory) / "download"
            copy.mkdir()
            for name in ("events.jsonl", "checkpoint.pt", "artifact_receipt.json"):
                (copy / name).write_bytes((root / name).read_bytes())
            self.assertTrue(verify(copy, receipt)["verified"])
            (copy / "events.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "verification failed"):
                verify(copy, receipt)
            (copy / "events.jsonl").write_text("event\n")
            (copy / "artifact_receipt.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                verify(copy, receipt)


if __name__ == "__main__":
    unittest.main()
