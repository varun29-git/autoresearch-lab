"""Campaign invariants that prevent architecture trials from changing the yardstick."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.loop import trial_config, trial_manifest
from research.protocol import (compare_losses, grouped_losses, load_candidate,
                               policy, validate_task_manifest)


class ResearchProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rules = policy()

    def test_baseline_candidate_and_fixed_training_policy(self):
        model = load_candidate(self.rules)
        rendered = trial_config(model, self.rules)
        self.assertEqual(rendered["model"], model)
        self.assertEqual(rendered["training"]["warmup_tokens"], 250_000_000)
        self.assertEqual(rendered["training"]["decay_tokens"], 500_000_000)
        self.assertEqual(rendered["training"]["gradient_accumulation_steps"], 3)
        self.assertEqual(rendered["optimizer"]["muon_lr"], 0.003)
        self.assertIsNone(rendered["training"]["time_limit_seconds"])

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


if __name__ == "__main__":
    unittest.main()
