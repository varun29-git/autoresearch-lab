"""Checks that architecture decisions use learning, stability and cost signals."""

import json
import tempfile
import unittest
from pathlib import Path

from research.signals import compare_checkpoint_diagnostics, compare_signals, summarize_run


def records(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


class ResearchSignalTests(unittest.TestCase):
    def test_learning_curve_speed_memory_and_stability_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline"
            candidate = Path(directory) / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            for run_dir, second_loss, speed, memory in ((baseline, 2.0, 100.0, 80),
                                                         (candidate, 1.8, 80.0, 100)):
                records(run_dir / "events.jsonl", [
                    {"event": "train", "tokens_seen": 100, "tokens_per_second": speed,
                     "peak_memory_bytes": memory, "grad_clip_coefficient": 0.5,
                     "grad_norm_before_clip": 2.0},
                    {"event": "validation", "tokens_seen": 100, "languages": {"en": {"loss": 3.0}}},
                    {"event": "train", "tokens_seen": 200, "tokens_per_second": speed,
                     "peak_memory_bytes": memory, "grad_clip_coefficient": 1.0,
                     "grad_norm_before_clip": 0.5},
                    {"event": "validation", "tokens_seen": 200, "languages": {"en": {"loss": second_loss}}},
                ])
                records(run_dir / "stability.jsonl", [
                    {"tokens_seen": 200, "probes": {
                        "layer_00.residual": {"rms": 4.0, "nonfinite_fraction": 0.0},
                        "layer_00.attention_sample": {"logits": {"max_abs": 10.0},
                                                       "min_entropy": 0.8},
                        "layer_00.fp8_input": {"relative_rms_error": 0.02},
                        "weight.parameter_sample": {"update_to_weight_rms": 0.003}}},
                ])
            base = summarize_run(baseline)
            cand = summarize_run(candidate)
            comparison = compare_signals(cand, base)
            self.assertGreater(comparison["english_loss_curve_auc_improvement"], 0)
            self.assertEqual(comparison["logged_training_speed_ratio"], 0.8)
            self.assertEqual(comparison["peak_memory_ratio"], 1.25)
            self.assertEqual(cand["logged_gradient_clip_fraction"], 0.5)
            self.assertEqual(cand["max_sampled_attention_logit_abs"], 10.0)
            self.assertFalse(cand["numerical_failure_or_sampled_nonfinite"])
            self.assertEqual(cand["train_phases"]["middle"]["logged_updates"], 1)
            self.assertEqual(cand["probe_metrics"]["layer_00.residual.rms"]["last"]["value"], 4.0)
            self.assertTrue(comparison["probe_differences"]["layer_00.residual.rms"]["same_last_sample_tokens"])
            self.assertIsNone(compare_signals(summarize_run(candidate, through_tokens=100),
                                              summarize_run(baseline, through_tokens=100))[
                                                  "english_loss_curve_auc_improvement"])

    def test_nonfinite_sample_and_incident_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            records(run_dir / "stability.jsonl", [
                {"tokens_seen": 5, "probes": {"weight.fp8_input": {"nonfinite_fraction": 0.1}}},
            ])
            records(run_dir / "incidents.jsonl", [{"event": "gradient_failure",
                                                   "state": {"tokens_seen": 5}}])
            result = summarize_run(run_dir)
            self.assertTrue(result["numerical_failure_or_sampled_nonfinite"])
            self.assertEqual(result["incidents"]["gradient_failure"], 1)

    def test_every_named_probe_is_retained_and_missing_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            records(run_dir / "stability.jsonl", [
                {"tokens_seen": 10, "probes": {"layer_00.attention_sample": {
                    "global_layer": True, "key_positions": 128,
                    "query": {"rms": 2.0}, "key": {"rms": 1.0},
                    "logits": {"max_abs": 8.0}, "min_entropy": None}}},
                {"tokens_seen": 20, "probes": {"layer_00.attention_sample": {
                    "query": {"rms": 3.0}, "key": {"rms": 1.5},
                    "logits": {"max_abs": 12.0}, "min_entropy": 0.7}}},
            ])
            result = summarize_run(run_dir)
            self.assertEqual(result["probe_metrics"]["layer_00.attention_sample.query.rms"]["first"],
                             {"tokens_seen": 10, "value": 2.0})
            self.assertEqual(result["probe_metrics"]["layer_00.attention_sample.query.rms"]["last"],
                             {"tokens_seen": 20, "value": 3.0})
            self.assertEqual(result["probe_metrics"]["layer_00.attention_sample.min_entropy"]["missing"], 1)
            self.assertNotIn("layer_00.attention_sample.key_positions", result["probe_metrics"])

    def test_checkpoint_diagnostics_retain_position_and_task_gains_and_regressions(self):
        base = {"position_buckets": {"en": {
                    "0-511": {"loss": 2.0, "tokens": 512},
                    "512-2047": {"loss": 2.0, "tokens": 1536}}},
                "tasks": {"summary": {"per_task": {"en/qa": {"mean": 2.0, "n": 2}}},
                          "records": [{"task": "en/qa", "item_id": "a", "loss": 1.0},
                                      {"task": "en/qa", "item_id": "b", "loss": 3.0}]}}
        cand = {"position_buckets": {"en": {
                    "0-511": {"loss": 1.8, "tokens": 512},
                    "512-2047": {"loss": 2.2, "tokens": 1536}}},
                "tasks": {"summary": {"per_task": {"en/qa": {"mean": 1.75, "n": 2}}},
                          "records": [{"task": "en/qa", "item_id": "a", "loss": 1.5},
                                      {"task": "en/qa", "item_id": "b", "loss": 2.0}]}}
        result = compare_checkpoint_diagnostics(cand, base)
        self.assertGreater(result["position_buckets"]["en"]["0-511"]["relative_improvement"], 0)
        self.assertLess(result["position_buckets"]["en"]["512-2047"]["relative_improvement"], 0)
        items = result["development_tasks"]["per_item"]
        self.assertGreater(items[0]["loss_difference"], 0)
        self.assertLess(items[1]["loss_difference"], 0)


if __name__ == "__main__":
    unittest.main()
