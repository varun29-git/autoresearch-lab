"""CPU regression tests for complete-trial comparison, selection and transfer."""

import json
from pathlib import Path
import tempfile
import unittest

from production.kimi_h100_5b import campaign
from production.kimi_h100_5b.selection import (SelectionError, aggregate_nll, compare_trials,
                                               load_trial_summary, transfer_confirmation)

MARGINS = dict(campaign.SELECTION_MARGINS)


def write_trial(root, trial, *, final_nll, pre_decay_nll, warmup_nll, complete=True,
                failed=False, domains=("en", "code"), missing_boundary=None):
    trial_dir = Path(root) / "trials" / trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    events = []
    boundaries = {238: warmup_nll, 1907: pre_decay_nll, 2384: final_nll}
    for update, nll in boundaries.items():
        if update == missing_boundary:
            continue
        languages = {domain: {"loss": nll + 0.001 * index, "tokens": 100_000,
                              "perplexity": None}
                     for index, domain in enumerate(domains)}
        events.append({"event": "validation", "updates": update, "tokens_seen": 0,
                       "languages": languages})
    if complete:
        events.append({"event": "complete", "updates": 2384,
                       "tokens_seen": 4_999_610_368, "next_micro_step": 0})
    (trial_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    if failed:
        (trial_dir / "incidents.jsonl").write_text(
            json.dumps({"event": "nonfinite_loss", "error": "nan"}) + "\n")
    return trial_dir


def full_matrix(root, final_by_trial):
    pre = {"H": 2.20, "R0": 2.19, "R1": 2.21, "R2": 2.22, "R3": 2.23}
    for trial, final in final_by_trial.items():
        write_trial(root, trial, final_nll=final, pre_decay_nll=pre[trial], warmup_nll=2.8)


class AggregateTests(unittest.TestCase):
    def test_token_weighted_aggregate(self):
        languages = {"en": {"loss": 2.0, "tokens": 300}, "hi": {"loss": 3.0, "tokens": 100}}
        self.assertAlmostEqual(aggregate_nll(languages), 2.25)
        with self.assertRaises(SelectionError):
            aggregate_nll({})


class SelectionTests(unittest.TestCase):
    def test_clear_winner_is_selected_provisionally(self):
        with tempfile.TemporaryDirectory() as directory:
            full_matrix(directory, {"H": 2.10, "R0": 2.09, "R1": 2.12, "R2": 2.13, "R3": 2.14})
            summaries = {trial: load_trial_summary(Path(directory) / "trials" / trial)
                         for trial in campaign.TRIAL_ORDER}
            report = compare_trials(summaries, MARGINS)
            self.assertEqual(report["status"], "selected")
            self.assertEqual(report["selection"], "R0")
            self.assertTrue(report["provisional"])
            self.assertAlmostEqual(report["superiority_gap"], 0.01)

    def test_tiny_advantage_is_a_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            full_matrix(directory, {"H": 2.101, "R0": 2.100, "R1": 2.12, "R2": 2.13, "R3": 2.14})
            summaries = {trial: load_trial_summary(Path(directory) / "trials" / trial)
                         for trial in campaign.TRIAL_ORDER}
            report = compare_trials(summaries, MARGINS)
            self.assertEqual(report["status"], "tie")
            self.assertEqual(report["selection"], "R0")
            self.assertTrue(report["provisional"])
            self.assertIn("not a proven ranking", report["note"])

    def test_failed_or_incomplete_arms_block_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            full_matrix(directory, {"H": 2.10, "R0": 2.09, "R1": 2.12, "R2": 2.13, "R3": 2.14})
            write_trial(directory, "R3", final_nll=2.14, pre_decay_nll=2.23, warmup_nll=2.8,
                        failed=True)
            write_trial(directory, "R2", final_nll=2.13, pre_decay_nll=2.22, warmup_nll=2.8,
                        missing_boundary=1907)
            summaries = {trial: load_trial_summary(Path(directory) / "trials" / trial)
                         for trial in campaign.TRIAL_ORDER}
            report = compare_trials(summaries, MARGINS)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("R2", report["excluded"])
            self.assertIn("R3", report["excluded"])
            self.assertIn("no winner", report["note"])

    def test_stopped_trial_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            write_trial(directory, "H", final_nll=2.1, pre_decay_nll=2.2, warmup_nll=2.8,
                        complete=False)
            summary = load_trial_summary(Path(directory) / "trials" / "H")
            self.assertFalse(summary["completed"])
            self.assertEqual(summary["missing_boundaries"], [])


class TransferTests(unittest.TestCase):
    def test_transfer_applied_once_with_unchanged_groups(self):
        record = transfer_confirmation({"status": "selected", "selection": "R3"})
        self.assertEqual(record["transferred_adamw_lr"], 0.00085)
        self.assertEqual(record["transferred_muon_lr"], 0.0082)
        self.assertEqual(record["transfer_multiplier"], "768/2304")
        self.assertEqual(record["unchanged"]["norm_lr"], 0.0003)
        self.assertEqual(record["unchanged"]["qk_norm_lr"], 0.0001)
        record = transfer_confirmation({"status": "tie", "selection": "H"})
        self.assertEqual(record["transferred_adamw_lr"], 0.002)

    def test_transfer_requires_completed_comparison(self):
        with self.assertRaises(SelectionError):
            transfer_confirmation({"status": "incomplete"})
        with self.assertRaises(SelectionError):
            transfer_confirmation({"status": "selected", "selection": "R9"})


if __name__ == "__main__":
    unittest.main()
