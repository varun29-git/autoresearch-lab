"""CPU-only checks for the optional matched QAT runner transaction."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.loop import _qat_probe_locked, latest_state, qat_probe_tokens, status
from research.protocol import load_candidate, policy, sha256


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class QATProbeRunnerTests(unittest.TestCase):
    def test_probe_runs_conversion_eval_before_training_and_records_receipt(self):
        rules = policy()
        model = load_candidate(rules)
        parent_tokens = rules["full_tokens"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "trials/baseline"
            dense_run = parent / "run"
            write(root / "source.json", {"config": {"seq_len": rules["max_seq_len"],
                                          "batch_size": 32, "total_train_tokens": 3_000_000_000,
                                          "source_mix": {"en": 1.0}, "phase_source_mixes": []},
                                         "shards": []})
            write(parent / "metadata.json", {"commit": "abc", "policy_sha256": "locked"})
            write(parent / "eval_full.json", {"tokens_seen": parent_tokens,
                                               "heldout": {name: {"loss": 2.0} for name in rules["domains"]}})
            write(dense_run / "checkpoints/latest.json", {"state": {"tokens_seen": parent_tokens,
                                                                    "updates": parent_tokens // (rules["microbatch"] *
                                                                                                 rules["max_seq_len"] *
                                                                                                 rules["gradient_accumulation_steps"]),
                                                                    "next_micro_step": parent_tokens // (rules["microbatch"] *
                                                                                                           rules["max_seq_len"])}})
            full_record = {"trial_id": "baseline", "stage": "full", "outcome": "completed",
                           "evaluation_sha256": sha256(parent / "eval_full.json")}
            settings = {"manifest": str(root / "source.json"), "data_root": str(root),
                        "task_manifest": str(root / "tasks.json"), "baseline_trial": "baseline",
                        "tag": "test"}
            commands = []

            def fake_run(command, log_path, timeout):
                commands.append(log_path.stem)
                log_path.write_text("mock subprocess\n")
                module = command[2]
                def argument(name):
                    return Path(command[command.index(name) + 1])
                if module == "terminus.preflight":
                    write(argument("--output"), {"ok": True})
                elif module == "terminus.transition":
                    run_dir = argument("--output-dir")
                    write(run_dir / "transition_receipt.json", {"event": "dense_to_qat_transition"})
                    write(run_dir / "checkpoints/latest.json", {"state": {"tokens_seen": parent_tokens}})
                elif module == "terminus.train":
                    run_dir = argument("--output-dir")
                    write(run_dir / "checkpoints/latest.json",
                          {"state": {"tokens_seen": parent_tokens + qat_probe_tokens(rules)}})
                elif module == "research.evaluate":
                    run_dir = argument("--run-dir")
                    write(argument("--output"), {"tokens_seen": latest_state(run_dir)["tokens_seen"],
                                                  "heldout": {name: {"loss": 2.1,
                                                                     "batches": [{"start": 0, "examples": 1,
                                                                                  "tokens": 1, "loss_sum": 2.1}]}
                                                              for name in rules["domains"]},
                                                  "tasks": {}})
                else:
                    raise AssertionError(module)
                return 0, 1.0, False

            with patch("research.loop.run_command", side_effect=fake_run):
                record = _qat_probe_locked(rules, settings, {"policy_sha256": "locked"}, root,
                                           "baseline", "abc", model, 200_290_816,
                                           [full_record], 1000, "Weight quantization may alter the ranking; compare conversion losses.")
            self.assertEqual(commands, ["preflight", "transition", "conversion_eval", "train", "evaluation"])
            self.assertEqual(record["outcome"], "completed")
            self.assertEqual(record["tokens_seen"], parent_tokens + qat_probe_tokens(rules))
            self.assertEqual(record["stage"], "qat_probe")
            self.assertEqual(len((root / "ledger.jsonl").read_text().splitlines()), 1)
            with patch("research.loop.read_lock", return_value=(rules, settings, {}, root)):
                report = status(root / "unused_campaign.json")
            self.assertEqual(report["comparisons"][0]["stage"], "qat_probe")
            self.assertIn("relative_qat_resilience", report["comparisons"][0])


if __name__ == "__main__":
    unittest.main()
