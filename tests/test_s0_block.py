import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from speed_research.measurement import measure_block
from terminus.model import BitNetLlama
from terminus.train import prepare_training, run_training
from tests.helpers import checkpoint_payload, fixture

ROOT = Path(__file__).resolve().parent.parent


class S0BlockTests(unittest.TestCase):
    def test_measured_block_matches_plain_trainer_bitwise(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset", examples=9, batch_size=2, accumulation=2)
            arguments = (root / "dataset/config.json", root / "dataset/manifest.json", root / "dataset")
            reference = run_training(*arguments, root / "run_a", stop_after_updates=2)
            self.assertEqual(reference, {"next_micro_step": 4, "tokens_seen": 64, "updates": 2})
            block_path = root / "block.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/s0_block.py"),
                            "--config", str(arguments[0]), "--manifest", str(arguments[1]),
                            "--data-root", str(arguments[2]), "--output-dir", str(root / "run_b"),
                            "--warmup-updates", "1", "--measured-updates", "1",
                            "--save-checkpoint",
                            "--out", str(block_path)],
                           check=True, capture_output=True, text=True, timeout=300)
            block = json.loads(block_path.read_text())
            self.assertEqual(block["block"]["updates"], 1)
            self.assertEqual(block["block"]["valid_tokens"], 32)
            self.assertTrue(block["block"]["finite"])
            self.assertTrue(block["block"]["completed"])
            self.assertEqual(block["counters_before"], {"updates": 1, "tokens_seen": 32})
            self.assertEqual(block["counters_after"], reference)
            expected, actual = checkpoint_payload(root / "run_a"), checkpoint_payload(root / "run_b")
            self.assertEqual(expected["state"], actual["state"])
            for name, weights in expected["model"].items():
                torch.testing.assert_close(actual["model"][name], weights, rtol=0, atol=0)
            for parameter_id, state in expected["optimizer"]["state"].items():
                for key, value in state.items():
                    restored = actual["optimizer"]["state"][parameter_id][key]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(restored, value, rtol=0, atol=0)
                    else:
                        self.assertEqual(restored, value)

    def test_profiled_updates_produce_trace_and_count_tokens(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset", examples=13, batch_size=2, accumulation=2)
            block_path = root / "block.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/s0_block.py"),
                            "--config", str(root / "dataset/config.json"),
                            "--manifest", str(root / "dataset/manifest.json"),
                            "--data-root", str(root / "dataset"), "--output-dir", str(root / "run"),
                            "--warmup-updates", "1", "--profile-updates", "1",
                            "--measured-updates", "1", "--out", str(block_path)],
                           check=True, capture_output=True, text=True, timeout=300)
            record = json.loads(block_path.read_text())
            self.assertEqual(record["profile_updates"], 1)
            self.assertEqual(record["counters_after"]["updates"], 3)
            self.assertEqual(record["block"]["valid_tokens"], 32)
            self.assertGreater(Path(record["profile_trace"]).stat().st_size, 0)
            self.assertIsNone(record["final_checkpoint"])
            self.assertFalse((root / "run" / "checkpoints").exists())

    def test_nonfinite_update_fails_the_block(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset")
            original = BitNetLlama.forward

            def broken_forward(model, *args, **kwargs):
                output = original(model, *args, **kwargs)
                output["loss_sum"] = output["loss_sum"] * float("nan")
                return output

            output_dir = root / "run"
            with prepare_training(root / "dataset/config.json", root / "dataset/manifest.json",
                                  root / "dataset", output_dir) as workload:
                with patch.object(BitNetLlama, "forward", broken_forward):
                    with self.assertRaises(FloatingPointError):
                        measure_block(workload.complete_update, updates=1,
                                      synchronize=lambda: None, read_memory=lambda: 1)
            incidents = [json.loads(line) for line in (output_dir / "incidents.jsonl").read_text().splitlines()]
            self.assertEqual(incidents[0]["event"], "nonfinite_loss")

    def test_operational_overrides_reject_semantic_fields(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            fixture(root / "dataset")
            with self.assertRaisesRegex(ValueError, "not operational"):
                with prepare_training(root / "dataset/config.json", root / "dataset/manifest.json",
                                      root / "dataset", root / "run",
                                      operational_overrides={"warmup_tokens": 5}):
                    pass

    def test_quick_timing_override_is_restricted_to_single_cuda_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "dataset")
            with self.assertRaisesRegex(ValueError, "one CUDA GPU"):
                with prepare_training(root / "dataset/config.json", root / "dataset/manifest.json",
                                      root / "dataset", root / "run", research_timing_only=True):
                    pass


if __name__ == "__main__":
    unittest.main()
