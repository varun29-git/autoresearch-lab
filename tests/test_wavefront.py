"""Wavefront-pipelined accumulation: bitwise parity vs the serial loop.

The pipeline changes only execution interleaving (side-stream forward under
main-stream backward); kernels, per-microstep order and gradient accumulation
order are identical, so a deterministic tiny model must match BITWISE.
"""
import json
import unittest
from pathlib import Path

import torch

from tests.helpers import checkpoint_payload, fixture


def run_wavefront_arm(root, wavefront, updates=3):
    root = Path(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["training"]["device"] = "cuda"
    config["training"]["log_interval"] = 1  # per-update loss records for the parity comparison
    config["training"]["checkpoint_interval"] = 10**9
    config["training"]["validation_interval"] = 10**9
    config["training"]["wavefront_pipeline"] = wavefront
    config_path.write_text(json.dumps(config))
    from terminus.train import run_training

    out = root / ("run_wavefront" if wavefront else "run_serial")
    run_training(str(config_path), str(root / "manifest.json"), str(root), str(out),
                 stop_after_updates=updates)
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()
              if '"train"' in line]
    losses = [event["loss"] for event in events]
    weights = checkpoint_payload(out)["model"]
    return losses, weights


@unittest.skipUnless(torch.cuda.is_available(), "native CUDA validation required")
class WavefrontParityTests(unittest.TestCase):
    def test_wavefront_matches_serial_bitwise(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            fixture(directory)
            losses_s, weights_s = run_wavefront_arm(directory, False)
            losses_w, weights_w = run_wavefront_arm(directory, True)
        self.assertEqual(len(losses_s), 3)
        self.assertEqual(losses_s, losses_w)  # exact same text-recorded values
        for name in weights_s:
            self.assertTrue(torch.equal(weights_s[name], weights_w[name]),
                            f"wavefront parity failed for {name}")

    def test_single_microstep_falls_back_cleanly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            config, manifest = fixture(directory, accumulation=1)
            losses_s, weights_s = run_wavefront_arm(directory, False)
            losses_w, weights_w = run_wavefront_arm(directory, True)
        self.assertEqual(losses_s, losses_w)
        for name in weights_s:
            self.assertTrue(torch.equal(weights_s[name], weights_w[name]))


if __name__ == "__main__":
    unittest.main()
