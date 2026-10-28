"""CPU regression tests for the external launch packet generator."""

import json
from pathlib import Path
import tempfile
import unittest

from production.kimi_h100_5b import campaign
from production.kimi_h100_5b.packet import create_packet


class PacketTests(unittest.TestCase):
    def test_packet_layout_gates_and_frozen_margins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qual"
            result = create_packet(root, tokenizer_path=campaign.REPOSITORY
                                   / "tokenizer/tokenizer-65k.json")
            self.assertEqual(result["status"], "prepared_not_launched")
            self.assertTrue(result["forecast_fits_arithmetic_cap"])
            for name in ("campaign.json", "forecast.json", "identity/source.json",
                         "evaluation/margins.json", "launch/LAUNCH.md"):
                self.assertTrue((root / name).is_file(), name)
            campaign_doc = json.loads((root / "campaign.json").read_text())
            self.assertEqual(campaign_doc["trials_in_order"], ["H", "R0", "R1", "R2", "R3"])
            self.assertEqual(campaign_doc["cap_hours"], 50.0)
            self.assertIsNone(campaign_doc["human_launch_record"])
            margins = json.loads((root / "evaluation/margins.json").read_text())
            self.assertEqual(margins["status"], "frozen_before_trials")
            self.assertEqual(margins["aggregate_superiority_nll"], 0.005)
            tokenizer = json.loads((root / "identity/tokenizer.json").read_text())
            self.assertEqual(len(tokenizer["sha256"]), 64)
            source = json.loads((root / "identity/source.json").read_text())
            self.assertIn("production/kimi_h100_5b/runner.py", source["source"])
            self.assertIn("train.py", source["source"])
            self.assertEqual(source["git"]["commit"],
                             campaign_doc["git"]["commit"])
            for gate in ("offline_contract", "data_evaluation", "runtime_native",
                         "execution_fit", "recovery_storage", "budget_launch"):
                doc = json.loads((root / "gates" / f"{gate}.json").read_text())
                self.assertEqual(doc["status"], "blocked_by_missing_input")
                self.assertTrue(doc["remaining"])
            for trial in campaign.TRIAL_ORDER:
                self.assertTrue((root / "trials" / trial).is_dir())
            launch = (root / "launch/LAUNCH.md").read_text()
            self.assertIn("never reset", launch)
            self.assertIn("Process exit does not stop billing", launch)

    def test_packet_refuses_overwrite_and_in_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qual"
            create_packet(root)
            with self.assertRaises(FileExistsError):
                create_packet(root)
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            create_packet(campaign.REPOSITORY / "qual-should-not-exist")


if __name__ == "__main__":
    unittest.main()
