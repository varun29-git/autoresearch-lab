"""CPU regression tests for the qualification config builder/validator."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from production.kimi_h100_5b import campaign
from production.kimi_h100_5b.builder import (build_confirmation_config, build_trial_config,
                                             main as builder_main,
                                             validate_qualification_config)


def proxy_manifest(root, *, total=4_999_610_368, batch=4):
    path = Path(root) / "manifest.json"
    path.write_text(json.dumps({
        "config": {"seq_len": 4096, "batch_size": batch, "total_train_tokens": total,
                   "validation_tokens": 50_000_000},
        "shards": [], "tokenizer": {}, "provenance": {}}))
    return path


def write_config(root, name, payload):
    path = Path(root) / name
    path.write_text(json.dumps(payload))
    return path


class BuilderTests(unittest.TestCase):
    def test_matrix_and_fully_resolved_defaults(self):
        configs = {trial: build_trial_config(trial, microbatch=4)
                   for trial in campaign.TRIAL_ORDER}
        self.assertEqual([c["optimizer"]["adamw_lr"] for c in configs.values()],
                         [0.006, 0.009, 0.00255, 0.009, 0.00255])
        self.assertEqual([c["optimizer"]["muon_lr"] for c in configs.values()],
                         [0.017, 0.017, 0.017, 0.0082, 0.0082])
        config = configs["H"]
        self.assertEqual(len(config["model"]), 46)
        self.assertEqual(len(config["training"]), 25)
        self.assertEqual(len(config["optimizer"]), 19)
        self.assertEqual(config["training"]["gradient_accumulation_steps"], 128)
        self.assertEqual(config["training"]["warmup_tokens"], 499_122_176)
        self.assertEqual(config["training"]["decay_tokens"], 1_000_341_504)
        self.assertEqual(config["training"]["log_interval"], 1)
        self.assertEqual(config["training"]["schedule"], "wsd")
        for key in ("dim", "ffn_dim", "n_heads", "n_kv_heads"):
            self.assertEqual(config["model"][key], campaign.PROXY_MODEL_CHANGES[key])
        for key, value in campaign.FIXED_OPTIMIZER.items():
            self.assertEqual(config["optimizer"][key], value)

    def test_transfer_applied_exactly_once(self):
        self.assertEqual(build_confirmation_config("H", microbatch=1)["optimizer"]["adamw_lr"], 0.002)
        self.assertEqual(build_confirmation_config("R0", microbatch=1)["optimizer"]["adamw_lr"], 0.003)
        self.assertEqual(build_confirmation_config("R3", microbatch=1)["optimizer"]["adamw_lr"], 0.00085)
        confirmation = build_confirmation_config("R2", microbatch=2)
        self.assertEqual(confirmation["optimizer"]["muon_lr"], 0.0082)
        self.assertEqual(confirmation["model"]["dim"], 2304)
        self.assertEqual(confirmation["training"]["warmup_tokens"], 100_663_296)
        self.assertEqual(confirmation["training"]["gradient_accumulation_steps"], 256)

    def test_unknown_trials_and_bad_microbatch_rejected(self):
        for call in (lambda: build_trial_config("R4", microbatch=4),
                     lambda: build_confirmation_config("h", microbatch=4),
                     lambda: campaign.transferred_adamw_lr("R5")):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaisesRegex(ValueError, "multiple"):
            build_trial_config("H", microbatch=3)
        with self.assertRaisesRegex(ValueError, "positive"):
            build_trial_config("H", microbatch=0)

    def test_validate_proxy_contract_and_parameter_count(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = proxy_manifest(directory)
            config = write_config(directory, "H.json", build_trial_config("H", microbatch=4))
            receipt = validate_qualification_config(config, manifest, trial_id="H")
            self.assertEqual(receipt["status"], "offline_contract_checks_passed")
            self.assertEqual(receipt["parameters"]["total"], 194_329_390)
            self.assertEqual(receipt["batch"]["global_batch_tokens"], 2_097_152)
            self.assertEqual(receipt["schedule"]["boundary_updates"], [238, 1907, 2384])
            self.assertTrue(receipt["unverified"])

    def test_validate_confirmation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = proxy_manifest(directory, total=1_006_632_960, batch=1)
            config = write_config(directory, "confirm.json",
                                  build_confirmation_config("R0", microbatch=1))
            receipt = validate_qualification_config(config, manifest, trial_id="R0",
                                                    confirmation=True)
            self.assertEqual(receipt["parameters"]["total"], 1_446_478_126)
            self.assertEqual(receipt["schedule"]["boundary_updates"], [48, 384, 480])

    def test_validate_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = proxy_manifest(directory)
            base = build_trial_config("H", microbatch=4)
            cases = []
            changed = json.loads(json.dumps(base))
            changed["optimizer"]["norm_lr"] = 0.0004
            cases.append(("optimizer differs", changed))
            changed = json.loads(json.dumps(base))
            changed["model"]["ffn_dim"] = 2048
            cases.append(("geometry", changed))
            changed = json.loads(json.dumps(base))
            changed["optimizer"]["muon_lr"] = 0.017  # R0 optimizer with H trial id is fine;
            changed["optimizer"]["adamw_lr"] = 0.009
            cases.append(("frozen trial matrix", changed))
            changed = json.loads(json.dumps(base))
            changed["training"]["decay_tokens"] = 999_999_999
            cases.append(("WSD", changed))
            changed = json.loads(json.dumps(base))
            changed["training"]["checkpoint_interval"] = 50
            cases.append(("cadence", changed))
            changed = json.loads(json.dumps(base))
            changed["training"]["cuda_graphs"] = True
            cases.append(("eager", changed))
            changed = json.loads(json.dumps(base))
            changed["training"]["gradient_accumulation_steps"] = 64
            cases.append(("global batch", changed))
            for fragment, payload in cases:
                path = write_config(directory, "drift.json", payload)
                with self.subTest(fragment=fragment), self.assertRaisesRegex(ValueError, fragment):
                    validate_qualification_config(path, manifest, trial_id="H")

    def test_validate_rejects_wrong_world_size_horizon_and_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_config(directory, "H.json", build_trial_config("H", microbatch=4))
            manifest = proxy_manifest(directory)
            with self.assertRaisesRegex(ValueError, "world size 1"):
                validate_qualification_config(config, manifest, trial_id="H", world_size=2)
            with self.assertRaisesRegex(ValueError, "horizon"):
                validate_qualification_config(
                    config, proxy_manifest(directory, total=5_000_000_000), trial_id="H")
            with self.assertRaisesRegex(ValueError, "unknown"):
                validate_qualification_config(config, manifest, trial_id="R9")
            wrong_batch = proxy_manifest(directory, batch=2)
            with self.assertRaisesRegex(ValueError, "global batch"):
                validate_qualification_config(config, wrong_batch, trial_id="H")

    def test_validate_rejects_in_repo_paths(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            validate_qualification_config(campaign.REPOSITORY / "x.json",
                                          campaign.REPOSITORY / "y.json", trial_id="H")

    def test_cli_build_validate_and_output_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = proxy_manifest(root)
            argv = ["builder", "--trial", "R1", "--microbatch", "4",
                    "--manifest", str(manifest), "--output", str(root / "R1.json")]
            with unittest.mock.patch.object(sys, "argv", argv):
                self.assertEqual(builder_main(), 0)
            self.assertTrue((root / "R1.json").is_file())
            with unittest.mock.patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    builder_main()
            argv = ["builder", "--validate", "--config", str(root / "R1.json"),
                    "--trial-id", "R1", "--manifest", str(manifest)]
            with unittest.mock.patch.object(sys, "argv", argv):
                self.assertEqual(builder_main(), 0)
            argv = ["builder", "--trial", "H", "--microbatch", "4",
                    "--manifest", str(manifest),
                    "--output", str(campaign.REPOSITORY / "in-repo.json")]
            with unittest.mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "outside the repository"):
                    builder_main()


if __name__ == "__main__":
    unittest.main()
