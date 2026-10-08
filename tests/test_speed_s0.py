import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.speed_s0 import (load_hashed_json, load_lock, phase_model_config,
                               phase_training_config, run_plan, validate_plan, verify_manifest_identity)
from research.azure_speed_worker import run_cycle
from speed_research.protocol import file_digest, make_lock, read_json
from tests.helpers import fixture


def draft_campaign(root):
    campaign = read_json(Path(__file__).resolve().parent.parent /
                         "speed_research/campaign.template.json")
    campaign.update({
        "status": "ready",
        "campaign_id": "stack-004-speed-test",
        "research_gpu_hour_cap": 10.0,
        "trial_gpu_hour_cap": 2.0,
        "research_hardware": "test H100 NVL",
        "production_hardware": "test 32x H100",
        "production_topology": "test topology",
        "environment_digest": "a" * 64,
        "quality_protocol": "test protocol",
        "qat_token_fraction": 0.4,
        "global_tokens_per_update": 524288,
        "data_manifest_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "optimizer_recipe_sha256": "d" * 64,
        "schedule_sha256": "e" * 64,
        "dense_checkpoint_sha256": "f" * 64,
        "qat_checkpoint_sha256": "0" * 64,
        "quality_protocol_sha256": "1" * 64,
    })
    return campaign


class PlanTests(unittest.TestCase):
    def test_validate_plan(self):
        self.assertEqual(validate_plan({"warmup_updates": 2, "measured_updates": 6,
                                        "phases": ["dense", "qat"]}),
                         {"warmup_updates": 2, "measured_updates": 6,
                          "profile_updates": 0, "phases": ["dense", "qat"]})
        for bad in ({"warmup_updates": -1, "measured_updates": 6, "phases": ["dense"]},
                    {"warmup_updates": 1, "measured_updates": 0, "phases": ["dense"]},
                    {"warmup_updates": 1, "measured_updates": 6, "phases": []},
                    {"warmup_updates": 1, "measured_updates": 6, "phases": ["dense", "dense"]},
                    {"warmup_updates": 1, "measured_updates": 6, "phases": ["inference"]},
                    {"warmup_updates": 1, "measured_updates": 6, "phases": ["dense"], "extra": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_plan(bad)

    def test_short_trial_starts_without_ten_minute_reserve_or_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "dataset")
            speed_root = root / "speed"
            speed_root.mkdir()
            (speed_root / "manifest.json").write_bytes((root / "dataset/manifest.json").read_bytes())
            (speed_root / "campaign.lock.sha256").write_text('{"sha256":"stub"}')
            plan = speed_root / "plan.json"
            plan.write_text('{"warmup_updates":1,"measured_updates":2,"phases":["dense"]}')
            recipe = {"kind": "adamw"}
            schedule = {"seed": 1337, "warmup_tokens": 0, "final_lr_fraction": .1,
                        "schedule": "cosine", "decay_tokens": None, "decay_style": "cosine",
                        "grad_clip": 1.0}
            calls = []

            def pretend_run(argv, **kwargs):
                calls.append((argv, kwargs))
                out = Path(argv[argv.index("--out") + 1])
                out.write_text('{"block":{"valid_tokens":32,"seconds":1}}')

            with patch("research.speed_s0.load_lock", return_value=draft_campaign(root)), \
                    patch("research.speed_s0.load_hashed_json", side_effect=[recipe, schedule]), \
                    patch("research.speed_s0.verify_manifest_identity",
                          return_value=verify_manifest_identity(
                              speed_root / "manifest.json", file_digest(speed_root / "manifest.json"))):
                result = run_plan(speed_root=speed_root, data_root=root / "dataset",
                                  results_root=root / "results", cycle_id="quick", plan_path=plan,
                                  budget_seconds=60, command=pretend_run)
            self.assertEqual(result["phases"]["dense"]["status"], "completed")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("preflight", " ".join(calls[0][0]))
            self.assertIn("--measured-updates", calls[0][0])


class WorkerShutdownTests(unittest.TestCase):
    def test_deallocation_follows_verified_archive_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            plan.write_text('{"warmup_updates":1,"measured_updates":2,"phases":["dense"]}')
            args = SimpleNamespace(results_root=str(root / "results"), cycle_id="quick",
                archive_only=False, speed_root=str(root), data_root=str(root), s0_plan=str(plan),
                trial_budget_seconds=60, blob_account="test", blob_container="test",
                blob_prefix="speed/test", verification_parent=str(root / "verify"),
                unit="", resource_group="test", vm_name="test")
            calls = []
            def fake_run(*parts):
                calls.append(parts)
                return ""
            with patch("research.azure_speed_worker.shutil.which", return_value=None), \
                    patch("research.azure_speed_worker.run_plan", return_value={"status": "completed"}):
                result = run_cycle(args, run=fake_run, archive=lambda *a, **k: {"verified": True})
            self.assertTrue(result["deallocation_requested"])
            self.assertEqual(sum(parts[:3] == ("az", "vm", "deallocate") for parts in calls), 1)

            calls.clear()
            with patch("research.azure_speed_worker.shutil.which", return_value=None), \
                    patch("research.azure_speed_worker.run_plan", return_value={"status": "completed"}):
                with self.assertRaisesRegex(RuntimeError, "archive failed"):
                    run_cycle(args, run=fake_run,
                              archive=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("archive failed")))
            self.assertFalse(any(parts[:3] == ("az", "vm", "deallocate") for parts in calls))


class LockTests(unittest.TestCase):
    def test_lock_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = make_lock(draft_campaign(Path(directory)))
            path = Path(directory) / "campaign.lock.json"
            path.write_text(json.dumps(lock))
            campaign = load_lock(path, file_digest(path))
            self.assertEqual(campaign["campaign_id"], "stack-004-speed-test")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_lock(path, "0" * 64)
            lock["campaign"]["trial_gpu_hour_cap"] = 99.0
            path.write_text(json.dumps(lock))
            with self.assertRaises(ValueError):
                load_lock(path, file_digest(path))

    def test_hashed_sidecar_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            path.write_text(json.dumps({"schedule": "cosine"}))
            self.assertEqual(load_hashed_json(path, file_digest(path), "schedule"),
                             {"schedule": "cosine"})
            with self.assertRaisesRegex(ValueError, "schedule"):
                load_hashed_json(path, "0" * 64, "schedule")


class DataGateTests(unittest.TestCase):
    def test_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "dataset")
            manifest_path = root / "dataset/manifest.json"
            manifest = verify_manifest_identity(manifest_path, file_digest(manifest_path))
            self.assertEqual(manifest.config.seq_len, 8)
            with self.assertRaisesRegex(ValueError, "manifest"):
                verify_manifest_identity(manifest_path, "0" * 64)


class PhaseConfigTests(unittest.TestCase):
    def test_phase_configs_from_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "dataset")  # batch 2, seq 8 -> 16 tokens per microstep
            manifest = verify_manifest_identity(root / "dataset/manifest.json",
                                                file_digest(root / "dataset/manifest.json"))
            campaign = draft_campaign(root)
            campaign["global_tokens_per_update"] = 16 * 64
            schedule = {"seed": 1337, "warmup_tokens": 300000000, "final_lr_fraction": 0.1,
                        "schedule": "cosine", "decay_tokens": None, "decay_style": "cosine",
                        "grad_clip": 1.0}
            dense = phase_training_config(campaign, schedule, manifest, "dense")
            self.assertEqual(dense["gradient_accumulation_steps"], 64)
            self.assertEqual(dense["quantization_warmup_tokens"], 0)
            self.assertFalse(dense["development_mode"])
            self.assertEqual(phase_model_config(campaign, "dense")["weight_mode"], "dense")
            self.assertEqual(phase_model_config(campaign, "qat")["weight_mode"], "ternary")
            self.assertEqual(phase_model_config(campaign, "qat")["n_layers"], 25)
            campaign["global_tokens_per_update"] = 16 * 64 + 8
            with self.assertRaisesRegex(ValueError, "whole number"):
                phase_training_config(campaign, schedule, manifest, "dense")


if __name__ == "__main__":
    unittest.main()
