"""No-cloud tests for the H100 one-experiment lifecycle."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.azure_cycle import (ForeignWorkloadError, budget_state, config,
                                  fetch_archive, reconcile_cycle, repair_cycle,
                                  run_cycle as controller_cycle)
from research.azure_worker import archive_and_verify, run_cycle
from research.protocol import sha256


class AzureLifecycleTests(unittest.TestCase):
    def test_budget_counts_vm_on_time_and_requires_archive_proof(self):
        events = [{"event": "start_requested", "cycle_id": "a"},
                  {"event": "vm_started", "cycle_id": "a", "time_unix": 100},
                  {"event": "vm_deallocated", "cycle_id": "a", "time_unix": 3700}]
        state = budget_state(events, 12)
        self.assertEqual(state["spent_vm_hours_estimate"], 1)
        self.assertEqual(state["unconfirmed_archives"], ["a"])
        events.append({"event": "archive_confirmed", "cycle_id": "a"})
        self.assertEqual(budget_state(events, 12)["unconfirmed_archives"], [])
        events.extend([{"event": "start_requested", "cycle_id": "b"},
                       {"event": "start_aborted", "cycle_id": "b"}])
        self.assertEqual(budget_state(events, 12)["unresolved_start_requests"], [])

    def test_config_rejects_ephemeral_result_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = {"resource_group": "test-resource-group", "vm_name": "h100",
                     "ssh_user": "test-user", "remote_repo": "/home/test-user/ar",
                     "remote_python": "/home/test-user/venv/bin/python",
                     "remote_campaign": "/mnt/ar/campaign.json",
                     "remote_data_root": "/mnt/ar/data",
                     "remote_results_root": "/mnt/ar/results",
                     "persistent_mount": "/mnt", "blob_account": "test-storage-account",
                     "blob_container": "terminus-autoresearch", "blob_prefix": "architecture",
                     "verification_parent": "/mnt/ar/verify",
                     "local_control_root": directory, "max_vm_hours": 12}
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "temporary resource disks"):
                config(path)

    def test_example_paths_cannot_start_a_paid_vm(self):
        with patch("research.azure_cycle.assert_clean_git") as git:
            with self.assertRaisesRegex(ValueError, "replace Azure-cycle example paths"):
                controller_cycle({"remote_campaign": "/srv/YOUR_PERSISTENT_DISK/campaign.json",
                                  "local_control_root": "/absolute/path/outside/git"},
                                 trial_id="baseline", stage="screen")
            git.assert_not_called()

    def test_archive_downloads_exact_files_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trial"
            root.mkdir()
            (root / "logs").mkdir()
            (root / "logs" / "train.log").write_text("all logs")
            (root / "checkpoint.pt").write_bytes(b"checkpoint")
            cloud = {}
            corrupt = False

            def fake_run(*args):
                nonlocal corrupt
                if args[3] == "upload-batch":
                    prefix = args[args.index("--destination-path") + 1]
                    source = Path(args[args.index("--source") + 1])
                    for path in source.rglob("*"):
                        if path.is_file():
                            cloud[f"{prefix}/{path.relative_to(source)}"] = path.read_bytes()
                    if corrupt:
                        cloud[f"{prefix}/checkpoint.pt"] = b"damaged"
                    return ""
                if args[3] == "list":
                    prefix = args[args.index("--prefix") + 1]
                    return json.dumps(sorted(name for name in cloud if name.startswith(prefix)))
                if args[3] == "download":
                    name = args[args.index("--name") + 1]
                    Path(args[args.index("--file") + 1]).write_bytes(cloud[name])
                    return ""
                raise AssertionError(args)

            result = archive_and_verify(root, account="account", container="container",
                                        blob_prefix="trial/one", verification_parent=Path(directory),
                                        run=fake_run)
            self.assertTrue(result["verified"])
            self.assertEqual(result["files"], 2)
            corrupt = True
            with self.assertRaisesRegex(ValueError, "verification failed"):
                archive_and_verify(root, account="account", container="container",
                                   blob_prefix="trial/two", verification_parent=Path(directory),
                                   run=fake_run)

    def test_vm_deallocation_only_follows_verified_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(campaign=root / "campaign.json", trial_id="trial-1",
                                   stage="screen", qat_rationale=None, archive_only=False,
                                   trial_budget_seconds=3600,
                                   cycle_id="cycle-1", unit="terminus-ar-cycle-1",
                                   blob_account="account", blob_container="container",
                                   blob_prefix="architecture/cycle-1",
                                   verification_parent=root / "verify",
                                   resource_group="test-resource-group", vm_name="h100")
            calls = []

            def fake_run(*command):
                calls.append(command)
                return ""

            def fake_trial(*_args, **_kwargs):
                trial_dir = root / "results" / "campaign" / "trials" / "trial-1"
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / "checkpoint.pt").write_bytes(b"weights")
                (trial_dir / "train_screen.log").write_text("training")
                return {"outcome": "completed"}

            with patch("research.azure_worker.campaign", return_value={
                "results_root": str(root / "results"), "tag": "campaign"}):
                with self.assertRaisesRegex(ValueError, "bad archive"):
                    run_cycle(args, run=fake_run, run_trial=fake_trial,
                              archive=lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad archive")))
                self.assertFalse(any(command[:3] == ("az", "vm", "deallocate") for command in calls))
                calls.clear()
                result = run_cycle(args, run=fake_run, run_trial=fake_trial,
                                   archive=lambda *_a, **_k: {"verified": True})
                self.assertTrue(result["deallocation_requested"])
                self.assertTrue(any(command[:3] == ("az", "vm", "deallocate") for command in calls))

    def test_offline_fetch_skips_checkpoint_unless_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_dir = root / "receipts" / "cycle"
            receipt_dir.mkdir(parents=True)
            content = {"train.log": b"logs", "checkpoints/step.pt": b"weights"}
            receipt = {"files": {name: {"sha256": hashlib.sha256(data).hexdigest()}
                                 for name, data in content.items()}}
            (receipt_dir / "artifact_receipt.json").write_text(json.dumps(receipt))
            settings = {"local_control_root": str(root), "blob_prefix": "architecture",
                        "blob_account": "account", "blob_container": "container"}

            def fake_run(*command):
                name = command[command.index("--name") + 1].removeprefix("architecture/cycle/")
                path = Path(command[command.index("--file") + 1])
                path.write_bytes(content[name])
                return ""

            result = fetch_archive(settings, "cycle", run=fake_run)
            self.assertEqual(result["files"], 1)
            self.assertEqual(sha256(root / "analysis" / "cycle" / "train.log"),
                             receipt["files"]["train.log"]["sha256"])
            self.assertFalse((root / "analysis" / "cycle" / "checkpoints" / "step.pt").exists())
            self.assertEqual(fetch_archive(settings, "cycle", include_checkpoints=True,
                                           run=fake_run)["files"], 2)

    def test_controller_does_not_power_off_foreign_gpu_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = {"local_control_root": directory, "max_vm_hours": 12,
                        "resource_group": "test-resource-group", "vm_name": "h100"}
            calls = []
            def fake_run(*command):
                calls.append(command)
                return ""
            with patch("research.azure_cycle.assert_clean_git", return_value="commit"), \
                 patch("research.azure_cycle.power_state", return_value="PowerState/deallocated"), \
                 patch("research.azure_cycle.wait_for_ssh", return_value="127.0.0.1"), \
                 patch("research.azure_cycle.stage_commit"), \
                 patch("research.azure_cycle.remote_preflight",
                       side_effect=ForeignWorkloadError("old job active")):
                with self.assertRaisesRegex(ForeignWorkloadError, "old job active"):
                    controller_cycle(settings, trial_id="trial-1", stage="screen", run=fake_run)
            self.assertTrue(any(command[:3] == ("az", "vm", "start") for command in calls))
            self.assertFalse(any(command[:3] == ("az", "vm", "deallocate") for command in calls))

    def test_repair_refuses_to_archive_active_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle_id = "20261005T120000Zdeadbeef"
            root = Path(directory)
            entries = [{"event": "start_requested", "cycle_id": cycle_id,
                        "trial_id": "trial-1", "stage": "screen"},
                       {"event": "vm_started", "cycle_id": cycle_id, "time_unix": 100},
                       {"event": "worker_started", "cycle_id": cycle_id}]
            (root / "azure_cycles.jsonl").write_text("".join(json.dumps(row) + "\n"
                                                         for row in entries))
            settings = {"local_control_root": str(root)}
            with patch("research.azure_cycle.power_state", return_value="PowerState/running"), \
                 patch("research.azure_cycle.wait_for_ssh", return_value="127.0.0.1"), \
                 patch("research.azure_cycle.ssh", return_value="active"), \
                 patch("research.azure_cycle.launch_worker") as launch:
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    repair_cycle(settings, cycle_id)
                launch.assert_not_called()

    def test_reconcile_requires_blob_receipt_before_closing_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            cycle_id = "20261005T120000Zdeadbeef"
            path = Path(directory) / "azure_cycles.jsonl"
            entries = [{"event": "start_requested", "cycle_id": cycle_id,
                        "trial_id": "trial-1", "stage": "screen"},
                       {"event": "vm_started", "cycle_id": cycle_id, "time_unix": 100},
                       {"event": "worker_started", "cycle_id": cycle_id}]
            path.write_text("".join(json.dumps(row) + "\n" for row in entries))
            with patch("research.azure_cycle.power_state", return_value="PowerState/deallocated"), \
                 patch("research.azure_cycle.confirm_archive",
                       side_effect=ValueError("missing receipt")):
                with self.assertRaisesRegex(ValueError, "missing receipt"):
                    reconcile_cycle({"local_control_root": directory}, cycle_id)
            self.assertEqual(len(path.read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
