import json
import tempfile
import unittest
from pathlib import Path

from research.speed_session import (CAP_SECONDS, SHUTDOWN_MARGIN_SECONDS,
                                    argument_parser,
                                    close_session, config, cutoff_command,
                                    log_path, observe, open_session, usage)
from research.azure_cycle import append_event, ledger


def settings(root):
    return {"resource_group": "group", "vm_name": "h100", "ssh_user": "researcher",
            "remote_repo": "/home/researcher/terminus", "persistent_mount": "/srv/data",
            "remote_results_root": "/srv/data/speed/results",
            "local_control_root": str(root / "control")}


class FakeAzure:
    def __init__(self):
        self.power = "PowerState/deallocated"
        self.calls = []

    def run(self, *args):
        self.calls.append(args)
        if args[:3] == ("az", "vm", "get-instance-view"):
            return self.power
        if args[:3] == ("az", "vm", "start"):
            self.power = "PowerState/running"
        if args[:3] == ("az", "vm", "deallocate"):
            self.power = "PowerState/deallocated"
        return ""


class SpeedSessionTests(unittest.TestCase):
    def test_campaign_cap_is_five_cumulative_vm_hours(self):
        self.assertEqual(CAP_SECONDS, 5 * 3600 + 5400 + 5400)  # + second 1.5h extension

    def test_documented_command_order_parses(self):
        parser = argument_parser()
        command = parser.parse_args(("exec", "--config", "/tmp/config.json", "--",
                                     "nvidia-smi", "-L"))
        self.assertEqual(command.remote_command, ["--", "nvidia-smi", "-L"])
        self.assertEqual(parser.parse_args(("close", "--config", "/tmp/config.json",
                                            "--reason", "budget")).reason, "budget")

    def test_config_requires_persistent_results_outside_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            value = settings(root)
            path.write_text(json.dumps(value))
            self.assertEqual(config(path)["remote_results_root"], "/srv/data/speed/results")
            value["remote_results_root"] = "/mnt/ephemeral/results"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "persistent"):
                config(path)

    def test_open_keeps_vm_running_and_counts_across_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            value = settings(Path(directory))
            azure = FakeAzure()
            clock = [1000.0]
            remote_calls = []

            def remote(*args, **kwargs):
                remote_calls.append(args[2])
                return "active"

            def open_now():
                return open_session(value, run=azure.run, now=lambda: clock[0],
                                    connect=lambda *a, **k: "host", remote=remote)

            first = open_now()
            self.assertEqual(first["vm_power_state"], "PowerState/running")
            self.assertEqual(first["remaining_seconds"], CAP_SECONDS)
            self.assertEqual(first["last_boot_ready_seconds"], 0)
            self.assertTrue(any("systemd-run" in call for call in remote_calls))
            self.assertTrue(any("mountpoint -q" in call for call in remote_calls))
            self.assertFalse(any(call[:3] == ("az", "vm", "deallocate") for call in azure.calls))

            clock[0] = 1600.0
            paused = close_session(value, reason="idle", run=azure.run, now=lambda: clock[0])
            self.assertEqual(paused["used_seconds"], 600)
            self.assertEqual(paused["last_shutdown_seconds"], 0)
            clock[0] = 2600.0
            second = open_now()
            self.assertEqual(second["remaining_seconds"], CAP_SECONDS - 600)
            timers = [call for call in remote_calls if "systemd-run" in call]
            self.assertIn(f"--on-active={CAP_SECONDS - SHUTDOWN_MARGIN_SECONDS - 600}s",
                          timers[-1])
            self.assertEqual(len([x for x in ledger(log_path(value)) if x["event"] == "allocated"]), 2)
            clock[0] = 3200.0
            finished = close_session(value, reason="budget", run=azure.run, now=lambda: clock[0])
            self.assertEqual(finished["used_seconds"], 1200)

    def test_vm_boot_time_reduces_the_live_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            value = settings(Path(directory))
            azure = FakeAzure()
            clock = [1000.0]
            remote_calls = []

            def connect(*args, **kwargs):
                clock[0] += 300
                return "host"

            def remote(*args, **kwargs):
                remote_calls.append(args[2])
                return "active"

            result = open_session(value, run=azure.run, now=lambda: clock[0],
                                  connect=connect, remote=remote)
            self.assertEqual(result["used_seconds"], 300)
            self.assertEqual(result["last_boot_ready_seconds"], 300)
            timer = next(call for call in remote_calls if "systemd-run" in call)
            self.assertIn(f"--on-active={CAP_SECONDS - 300 - SHUTDOWN_MARGIN_SECONDS}s",
                          timer)

    def test_start_failure_releases_vm(self):
        with tempfile.TemporaryDirectory() as directory:
            value = settings(Path(directory))
            azure = FakeAzure()

            def remote(*args, **kwargs):
                raise RuntimeError("cutoff unavailable")

            with self.assertRaisesRegex(RuntimeError, "cutoff unavailable"):
                open_session(value, run=azure.run, now=lambda: 1000.0,
                             connect=lambda *a, **k: "host", remote=remote)
            self.assertEqual(azure.power, "PowerState/deallocated")
            self.assertIsNone(usage(ledger(log_path(value)), 1000)["active_session_id"])

    def test_lost_agent_reconciles_cutoff_without_resetting_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            value = settings(Path(directory))
            append_event(log_path(value), {"event": "allocated", "session_id": "s1",
                                           "time_unix": 1000.0})
            azure = FakeAzure()
            state = observe(value, run=azure.run, now=lambda: 1500.0)
            self.assertEqual(state["used_seconds"], 500)
            self.assertIsNone(state["active_session_id"])
            self.assertEqual(len(ledger(log_path(value))), 2)

    def test_cutoff_is_before_cap_and_stop_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            value = settings(Path(directory))
            command = cutoff_command(value, "session1", CAP_SECONDS - SHUTDOWN_MARGIN_SECONDS)
            self.assertIn(f"--on-active={CAP_SECONDS - SHUTDOWN_MARGIN_SECONDS}s", command)
            self.assertIn("az vm deallocate", command)
            azure = FakeAzure()
            self.assertIsNone(close_session(value, reason="goal", run=azure.run,
                                            now=lambda: 1000)["active_session_id"])


if __name__ == "__main__":
    unittest.main()
