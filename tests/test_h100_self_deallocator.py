"""Safety checks for the VM-local H100 deallocation guard."""

import json
import time

import pytest

from production.h100_24h import self_deallocator as guard


def test_failure_requires_matching_terminal_or_stale_status_and_no_controller():
    now = 2000.0
    launch = {"allocation_started_unix": 1000.0}
    state = {"status": "failed", "returncode": 2, "time_unix": 1990.0,
             "allocation_started_unix": 1000.0}
    terminal = {"status": "failed", "returncode": 2, "time_unix": 1990.0}
    assert guard.valid_status(state, launch, 1100.0, now)
    assert guard.failure_evidence(state, now, [42], terminal) is None
    assert guard.failure_evidence(state, now, [], None) is None
    assert guard.failure_evidence(state, now, [],
                                  {**terminal, "returncode": 3}) is None
    assert guard.failure_evidence(state, now, [], terminal)[0] == "campaign_failed"
    assert not guard.valid_status({**state, "allocation_started_unix": 999.0 - 5},
                                  launch, 1100.0, now)
    assert not guard.valid_status({**state, "time_unix": 1099.0}, launch, 1100.0, now)

    running = {**state, "status": "running", "time_unix": now - 299}
    assert guard.failure_evidence(running, now, [], terminal) is None
    running["time_unix"] = now - 301
    assert guard.failure_evidence(running, now, [42], terminal) is None
    assert guard.failure_evidence(running, now, [], terminal)[0] == (
        "supervisor_stale_without_campaign_process")


def _packet(tmp_path):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    now = time.time()
    allocation = {"allocation_started_unix": now - 3600,
                  "resource_group": "H100_GROUP", "vm": "h100"}
    (launch_dir / "allocation.json").write_text(json.dumps(allocation))
    (launch_dir / "supervisor_generation.json").write_text(
        json.dumps({"not_before_unix": now - 1800}))
    state = {"status": "failed", "returncode": 2,
             "time_unix": now - 5,
             "allocation_started_unix": allocation["allocation_started_unix"]}
    evidence = ("campaign_failed", state["time_unix"], 2)
    vm = ("/subscriptions/sub/resourceGroups/H100_GROUP/providers/"
          "Microsoft.Compute/virtualMachines/h100")
    return state, evidence, vm


def test_guest_requires_three_matching_observations_and_recheck(tmp_path, monkeypatch):
    state, evidence, vm = _packet(tmp_path)
    observations, actions = [], []

    def observe(*_args):
        observations.append(True)
        return state, evidence, []

    monkeypatch.setattr(guard, "observe", observe)
    monkeypatch.setattr(guard, "preserve", lambda *_args: actions.append("preserve"))
    monkeypatch.setattr(guard, "request_deallocation",
                        lambda *_args: actions.append("deallocate"))
    monkeypatch.setattr(guard.time, "sleep", lambda _seconds: None)
    assert guard.main(["--root", str(tmp_path), "--vm-resource-id", vm,
                       "--backup-uri", "s3://bucket/campaign"]) == 0
    assert len(observations) == 4  # Three polls, then one last safety read.
    assert actions == ["preserve", "deallocate"]


def test_transient_observation_error_clears_guest_candidate(tmp_path, monkeypatch):
    state, evidence, vm = _packet(tmp_path)
    sequence = [(state, evidence, []), RuntimeError("transient read"),
                (state, evidence, []), (state, evidence, [])]
    actions = []

    def observe(*_args):
        item = sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    class StopPolling(Exception):
        pass

    sleeps = []

    def sleep(_seconds):
        sleeps.append(True)
        if len(sleeps) == 4:
            raise StopPolling

    monkeypatch.setattr(guard, "observe", observe)
    monkeypatch.setattr(guard, "preserve", lambda *_args: actions.append("preserve"))
    monkeypatch.setattr(guard, "request_deallocation",
                        lambda *_args: actions.append("deallocate"))
    monkeypatch.setattr(guard.time, "sleep", sleep)
    with pytest.raises(StopPolling):
        guard.main(["--root", str(tmp_path), "--vm-resource-id", vm,
                    "--backup-uri", "s3://bucket/campaign"])
    assert actions == []


def test_arm_request_uses_imds_token_and_vm_scoped_post(monkeypatch):
    requests = []

    class Response:
        def __init__(self, status, body=b""):
            self.status, self.body = status, body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            return self.body

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            if len(requests) == 1:
                return Response(200, b'{"access_token":"fake-token"}')
            return Response(202)

    monkeypatch.setattr(guard.urllib.request, "build_opener", lambda *_args: Opener())
    vm = "/subscriptions/sub/resourceGroups/H100_GROUP/providers/Microsoft.Compute/virtualMachines/h100"
    assert guard.arm_deallocate(vm) == 202
    assert "169.254.169.254" in requests[0].full_url
    assert requests[1].get_method() == "POST"
    assert requests[1].full_url.endswith("/virtualMachines/h100/deallocate?api-version=2024-07-01")
    assert requests[1].get_header("Authorization") == "Bearer fake-token"


def test_accepted_azure_request_writes_done_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "arm_deallocate", lambda _vm: 202)
    guard.request_deallocation(tmp_path, "/vm", tmp_path / "events.jsonl",
                               ("campaign_failed", 123.0, 2))
    receipt = json.loads((tmp_path / "launch/self_deallocation_done.json").read_text())
    assert receipt["azure_http_status"] == 202
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "azure_deallocation_accepted"
