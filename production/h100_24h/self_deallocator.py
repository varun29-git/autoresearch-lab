"""Independent VM-local deallocation guard for an authorized H100 allocation.

The guest's system-assigned identity must have the deallocate action scoped to
this VM. No Azure credential is stored in the launch packet. The guard never
starts or resumes training and acts only after repeated terminal/stale evidence
or the fixed allocation deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request


POLL_SECONDS = 60
CONFIRMATIONS = 3
STALE_RUNNING_SECONDS = 300
STALE_FINALIZING_SECONDS = 600
DEALLOCATE_AFTER_HOURS = 23.75
EXPIRE_AFTER_HOURS = 24
LOG_SUFFIXES = {".json", ".jsonl", ".log", ".md"}

UPLOAD_CODE = '''
import base64, hashlib, pathlib, sys
import boto3
from botocore.config import Config
path, uri = pathlib.Path(sys.argv[1]), sys.argv[2]
bucket, key = uri[5:].split('/', 1)
digest = base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()
client = boto3.client('s3', config=Config(connect_timeout=15, read_timeout=60,
                                        retries={'mode':'standard','max_attempts':3}))
with path.open('rb') as body:
    client.put_object(Bucket=bucket, Key=key, Body=body, ChecksumSHA256=digest)
head = client.head_object(Bucket=bucket, Key=key, ChecksumMode='ENABLED')
assert head['ChecksumSHA256'] == digest
'''


def append_event(path: Path, event: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps({"time_unix": time.time(), **event},
                                    sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        print(json.dumps({"event": "event_log_unavailable", "error": str(error),
                          "original_event": event.get("event")}),
              file=sys.stderr, flush=True)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def active_campaign_pids() -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(args[index] == b"-m" and
               args[index + 1] == b"production.h100_24h.auto"
               for index in range(len(args) - 1)):
            found.append(int(proc.name))
    return found


def last_terminal_event(path: Path) -> dict | None:
    if not path.is_file():
        return None
    terminal = None
    with path.open() as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("event") == "campaign_terminal":
                terminal = item
    return terminal


def valid_status(state: dict, launch: dict, generation: float,
                 now: float) -> bool:
    if not isinstance(state, dict) or state.get("status") not in {
            "running", "stopping", "finalizing", "completed", "failed"}:
        return False
    try:
        if abs(float(state["allocation_started_unix"]) -
               float(launch["allocation_started_unix"])) > 1:
            return False
        if not generation <= float(state["time_unix"]) <= now + 60:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    if state["status"] in {"completed", "failed"}:
        if type(state.get("returncode")) is not int:
            return False
        if state["status"] == "completed" and state["returncode"] != 0:
            return False
    return True


def failure_evidence(state: dict, now: float, pids: list[int],
                     terminal: dict | None) -> tuple | None:
    if pids:
        return None
    status, stamp = state["status"], state["time_unix"]
    if status in {"completed", "failed"}:
        if (terminal is not None and terminal.get("status") == status and
                terminal.get("returncode") == state["returncode"] and
                terminal.get("time_unix") == stamp):
            return ("campaign_" + status, stamp, state["returncode"])
    elif status in {"running", "stopping"} and now - stamp >= STALE_RUNNING_SECONDS:
        return ("supervisor_stale_without_campaign_process", stamp)
    elif status == "finalizing" and now - stamp >= STALE_FINALIZING_SECONDS:
        return ("finalization_stale_without_campaign_process", stamp)
    return None


def observe(root: Path, launch: dict, generation: float,
            now: float) -> tuple[dict | None, tuple | None, list[int]]:
    status_path = root / "supervision/status.json"
    state = json.loads(status_path.read_text())
    if not valid_status(state, launch, generation, now):
        return state, None, []
    pids = active_campaign_pids()
    terminal = last_terminal_event(root / "supervision/events.jsonl")
    return state, failure_evidence(state, now, pids, terminal), pids


def preserve(root: Path, backup_uri: str, evidence: tuple,
             event_path: Path) -> None:
    write_json(root / "launch/self_deallocation_intent.json",
               {"time_unix": time.time(), "evidence": evidence,
                "allocation_started_unix": json.loads(
                    (root / "launch/allocation.json").read_text())["allocation_started_unix"]})
    fallback = root / "launch/persist_failed_checkpoint.py"
    try:
        result = subprocess.run(["/usr/bin/python3", str(fallback)],
                                capture_output=True, text=True, timeout=180)
        append_event(event_path, {"event": "checkpoint_fallback_checked",
                                  "returncode": result.returncode,
                                  "output": result.stdout[-1000:],
                                  "error": result.stderr[-1000:]})
    except Exception as error:
        append_event(event_path, {"event": "checkpoint_fallback_unavailable",
                                  "error": str(error)})

    snapshots = root / "supervision/snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    archive_path = snapshots / f"self_deallocation_{int(time.time())}.tar.gz"
    try:
        with tarfile.open(archive_path, "w:gz") as target:
            for directory, names, files in os.walk(root, followlinks=True):
                names[:] = [name for name in names if name not in {
                    "snapshots", ".git", "__pycache__"}]
                for name in files:
                    path = Path(directory) / name
                    if path.suffix in LOG_SUFFIXES:
                        target.add(path, arcname=str(path.relative_to(root)),
                                   recursive=False)
        uri = backup_uri.rstrip("/") + "/logs/" + archive_path.name
        upload = subprocess.run(
            ["/mnt/ci1-24h/runtime/venv/bin/python", "-c", UPLOAD_CODE,
             str(archive_path), uri], capture_output=True, text=True,
            timeout=180)
        if upload.returncode:
            raise RuntimeError(upload.stderr[-1000:])
        append_event(event_path, {"event": "failure_logs_durable", "uri": uri,
                                  "bytes": archive_path.stat().st_size,
                                  "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()})
    except Exception as error:
        append_event(event_path, {"event": "failure_log_upload_unavailable",
                                  "error": str(error)})


def arm_deallocate(vm_resource_id: str) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    imds = ("http://169.254.169.254/metadata/identity/oauth2/token"
            "?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F")
    token_request = urllib.request.Request(imds, headers={"Metadata": "true"})
    with opener.open(token_request, timeout=10) as response:
        token = json.load(response)["access_token"]
    url = ("https://management.azure.com" + vm_resource_id +
           "/deallocate?api-version=2024-07-01")
    request = urllib.request.Request(
        url, data=b"", method="POST",
        headers={"Authorization": "Bearer " + token,
                 "Content-Type": "application/json"})
    with opener.open(request, timeout=30) as response:
        if response.status not in {200, 202, 204}:
            raise RuntimeError(f"Azure deallocate returned {response.status}")
        return response.status


def request_deallocation(root: Path, vm_resource_id: str,
                         event_path: Path, evidence: tuple) -> None:
    done = root / "launch/self_deallocation_done.json"
    for attempt in range(5):
        try:
            status = arm_deallocate(vm_resource_id)
            try:
                write_json(done, {"time_unix": time.time(), "evidence": evidence,
                                  "azure_http_status": status})
                append_event(event_path, {"event": "azure_deallocation_accepted",
                                          "azure_http_status": status,
                                          "evidence": evidence})
            except OSError as error:
                print(f"Azure accepted deallocation; local receipt failed: {error}",
                      file=sys.stderr, flush=True)
            return
        except Exception as error:
            append_event(event_path, {"event": "azure_deallocation_error",
                                      "attempt": attempt + 1,
                                      "error": str(error)[:1000]})
            time.sleep(30)
    raise RuntimeError("Azure deallocation request was not accepted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vm-resource-id", required=True)
    parser.add_argument("--backup-uri", required=True)
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    launch = json.loads((root / "launch/allocation.json").read_text())
    generation = json.loads((root / "launch/supervisor_generation.json").read_text())["not_before_unix"]
    vm_id = args.vm_resource_id.rstrip("/")
    if not (vm_id.lower().endswith("/resourcegroups/" +
             launch["resource_group"].lower() +
             "/providers/microsoft.compute/virtualmachines/" +
             launch["vm"].lower()) and args.backup_uri.startswith("s3://")):
        raise ValueError("VM identity or backup URI does not match allocation")
    event_path = root / "launch/self_deallocator.jsonl"
    done = root / "launch/self_deallocation_done.json"

    if args.check_once:
        now = time.time()
        state, evidence, pids = observe(root, launch, generation, now)
        print(json.dumps({"status": state.get("status") if isinstance(state, dict) else None,
                          "valid": valid_status(state, launch, generation, now),
                          "active_campaign_pids": pids, "evidence": evidence},
                         sort_keys=True))
        return 0

    if done.is_file():
        return 0
    if time.time() >= launch["allocation_started_unix"] + EXPIRE_AFTER_HOURS * 3600:
        append_event(event_path, {"event": "allocation_expired_without_action"})
        return 0
    append_event(event_path, {"event": "self_deallocator_started",
                              "allocation_started_unix": launch["allocation_started_unix"],
                              "required_confirmations": CONFIRMATIONS})
    candidate, confirmations = None, 0
    while True:
        now = time.time()
        if done.is_file():
            return 0
        try:
            if now >= launch["allocation_started_unix"] + DEALLOCATE_AFTER_HOURS * 3600:
                state, pids = {}, []
                evidence = ("allocation_deadline", launch["allocation_started_unix"])
            else:
                state, evidence, pids = observe(root, launch, generation, now)
                if not valid_status(state, launch, generation, now):
                    evidence = None
                    append_event(event_path, {"event": "invalid_status_ignored"})
            if evidence is None:
                candidate, confirmations = None, 0
            else:
                if evidence == candidate:
                    confirmations += 1
                else:
                    candidate, confirmations = evidence, 1
                    append_event(event_path, {"event": "deallocation_candidate",
                                              "evidence": evidence,
                                              "status": state.get("status"),
                                              "active_campaign_pids": pids})
                if confirmations >= CONFIRMATIONS:
                    append_event(event_path, {"event": "deallocation_confirmed",
                                              "evidence": evidence})
                    try:
                        preserve(root, args.backup_uri, evidence, event_path)
                    except Exception as error:
                        append_event(event_path, {"event": "preservation_unavailable",
                                                  "error": str(error)[:1000]})
                    if evidence[0] != "allocation_deadline":
                        _, check_evidence, check_pids = observe(
                            root, launch, generation, time.time())
                        if check_evidence != evidence or check_pids:
                            append_event(event_path, {"event": "deallocation_cancelled_after_recheck",
                                                      "evidence": evidence})
                            candidate, confirmations = None, 0
                            time.sleep(POLL_SECONDS)
                            continue
                    request_deallocation(root, vm_id, event_path, evidence)
                    return 0
        except Exception as error:
            candidate, confirmations = None, 0
            append_event(event_path, {"event": "monitor_error", "error": str(error)[:1000]})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
