"""Persistent process supervision, durable status and bounded log mirroring.

The external launch record supplies the allocation time. The supervisor
terminates the whole process group before the deadline and records a terminal
status for the independent Azure deallocation watcher. It never retries a
numerically failed trial or alters the scientific packet.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading
import time


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_log(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps({"time_unix": time.time(), **value},
                                sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def mirror(root, python, uri):
    archives = root / "supervision" / "snapshots"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / "latest.tar.gz"
    temporary = archives / "building.tar.gz"
    with tarfile.open(temporary, "w:gz") as target:
        for directory, names, files in os.walk(root, followlinks=True):
            names[:] = [name for name in names if name not in {"snapshots", ".git", "__pycache__"}]
            for name in files:
                path = Path(directory) / name
                if path.suffix in {".json", ".jsonl", ".log", ".md"}:
                    target.add(path, arcname=str(path.relative_to(root)), recursive=False)
    os.replace(temporary, archive)
    subprocess.run([python, "-c", UPLOAD_CODE, str(archive),
                    uri.rstrip("/") + "/logs/latest.tar.gz"], check=True, timeout=180)
    return {"bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def supervise(command, root, *, started_unix, stop_after_hours=23.25,
              grace_seconds=1200, mirror_python=None, backup_uri=None,
              poll_seconds=10, mirror_seconds=300):
    root = Path(root)
    directory = root / "supervision"
    directory.mkdir(parents=True, exist_ok=True)
    status_path, event_path = directory / "status.json", directory / "events.jsonl"
    deadline = started_unix + stop_after_hours * 3600
    if time.time() >= deadline:
        raise ValueError("allocation deadline already reached")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)
    latest = {}

    def drain():
        with (directory / "stdout.log").open("a", buffering=1) as handle:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                try:
                    record = json.loads(line)
                    if isinstance(record, dict) and record.get("event") in {
                            "train", "validation", "complete", "stopped", "startup"}:
                        latest.update({"record": record, "observed_unix": time.time()})
                    elif isinstance(record, dict) and record.get("event") == "stage_start":
                        latest.update({"stage": record, "observed_unix": time.time()})
                except ValueError:
                    pass
            handle.flush()
            os.fsync(handle.fileno())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    mirror_lock = threading.Lock()
    mirror_thread = None

    def upload():
        with mirror_lock:
            try:
                receipt = mirror(root, mirror_python, backup_uri)
                append_log(event_path, {"event": "logs_durable", **receipt})
            except Exception as error:
                append_log(event_path, {"event": "log_mirror_failure", "error": str(error)})

    append_log(event_path, {"event": "supervisor_started", "pid": process.pid,
                            "allocation_started_unix": started_unix, "stop_deadline_unix": deadline})
    stop = {"reason": None, "sent_unix": None}

    def group_alive():
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def request_stop(signum, frame):
        stop["reason"] = stop["reason"] or f"supervisor_signal_{signum}"

    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    next_mirror = time.time()
    try:
        while process.poll() is None or (stop["reason"] and group_alive()):
            now = time.time()
            if now >= deadline:
                stop["reason"] = stop["reason"] or "allocation_deadline"
            if stop["reason"] and stop["sent_unix"] is None:
                os.killpg(process.pid, signal.SIGTERM)
                stop["sent_unix"] = now
                append_log(event_path, {"event": "graceful_stop", "reason": stop["reason"]})
            if stop["sent_unix"] and now - stop["sent_unix"] >= grace_seconds:
                os.killpg(process.pid, signal.SIGKILL)
            write_status(status_path, {"status": "stopping" if stop["reason"] else "running",
                                       "time_unix": now, "pid": process.pid, "latest": dict(latest),
                                       "allocation_started_unix": started_unix,
                                       "elapsed_hours": (now - started_unix) / 3600,
                                       "stop_deadline_unix": deadline, "reason": stop["reason"]})
            if (backup_uri and mirror_python and now >= next_mirror
                    and (mirror_thread is None or not mirror_thread.is_alive())):
                mirror_thread = threading.Thread(target=upload, daemon=True)
                mirror_thread.start()
                next_mirror = now + mirror_seconds
            time.sleep(poll_seconds)
        reader.join(timeout=15)
        result = {"status": "completed" if process.returncode == 0 and not stop["reason"] else "failed",
                  "returncode": process.returncode, "reason": stop["reason"],
                  "time_unix": time.time(), "latest": dict(latest),
                  "allocation_started_unix": started_unix,
                  "elapsed_hours": (time.time() - started_unix) / 3600}
        append_log(event_path, {"event": "campaign_terminal", **result})
        write_status(status_path, {**result, "status": "finalizing"})
        if mirror_thread is not None:
            mirror_thread.join(timeout=190)
        if backup_uri and mirror_python:
            upload()
        write_status(status_path, result)
        return result
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qual-root", required=True)
    parser.add_argument("--command-json", required=True)
    parser.add_argument("--launch-record", required=True)
    parser.add_argument("--mirror-python")
    parser.add_argument("--backup-uri")
    args = parser.parse_args()
    launch = json.loads(Path(args.launch_record).read_text())
    command = json.loads(Path(args.command_json).read_text())
    result = supervise(command, args.qual_root, started_unix=launch["allocation_started_unix"],
                       mirror_python=args.mirror_python, backup_uri=args.backup_uri)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
