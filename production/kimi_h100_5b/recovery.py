"""Durable checkpoint backup, verified restore and backup-health watchdog.

Every upload is a two-phase commit: payload and receipt land under unique
temporary names, are re-hashed from the destination, and only then renamed
and marked durable with a completion marker. A corrupt or interrupted upload
can never be mistaken for a durable copy. Retention safety: nothing here
deletes a local checkpoint; local retention stays with
production.checkpoints.maintain, which archives before pruning.

The S3 destination is a thin boto3 transport using ambient credentials; it
is implemented but native-unrun in this handoff. Never log credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from terminus.integrity import file_sha256, fsync_directory, write_json_atomic

PARTIAL_SUFFIX = ".partial"
MARKER_SUFFIX = ".uploaded.json"


class UploadError(RuntimeError):
    """A backup could not be committed durably; never silently dropped."""


class LocalDestination:
    """Directory destination with two-phase commit semantics (test double for S3)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file_verified(self, source, name: str, *, expected_sha256: str) -> dict:
        source = Path(source)
        temporary = self.root / f".{name}.{uuid.uuid4().hex}{PARTIAL_SUFFIX}"
        final = self.root / name
        try:
            with source.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            actual = file_sha256(temporary)
            if actual != expected_sha256:
                raise UploadError(f"uploaded content digest mismatch for {name}")
            if final.exists() and file_sha256(final) != expected_sha256:
                raise UploadError(f"destination content collision for {name}")
            os.replace(temporary, final)
            fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        return {"name": name, "sha256": expected_sha256, "size_bytes": final.stat().st_size}

    def read_sha256(self, name: str) -> str:
        return file_sha256(self.root / name)

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def sweep_partials(self, *, older_than_seconds: float = 3600.0, now=None) -> list:
        """Remove interrupted-upload residue older than the bound; never final files."""
        now = time.time() if now is None else now
        removed = []
        for path in self.root.glob(f"*{PARTIAL_SUFFIX}"):
            if now - path.stat().st_mtime > older_than_seconds:
                path.unlink()
                removed.append(path.name)
        return removed


class S3Destination:  # native-unrun transport: exercised only on the allocated VM
    """Thin boto3 transport with checksum-verified puts and ambient credentials."""

    def __init__(self, uri: str):
        import boto3  # noqa: delayed import keeps CPU tests free of the client

        from terminus.s3io import split_s3_uri

        self.bucket, self.prefix = split_s3_uri(uri.rstrip("/") + "/x")
        self.prefix = self.prefix[:-1]
        self.client = boto3.client("s3")

    def _key(self, name: str) -> str:
        return self.prefix + name

    def put_file_verified(self, source, name: str, *, expected_sha256: str) -> dict:
        import base64

        source = Path(source)
        digest = base64.b64encode(bytes.fromhex(expected_sha256)).decode()
        self.client.put_object(Bucket=self.bucket, Key=self._key(name),
                               Body=source.open("rb"), ChecksumSHA256=digest)
        head = self.client.head_object(Bucket=self.bucket, Key=self._key(name),
                                       ChecksumMode="ENABLED")
        remote = head.get("ChecksumSHA256")
        if remote != digest:
            raise UploadError(f"S3 checksum mismatch for {name}")
        return {"name": name, "sha256": expected_sha256, "size_bytes": head["ContentLength"]}

    def read_sha256(self, name: str) -> str:
        raise NotImplementedError("S3 restore downloads then hashes locally")

    def exists(self, name: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(name))
            return True
        except self.client.exceptions.ClientError:
            return False

    def download(self, name: str, target) -> None:
        self.client.download_file(self.bucket, self._key(name), str(target))


def destination(spec):
    if spec.startswith("s3://"):
        return S3Destination(spec)
    return LocalDestination(spec)


def upload_checkpoint(checkpoint_dir, step_name: str, dest, *, now=None) -> dict:
    """Durably back up one committed checkpoint payload and its sidecar receipt."""
    import json

    checkpoint_dir = Path(checkpoint_dir)
    payload = checkpoint_dir / step_name
    receipt_path = checkpoint_dir / (payload.stem + ".json")
    if not payload.is_file() or not receipt_path.is_file():
        raise UploadError(f"checkpoint payload or receipt missing for {step_name}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("file") != payload.name or receipt.get("sha256") != file_sha256(payload):
        raise UploadError(f"local checkpoint integrity failure for {step_name}")
    started = time.time() if now is None else now
    uploaded_payload = dest.put_file_verified(payload, payload.name,
                                              expected_sha256=receipt["sha256"])
    uploaded_receipt = dest.put_file_verified(receipt_path, receipt_path.name,
                                              expected_sha256=file_sha256(receipt_path))
    marker = {"schema": 1, "checkpoint": payload.name, "receipt": receipt_path.name,
              "payload_sha256": receipt["sha256"], "contract_sha256": receipt.get("contract_sha256"),
              "state": receipt.get("state"), "durable_unix": started}
    import json as _json

    marker_path = checkpoint_dir / (payload.stem + MARKER_SUFFIX)
    write_json_atomic(marker_path, marker)
    dest.put_file_verified(marker_path, marker_path.name,
                           expected_sha256=file_sha256(marker_path))
    return {"schema": 1, "status": "durable", "payload": uploaded_payload,
            "receipt_file": uploaded_receipt["name"], "marker": marker,
            "seconds": (time.time() if now is None else now) - started}


def verify_remote(dest, step_name: str, *, expected_sha256: str) -> dict:
    """Re-read the remote payload and marker; a mismatch means not durable."""
    if isinstance(dest, S3Destination):
        raise NotImplementedError("remote re-read runs in the native recovery drill")
    marker_name = Path(step_name).stem + MARKER_SUFFIX
    if not dest.exists(marker_name) or not dest.exists(step_name):
        return {"checkpoint": step_name, "durable": False}
    actual = dest.read_sha256(step_name)
    return {"checkpoint": step_name, "durable": actual == expected_sha256,
            "remote_sha256": actual, "expected_sha256": expected_sha256}


def restore_checkpoint(dest, step_name: str, target_dir, *, expected_sha256: str) -> dict:
    """Verified restore into a fresh directory; hashes must match the receipt."""
    import json

    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        raise UploadError("restore target must be a fresh directory")
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = target_dir / step_name
    if isinstance(dest, S3Destination):
        dest.download(step_name, payload)
        dest.download(Path(step_name).stem + ".json", target_dir / (Path(step_name).stem + ".json"))
    else:
        shutil.copyfile(dest.root / step_name, payload)
        shutil.copyfile(dest.root / (Path(step_name).stem + ".json"),
                        target_dir / (Path(step_name).stem + ".json"))
    actual = file_sha256(payload)
    if actual != expected_sha256:
        payload.unlink()
        raise UploadError(f"restored checkpoint digest mismatch for {step_name}")
    receipt = json.loads((target_dir / (Path(step_name).stem + ".json")).read_text())
    if receipt.get("sha256") != expected_sha256 or receipt.get("file") != step_name:
        raise UploadError(f"restored receipt does not match {step_name}")
    return {"schema": 1, "status": "restored_verified", "checkpoint": step_name,
            "sha256": actual, "state": receipt.get("state")}


def backup_health(checkpoint_dir, dest, *, max_pending_bytes: int, max_pending_seconds: float,
                  min_free_bytes: int, now=None) -> dict:
    """Watchdog: pending upload bytes/age and free space against declared bounds."""
    import json

    now = time.time() if now is None else now
    checkpoint_dir = Path(checkpoint_dir)
    pending_bytes, oldest_age, pending, durable = 0, 0.0, [], []
    for receipt_path in sorted(checkpoint_dir.glob("step_*.json")):
        if receipt_path.name == "latest.json":
            continue
        receipt = json.loads(receipt_path.read_text())
        payload = checkpoint_dir / receipt.get("file", "")
        if not payload.is_file():
            continue
        remote = verify_remote(dest, payload.name, expected_sha256=receipt["sha256"])
        if remote["durable"]:
            durable.append(payload.name)
            continue
        age = now - payload.stat().st_mtime
        pending.append({"checkpoint": payload.name, "age_seconds": age,
                        "size_bytes": payload.stat().st_size})
        pending_bytes += payload.stat().st_size
        oldest_age = max(oldest_age, age)
    free_bytes = shutil.disk_usage(checkpoint_dir).free
    alarms = []
    if pending_bytes > max_pending_bytes:
        alarms.append("backup_queue_bytes_exceeded")
    if oldest_age > max_pending_seconds:
        alarms.append("backup_queue_age_exceeded")
    if free_bytes < min_free_bytes:
        alarms.append("low_disk_space")
    return {"schema": 1, "durable": durable, "pending": pending,
            "pending_bytes": pending_bytes, "oldest_pending_seconds": oldest_age,
            "free_bytes": free_bytes, "alarms": alarms}


class BackupQueue:
    """Single-worker asynchronous uploader; failures surface, never vanish."""

    def __init__(self, checkpoint_dir, dest):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dest = dest
        self._condition = threading.Condition()
        self._pending = []
        self.failed = []
        self.completed = []
        self._closed = False
        self._worker = threading.Thread(target=self._run, daemon=True, name="ci1-backup")
        self._worker.start()

    def enqueue(self, step_name: str) -> None:
        with self._condition:
            if self._closed:
                raise UploadError("backup queue is closed")
            if step_name in self._pending or step_name in self.completed:
                return
            self._pending.append(step_name)
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending:
                    return
                step = self._pending[0]
            try:
                upload_checkpoint(self.checkpoint_dir, step, self.dest)
            except Exception as exc:  # surfaced via status(); item stays accountable
                with self._condition:
                    self._pending.remove(step)
                    self.failed.append({"checkpoint": step, "error": f"{type(exc).__name__}: {exc}"})
            else:
                with self._condition:
                    self._pending.remove(step)
                    self.completed.append(step)

    def status(self) -> dict:
        with self._condition:
            return {"pending": list(self._pending), "completed": list(self.completed),
                    "failed": list(self.failed)}

    def close(self, *, timeout: float = 600.0) -> dict:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout)
        status = self.status()
        if status["pending"] or self._worker.is_alive():
            raise UploadError("backup queue did not drain before close")
        return status
