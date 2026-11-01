"""Launch-critical checksum, backup isolation and supervision regressions."""

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from production.h100_24h.auto import _runner_args
from production.h100_24h.recovery import S3Destination, UploadError, verify_remote
from production.h100_24h.supervisor import supervise


class FakeMultipart:
    def __init__(self, corrupt=False):
        self.corrupt = corrupt
        self.parts = {}
        self.aborted = False

    def create_multipart_upload(self, **args):
        self.metadata = args["Metadata"]
        return {"UploadId": "test"}

    def upload_part(self, **args):
        data = args["Body"]
        digest = base64.b64encode(hashlib.sha256(data).digest()).decode()
        assert digest == args["ChecksumSHA256"]
        self.parts[args["PartNumber"]] = data
        return {"ETag": str(args["PartNumber"]),
                "ChecksumSHA256": "corrupt" if self.corrupt else digest}

    def complete_multipart_upload(self, **args):
        assert len(args["MultipartUpload"]["Parts"]) == len(self.parts)

    def abort_multipart_upload(self, **args):
        self.aborted = True

    def head_object(self, **args):
        parts = [self.parts[k] for k in sorted(self.parts)]
        digest = hashlib.sha256(b"".join(hashlib.sha256(part).digest() for part in parts)).digest()
        return {"ContentLength": sum(map(len, parts)), "Metadata": self.metadata,
                "ChecksumSHA256": base64.b64encode(digest).decode() + f"-{len(parts)}"}

    def get_object(self, **args):
        return {"Body": io.BytesIO(json.dumps({"payload_sha256": self.metadata["sha256"]}).encode())}


def transport(client):
    dest = S3Destination.__new__(S3Destination)
    dest.bucket, dest.prefix, dest.client = "bucket", "trial/", client
    dest.part_bytes, dest.multipart_threshold = 1024, 1024
    dest.exists = lambda name: True
    return dest


def test_multipart_checksum_and_marker_verify_actual_parts(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(bytes(range(256)) * 13)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dest = transport(FakeMultipart())
    report = dest.put_file_verified(path, path.name, expected_sha256=digest)
    assert report["size_bytes"] == path.stat().st_size
    assert verify_remote(dest, path.name, expected_sha256=digest)["durable"]
    assert not verify_remote(dest, path.name, expected_sha256="0" * 64)["durable"]


def test_corrupt_multipart_aborts_without_durable_commit(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"x" * 2500)
    client = FakeMultipart(corrupt=True)
    with pytest.raises(UploadError, match="part checksum"):
        transport(client).put_file_verified(path, path.name,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert client.aborted


def test_every_arm_has_distinct_checkpoint_namespace(tmp_path):
    args = SimpleNamespace(data_root="/data", runtime_report="runtime",
                           staging_full="full", staging_proxy="proxy", preflight_full="full",
                           preflight_proxy="proxy", backup_destination="s3://bucket/campaign")
    destinations = []
    for trial in ("H", "R0", "R1", "R2", "R3"):
        command = _runner_args(tmp_path, args, trial=trial, confirmation=False,
                               rate=163248, future_work_hours=1)
        destinations.append(command[command.index("--backup-destination") + 1])
    command = _runner_args(tmp_path, args, trial="R0", confirmation=True,
                           rate=35000, future_work_hours=0)
    destinations.append(command[command.index("--backup-destination") + 1])
    assert len(set(destinations)) == 6


def test_supervisor_persists_failure_and_full_stdout(tmp_path):
    result = supervise([sys.executable, "-c", 'print("diagnostic", flush=True); raise SystemExit(3)'],
                       tmp_path, started_unix=time.time(), poll_seconds=.01)
    assert result["status"] == "failed"
    assert result["returncode"] == 3
    assert "diagnostic" in (tmp_path / "supervision/stdout.log").read_text()
    assert json.loads((tmp_path / "supervision/status.json").read_text())["status"] == "failed"


def test_supervisor_stops_at_allocation_deadline(tmp_path):
    result = supervise([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
                       started_unix=time.time(), stop_after_hours=.00001,
                       grace_seconds=.1, poll_seconds=.01)
    assert result["status"] == "failed"
    assert result["reason"] == "allocation_deadline"
