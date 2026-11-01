"""Checksum-verified multipart S3 transport for full-model checkpoints."""

import base64
import hashlib
import json
from pathlib import Path
import shutil
import time

from production.kimi_h100_5b.recovery import *  # noqa: F401,F403
from production.kimi_h100_5b.recovery import S3Destination as _LegacyS3


def multipart_identity(path, part_bytes):
    full, parts = hashlib.sha256(), []
    with Path(path).open("rb") as handle:
        while chunk := handle.read(part_bytes):
            full.update(chunk)
            parts.append(hashlib.sha256(chunk).digest())
    composite = base64.b64encode(hashlib.sha256(b"".join(parts)).digest()).decode()
    return full.hexdigest(), parts, f"{composite}-{len(parts)}"


class S3Destination(_LegacyS3):
    part_bytes = 256 * 1024**2
    multipart_threshold = 64 * 1024**2

    def __init__(self, uri):
        super().__init__(uri)
        import boto3
        from botocore.config import Config
        self.client = boto3.client("s3", config=Config(
            connect_timeout=15, read_timeout=120,
            retries={"mode": "standard", "max_attempts": 5}))

    def put_file_verified(self, source, name, *, expected_sha256):
        source = Path(source)
        if source.stat().st_size < self.multipart_threshold:
            with source.open("rb") as body:
                digest = base64.b64encode(bytes.fromhex(expected_sha256)).decode()
                self.client.put_object(Bucket=self.bucket, Key=self._key(name), Body=body,
                                       ChecksumSHA256=digest,
                                       Metadata={"sha256": expected_sha256})
            head = self.client.head_object(Bucket=self.bucket, Key=self._key(name),
                                           ChecksumMode="ENABLED")
            if head.get("ChecksumSHA256") != digest:
                raise UploadError(f"S3 checksum mismatch for {name}")
        else:
            actual, hashes, composite = multipart_identity(source, self.part_bytes)
            if actual != expected_sha256:
                raise UploadError(f"source checksum mismatch for {name}")
            created = self.client.create_multipart_upload(
                Bucket=self.bucket, Key=self._key(name), ChecksumAlgorithm="SHA256",
                Metadata={"sha256": expected_sha256, "composite-sha256": composite})
            upload_id = created["UploadId"]
            try:
                parts = []
                with source.open("rb") as handle:
                    for number, part_hash in enumerate(hashes, 1):
                        chunk = handle.read(self.part_bytes)
                        if hashlib.sha256(chunk).digest() != part_hash:
                            raise UploadError("checkpoint changed during multipart upload")
                        checksum = base64.b64encode(part_hash).decode()
                        result = self.client.upload_part(
                            Bucket=self.bucket, Key=self._key(name), UploadId=upload_id,
                            PartNumber=number, Body=chunk, ChecksumSHA256=checksum)
                        if result.get("ChecksumSHA256") != checksum:
                            raise UploadError(f"S3 part checksum mismatch: {name} part {number}")
                        parts.append({"PartNumber": number, "ETag": result["ETag"],
                                      "ChecksumSHA256": checksum})
                self.client.complete_multipart_upload(
                    Bucket=self.bucket, Key=self._key(name), UploadId=upload_id,
                    MultipartUpload={"Parts": parts})
            except BaseException:
                self.client.abort_multipart_upload(Bucket=self.bucket, Key=self._key(name),
                                                   UploadId=upload_id)
                raise
            head = self.client.head_object(Bucket=self.bucket, Key=self._key(name),
                                           ChecksumMode="ENABLED")
            if (head.get("ChecksumSHA256") != composite
                    or head.get("Metadata", {}).get("sha256") != expected_sha256):
                raise UploadError(f"S3 multipart checksum mismatch for {name}")
        if head["ContentLength"] != source.stat().st_size:
            raise UploadError(f"S3 object size mismatch for {name}")
        return {"name": name, "sha256": expected_sha256, "size_bytes": head["ContentLength"]}


def destination(spec):
    return S3Destination(spec) if spec.startswith("s3://") else LocalDestination(spec)


def verify_remote(dest, step_name, *, expected_sha256):
    marker_name = Path(step_name).stem + MARKER_SUFFIX
    if not dest.exists(marker_name) or not dest.exists(step_name):
        return {"checkpoint": step_name, "durable": False}
    if isinstance(dest, S3Destination):
        head = dest.client.head_object(Bucket=dest.bucket, Key=dest._key(step_name),
                                       ChecksumMode="ENABLED")
        metadata = head.get("Metadata", {})
        expected = base64.b64encode(bytes.fromhex(expected_sha256)).decode()
        checksum = head.get("ChecksumSHA256")
        valid = (checksum == expected or
                 (metadata.get("sha256") == expected_sha256
                  and checksum == metadata.get("composite-sha256")))
        with dest.client.get_object(Bucket=dest.bucket, Key=dest._key(marker_name))["Body"] as body:
            marker = json.loads(body.read())
        valid = valid and marker.get("payload_sha256") == expected_sha256
        return {"checkpoint": step_name, "durable": bool(valid),
                "remote_checksum": checksum, "expected_sha256": expected_sha256}
    actual = dest.read_sha256(step_name)
    return {"checkpoint": step_name, "durable": actual == expected_sha256}


def backup_health(checkpoint_dir, dest, *, max_pending_bytes, max_pending_seconds,
                  min_free_bytes, now=None):
    now = time.time() if now is None else now
    root = Path(checkpoint_dir)
    pending, durable = [], []
    for path in sorted(root.glob("step_*.json")):
        if path.name.endswith(MARKER_SUFFIX):
            continue
        receipt = json.loads(path.read_text())
        payload = root / receipt["file"]
        if not payload.is_file():
            continue
        if verify_remote(dest, payload.name, expected_sha256=receipt["sha256"])["durable"]:
            durable.append(payload.name)
        else:
            pending.append({"checkpoint": payload.name, "size_bytes": payload.stat().st_size,
                            "age_seconds": now - payload.stat().st_mtime})
    pending_bytes = sum(item["size_bytes"] for item in pending)
    oldest = max((item["age_seconds"] for item in pending), default=0.0)
    free = shutil.disk_usage(root).free
    alarms = []
    if pending_bytes > max_pending_bytes:
        alarms.append("backup_queue_bytes_exceeded")
    if oldest > max_pending_seconds:
        alarms.append("backup_queue_age_exceeded")
    if free < min_free_bytes:
        alarms.append("low_disk_space")
    return {"pending": pending, "durable": durable, "pending_bytes": pending_bytes,
            "oldest_pending_seconds": oldest, "free_bytes": free, "alarms": alarms}
