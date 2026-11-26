"""Verified durable checkpoint storage; ambient credentials, no provisioning.

Multipart SHA256 is explicitly a composite checksum, never confused with a
whole-file SHA256 or an ETag. Object prefixes must be dedicated to one run.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from production.artifacts import external_path
from terminus.integrity import file_sha256, fsync_directory


def safe_name(name):
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('storage keys must be plain filenames')
    return name


def sync_download(target):
    with Path(target).open('rb') as handle:
        os.fsync(handle.fileno())
    fsync_directory(Path(target).parent)


class LocalStore:
    def __init__(self, root):
        self.root = external_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = str(self.root)

    def read_json(self, name):
        return json.loads((self.root / safe_name(name)).read_text())

    def exists(self, name):
        return (self.root / safe_name(name)).is_file()

    def verify(self, name, digest, size):
        path = self.root / safe_name(name)
        return path.is_file() and path.stat().st_size == size and file_sha256(path) == digest

    def put(self, source, name, digest):
        source = Path(source); name = safe_name(name)
        final = self.root / name
        if final.exists():
            if not self.verify(name, digest, source.stat().st_size):
                raise ValueError(f'durable content collision: {name}')
            return
        temporary = self.root / f'.{uuid.uuid4().hex}.partial'
        try:
            with source.open('rb') as src, temporary.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024**2)
                dst.flush(); os.fsync(dst.fileno())
            if file_sha256(temporary) != digest:
                raise ValueError('durable copy checksum mismatch')
            os.replace(temporary, final); fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def download(self, name, target):
        shutil.copyfile(self.root / safe_name(name), target)
        sync_download(target)


class S3Store:
    part_bytes = 256 * 1024**2
    threshold = 64 * 1024**2

    def __init__(self, uri, *, client=None):
        from terminus.s3io import split_s3_uri
        self.bucket, prefix = split_s3_uri(uri.rstrip('/') + '/placeholder')
        self.prefix = prefix[:-len('placeholder')]
        if not self.prefix.strip('/'):
            raise ValueError('S3 checkpoint destination requires a dedicated run prefix')
        self.identity = f's3://{self.bucket}/{self.prefix}'
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client('s3', config=Config(connect_timeout=10, read_timeout=60,
                                   retries={'mode':'standard', 'max_attempts':3}))
        self.client = client

    def key(self, name):
        return self.prefix + safe_name(name)

    def head(self, name):
        try:
            return self.client.head_object(Bucket=self.bucket, Key=self.key(name), ChecksumMode='ENABLED')
        except Exception as exc:
            response = getattr(exc, 'response', {})
            if response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
                return None
            raise  # Permissions/network failures are not a missing object.

    def exists(self, name):
        return self.head(name) is not None

    def read_json(self, name):
        with self.client.get_object(Bucket=self.bucket, Key=self.key(name))['Body'] as body:
            return json.loads(body.read())

    def verify(self, name, digest, size):
        head = self.head(name)
        if head is None or head['ContentLength'] != size:
            return False
        full = base64.b64encode(bytes.fromhex(digest)).decode()
        checksum = head.get('ChecksumSHA256')
        metadata = head.get('Metadata', {})
        composite = metadata.get('composite-sha256')
        return checksum == full or (bool(composite) and checksum == composite and metadata.get('sha256') == digest)

    def put(self, source, name, digest):
        source = Path(source); safe_name(name)
        size = source.stat().st_size
        if self.exists(name):
            if not self.verify(name, digest, size):
                raise ValueError(f'S3 content collision: {name}')
            return
        if size < self.threshold:
            with source.open('rb') as body:
                self.client.put_object(Bucket=self.bucket, Key=self.key(name), Body=body,
                    ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(), Metadata={'sha256':digest})
        else:
            full, hashes = hashlib.sha256(), []
            with source.open('rb') as body:
                while chunk := body.read(self.part_bytes):
                    full.update(chunk); hashes.append(hashlib.sha256(chunk).digest())
            if full.hexdigest() != digest:
                raise ValueError('checkpoint changed before multipart upload')
            composite = base64.b64encode(hashlib.sha256(b''.join(hashes)).digest()).decode() + f'-{len(hashes)}'
            upload = self.client.create_multipart_upload(Bucket=self.bucket, Key=self.key(name),
                ChecksumAlgorithm='SHA256', Metadata={'sha256':digest, 'composite-sha256':composite})['UploadId']
            try:
                parts = []
                with source.open('rb') as body:
                    for number, expected in enumerate(hashes, 1):
                        chunk = body.read(self.part_bytes)
                        if hashlib.sha256(chunk).digest() != expected:
                            raise ValueError('checkpoint changed during multipart upload')
                        checksum = base64.b64encode(expected).decode()
                        result = self.client.upload_part(Bucket=self.bucket, Key=self.key(name), UploadId=upload,
                            PartNumber=number, Body=chunk, ChecksumSHA256=checksum)
                        if result.get('ChecksumSHA256') != checksum:
                            raise ValueError('S3 multipart part checksum mismatch')
                        parts.append({'PartNumber':number, 'ETag':result['ETag'], 'ChecksumSHA256':checksum})
                self.client.complete_multipart_upload(Bucket=self.bucket, Key=self.key(name), UploadId=upload,
                                                     MultipartUpload={'Parts':parts})
            except BaseException:
                try:
                    self.client.abort_multipart_upload(Bucket=self.bucket, Key=self.key(name), UploadId=upload)
                except Exception:
                    pass  # Preserve the original error; provider lifecycle must expire abandoned parts.
                raise
        if not self.verify(name, digest, size):
            raise ValueError(f'S3 upload verification failed: {name}')

    def download(self, name, target):
        self.client.download_file(self.bucket, self.key(name), str(target))
        sync_download(target)


def make_store(destination):
    return S3Store(destination) if destination.startswith('s3://') else LocalStore(destination)
