"""Resume-safe source re-tokenization with native BPE and per-document round trips.

A job binds one original source object to all selected, disjoint index parts.
Reading the source once avoids repeated Parquet decompression for train/val or
chunked outputs. Published candidates retain holdout membership and document
identities; global family reconciliation and curriculum packing remain separate.
This command uses CPU workers and never provisions cloud resources.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from functools import lru_cache
import shutil
import re
from urllib.request import build_opener, HTTPRedirectHandler, Request
from urllib.parse import urlsplit
import struct
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from terminus.hashing import file_sha256
from terminus.tokenizer_native import Native65kTokenizer
from terminus import unicode_contract as unicode

SINGLE_UPLOAD_LIMIT = 5 * 1024 ** 3
MULTIPART_CHUNK_BYTES = 64 * 1024 ** 2


def _output_matches(head, item):
    checksum = item.get("s3_checksum_sha256", base64.b64encode(bytes.fromhex(item["sha256"])).decode())
    return (head["ContentLength"] == item["bytes"] and head.get("ChecksumSHA256") == checksum
            and (item.get("s3_checksum_type") != "COMPOSITE" or
                 (head.get("ChecksumType") == "COMPOSITE" and
                  head.get("Metadata", {}).get("sha256") == item["sha256"])))


class _SourceRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("source redirect left HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


urlopen = build_opener(_SourceRedirect()).open


@lru_cache(maxsize=2)
def _tokenizer(path, digest):
    # A process handles many source files. Compile the vocabulary once per
    # artifact per process, while bounded native caches remain reusable.
    return Native65kTokenizer(path, expected_sha256=digest)


def job_fingerprint(job, plan):
    contract = {"job": job, "tokenizer_sha256": plan["tokenizer_sha256"],
                "runtime_files": plan.get("runtime_files", {}),
                "comparison": plan.get("comparison"), "schema": 1}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def completed_job(s3, job, plan):
    """Trust only a matching receipt whose inputs and durable outputs still match."""
    bucket = plan["bucket"]; key = plan["output_prefix"] + "/completed/" + job_fingerprint(job, plan) + ".json"
    try:
        pointer = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"): return None
        raise
    receipt = pointer["receipt"]
    prefix = f"s3://{bucket}/"
    if not receipt["uri"].startswith(prefix + plan["output_prefix"] + "/results/"):
        raise ValueError("completion receipt is outside this campaign")
    payload = s3.get_object(Bucket=bucket, Key=receipt["uri"][len(prefix):])["Body"].read()
    if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise ValueError("completion receipt hash changed")
    result = json.loads(payload)
    if result.get("job_fingerprint") != job_fingerprint(job, plan) or result.get("status") != "verified":
        raise ValueError("completion receipt does not match this source job")
    check_source(s3, bucket, job)
    for part in job["parts"]:
        check_object(s3, bucket, part["index_key"], part["index_snapshot"])
        check_object(s3, bucket, part["token_key"], part["token_snapshot"])
    for output in result["outputs"]:
        for kind in ("bin", "index"):
            item = output[kind]
            if not item["uri"].startswith(prefix + plan["output_prefix"] + "/candidate-output/"):
                raise ValueError("completed output is outside this campaign")
            head = s3.head_object(Bucket=bucket, Key=item["uri"][len(prefix):], ChecksumMode="ENABLED")
            if not _output_matches(head, item):
                raise ValueError("completed output is missing or changed")
    return result | {"receipt": receipt, "resumed_from_verified_receipt": True}


def s3_client(profile=None):
    return boto3.Session(profile_name=profile, region_name="us-east-1").client(
        "s3", config=Config(connect_timeout=15, read_timeout=120,
                            retries={"max_attempts": 4}, max_pool_connections=16))


def check_object(s3, bucket, key, snapshot):
    head = s3.head_object(Bucket=bucket, Key=key)
    if (head["ETag"], head["ContentLength"]) != (snapshot["etag"], snapshot["bytes"]):
        raise ValueError("object changed from its frozen snapshot: " + key)
    return head


def download(s3, bucket, key, path, snapshot):
    check_object(s3, bucket, key, snapshot)
    response = s3.get_object(Bucket=bucket, Key=key, IfMatch=snapshot["etag"])
    digest = hashlib.sha256()
    with Path(path).open("wb") as handle:
        for chunk in iter(lambda: response["Body"].read(8 << 20), b""):
            handle.write(chunk); digest.update(chunk)
    check_object(s3, bucket, key, snapshot)
    if Path(path).stat().st_size != snapshot["bytes"]:
        raise ValueError("source read length differs from frozen snapshot")
    value = digest.hexdigest()
    if snapshot.get("sha256") and snapshot["sha256"] != value:
        raise ValueError("source payload hash differs from frozen snapshot")
    return value


def check_source(s3, bucket, job):
    recovery = _text_recovery(job)
    if recovery.get("tail_completion"):
        tail = recovery["tail_completion"]
        check_object(s3, bucket, tail["source_key"], tail["source_snapshot"])
    key = job["source_key"]
    if not key.startswith("https://"):
        return check_object(s3, bucket, key, job["source_snapshot"])
    # A branch name is mutable. Remote originals require both a commit URL and
    # the dataset's published LFS payload hash, verified over the whole file.
    if not re.fullmatch(r"https://huggingface\.co/datasets/[\w.-]+/[\w.-]+/resolve/[a-f0-9]{40}/[^?#]+", key):
        raise ValueError("remote source must pin an immutable Hugging Face commit")
    snap = job["source_snapshot"]
    if not re.fullmatch(r"[a-f0-9]{64}", snap.get("sha256", "")) or type(snap.get("bytes")) is not int or snap["bytes"] <= 0:
        raise ValueError("remote source needs its published SHA-256 and byte length")


def download_source(s3, bucket, job, path):
    check_source(s3, bucket, job)
    if not job["source_key"].startswith("https://"):
        digest = download(s3, bucket, job["source_key"], path, job["source_snapshot"])
        tail = _text_recovery(job).get("tail_completion")
        if tail:
            snapshot = tail["source_snapshot"]
            response = s3.get_object(Bucket=bucket, Key=tail["source_key"],
                Range=f"bytes=0-{tail['bytes'] - 1}", IfMatch=snapshot["etag"])
            try:
                raw = response["Body"].read(tail["bytes"] + 1)
            finally:
                response["Body"].close()
            if (len(raw) != tail["bytes"] or hashlib.sha256(raw).hexdigest() != tail["sha256"]
                    or not raw.endswith(b"\n") or raw.count(b"\n") != 1):
                raise ValueError("text tail completion differs from its reviewed first-line bytes")
            # Append the next raw object's first line exactly, including its LF.
            # Never decode the two fragments independently or replace bad bytes.
            with Path(path).open("ab") as handle:
                handle.write(raw)
            check_object(s3, bucket, tail["source_key"], snapshot)
            digest = file_sha256(path)
        return digest
    snapshot = job["source_snapshot"]; digest = hashlib.sha256(); size = 0
    headers = {"User-Agent": "Terminus-native-source-conversion/2"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    request = Request(job["source_key"], headers=headers)
    with urlopen(request, timeout=120) as response, Path(path).open("wb") as handle:
        if not response.geturl().startswith("https://"):
            raise ValueError("source redirect left HTTPS")
        while chunk := response.read(8 << 20):
            size += len(chunk)
            if size > snapshot["bytes"]: raise ValueError("remote source exceeds its pinned byte length")
            digest.update(chunk); handle.write(chunk)
    if size != snapshot["bytes"] or digest.hexdigest() != snapshot["sha256"]:
        raise ValueError("remote source differs from its pinned payload hash or size")
    return digest.hexdigest()


def publish(s3, bucket, path, key, *, expected_sha256=None):
    path = Path(path); digest = expected_sha256 or file_sha256(path); size = path.stat().st_size
    if len(digest)!=64 or any(c not in '0123456789abcdef'for c in digest):
        raise ValueError('invalid publication SHA-256')
    if size > SINGLE_UPLOAD_LIMIT:
        return _publish_multipart(s3, bucket, path, key, digest, size)
    checksum = base64.b64encode(bytes.fromhex(digest)).decode()
    try:
        with path.open("rb") as handle:
            s3.put_object(Bucket=bucket, Key=key, Body=handle, ContentLength=size,
                          ChecksumSHA256=checksum, IfNoneMatch="*")
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
            raise
    head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    if head.get("ChecksumSHA256") != checksum or head["ContentLength"] != size:
        raise ValueError("published object checksum differs from local output")
    return {"uri": f"s3://{bucket}/{key}", "sha256": digest, "bytes": size}


def _publish_multipart(s3, bucket, path, key, digest, size):
    """Verify each uploaded part and S3's composite digest; retain full-file SHA.

    A multipart SHA-256 is a hash of part hashes, not the full-file SHA-256.
    Both are pinned in the receipt so resume never confuses the two formats.
    """
    if (size + MULTIPART_CHUNK_BYTES - 1) // MULTIPART_CHUNK_BYTES > 10000:
        raise ValueError("source output exceeds the bounded multipart capacity")
    upload = s3.create_multipart_upload(Bucket=bucket, Key=key, ChecksumAlgorithm="SHA256",
        ChecksumType="COMPOSITE", Metadata={"sha256": digest})["UploadId"]
    parts = []; part_hashes = []; finished = False;full_hash=hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(MULTIPART_CHUNK_BYTES):
                full_hash.update(chunk)
                raw_hash = hashlib.sha256(chunk).digest(); part_hashes.append(raw_hash)
                checksum = base64.b64encode(raw_hash).decode(); number = len(parts) + 1
                response = s3.upload_part(Bucket=bucket, Key=key, UploadId=upload,
                    PartNumber=number, Body=chunk, ChecksumSHA256=checksum)
                if response.get("ChecksumSHA256") != checksum:
                    raise ValueError("uploaded part checksum differs from local bytes")
                parts.append({"PartNumber": number, "ETag": response["ETag"], "ChecksumSHA256": checksum})
        if full_hash.hexdigest()!=digest:
            raise ValueError('multipart source differs from publication SHA-256')
        checksum = base64.b64encode(hashlib.sha256(b"".join(part_hashes)).digest()).decode() + "-" + str(len(parts))
        try:
            s3.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload,
                MultipartUpload={"Parts": parts}, ChecksumType="COMPOSITE", IfNoneMatch="*")
            finished = True
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
                raise
        item = {"uri": f"s3://{bucket}/{key}", "sha256": digest, "bytes": size,
                "s3_checksum_type": "COMPOSITE", "s3_checksum_sha256": checksum,
                "multipart_chunk_bytes": MULTIPART_CHUNK_BYTES, "multipart_parts": len(parts)}
        head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if not _output_matches(head, item):
            raise ValueError("published multipart object checksum differs from local output")
        return item
    finally:
        if not finished:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload)


def _text_recovery(job):
    recovery = job.get("text_source_recovery", {})
    if not recovery:
        return {}
    if (job.get("source_format") != "text_lines" or job["source_key"].startswith("https://")
            or recovery.get("schema") != 1):
        raise ValueError("text source recovery requires a reviewed S3 text-lines recipe")
    skip = recovery.get("unselected_prefix_lines", 0)
    if type(skip) is not int or not 0 <= skip <= 2:
        raise ValueError("text recovery prefix must be zero, one or two unselected physical lines")
    tail = recovery.get("tail_completion")
    if tail:
        if (not isinstance(tail.get("source_key"), str) or tail["source_key"].startswith("https://")
                or tail["source_key"] == job["source_key"] or type(tail.get("bytes")) is not int
                or not 0 < tail["bytes"] <= 1024 ** 2
                or not re.fullmatch(r"[a-f0-9]{64}", tail.get("sha256", ""))
                or not isinstance(tail.get("source_snapshot"), dict)):
            raise ValueError("text recovery tail needs distinct pinned source and bounded prefix hash")
    return recovery


def _source_batches(path, job):
    if job.get("source_format", "parquet") == "text_lines":
        base = job.get("line_number_base")
        if base not in (0, 1):
            raise ValueError("text sources require a proven zero- or one-based line convention")
        skip = _text_recovery(job).get("unselected_prefix_lines", 0)
        batch = []
        with path.open("rb") as handle:
            for number, line in enumerate(handle):
                if number < skip:
                    continue
                value = line.removesuffix(b"\n").removesuffix(b"\r").decode(
                    "utf-8-sig" if number == 0 else "utf-8", errors="strict")
                batch.append((number + base, {"text": value}))
                if len(batch) == 512:
                    yield batch; batch = []
        if batch: yield batch
        return
    if job.get("source_format", "parquet") != "parquet":
        raise ValueError("unsupported source format")
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    columns = job.get("text_columns", ["text"])
    available = set(parquet.schema_arrow.names)
    if not all(c in available for c in columns):
        raise ValueError("explicit source text columns are missing")
    ids = [c for c in ("uuid", "source_uuid", "doc_id", "document_id", "id") if c in available]
    number = 0
    for batch in parquet.iter_batches(batch_size=512, columns=sorted(set(columns + ids)), use_threads=False):
        values = batch.to_pylist()
        yield list(enumerate(values, start=number)); number += len(values)


def _source_text(value, job):
    renderer = job.get("renderer", "field")
    columns = job.get("text_columns", ["text"])
    if renderer == "field" and len(columns) == 1:
        text = value[columns[0]]
    elif renderer == "join_nonempty_strings":
        # Field order and separator are an explicit, source-checked recipe.
        if any(value.get(c) is not None and not isinstance(value[c], str) for c in columns):
            raise ValueError("source renderer encountered a non-string field")
        text = job["field_separator"].join(value[c] for c in columns if value.get(c))
    else:
        raise ValueError("source renderer requires a reviewed explicit recipe")
    if not isinstance(text, str) or not text:
        raise ValueError("selected document has no source text")
    return text


def convert_job(job, plan, work_root, profile=None):
    start = time.monotonic(); client = s3_client(profile)
    bucket = plan["bucket"]; prefix = plan["output_prefix"]; job_id = job["job_id"]
    if Path(job_id).name != job_id:
        raise ValueError("job ID must be a simple filename")
    for path, digest in plan.get("runtime_files", {}).items():
        if file_sha256(path) != digest:
            raise ValueError("runtime file changed: " + path)
    if plan.get("resume_verified", False):
        previous = completed_job(client, job, plan)
        if previous is not None: return previous
    work = Path(work_root) / job_id; work.mkdir(parents=True, exist_ok=True)
    result = {"schema": 1, "job_id": job_id, "cell": job.get("cell"),
              "status": "started", "job_fingerprint": job_fingerprint(job, plan), "training_accepted": False,
              "global_heldout_reconciliation_complete": False}
    handles = []; part_states = []; disk_selection = None
    try:
        recovery = _text_recovery(job)
        skip_prefix = recovery.get("unselected_prefix_lines", 0)
        for path, digest in plan.get("runtime_files", {}).items():
            if file_sha256(path) != digest:
                raise ValueError("runtime file changed: " + path)
        tok = _tokenizer(plan["tokenizer_path"], plan["tokenizer_sha256"])
        result["encoder_identity"] = tok.identity()
        encoding_threads = plan.get("encoding_threads", 1)
        if type(encoding_threads) is not int or not 1 <= encoding_threads <= 32:
            raise ValueError("invalid native encoding thread count")
        encoder = tok
        if encoding_threads > 1:
            from terminus.tokenizer_parallel import native_batch_pool
            encoder = native_batch_pool(plan["tokenizer_path"], plan["tokenizer_sha256"], encoding_threads)
        result["encoding_threads"] = encoding_threads
        comparison = plan.get("comparison")
        reference = (_tokenizer(comparison["tokenizer_path"], comparison["tokenizer_sha256"]) if comparison else None)
        expansions = ({int(k): v for k, v in comparison.get("expand_target_ids", {}).items()}
                      if comparison else {})
        if plan.get("force_disk_selection") or sum(p["index_snapshot"]["bytes"] for p in job["parts"]) > (32 << 20):
            from .native_selection import DiskSelection
            selection_path = work / "selection.sqlite"
            selection_path.unlink(missing_ok=True)
            disk_selection = DiskSelection(selection_path)
        by_number = {}; by_uid = {}; expected_anchors = {}; seen = set()
        for part_index, part in enumerate(job["parts"]):
            if type(part.get("validation")) is not bool:
                raise ValueError("original holdout assignment must be explicit")
            check_object(client, bucket, part["token_key"], part["token_snapshot"])
            index_path = work / f"original-{part_index}.index.jsonl"
            index_sha = download(client, bucket, part["index_key"], index_path, part["index_snapshot"])
            out_path = work / f"candidate-{part_index}.uint16"
            index_out = work / f"candidate-{part_index}.index.jsonl"
            out = out_path.open("wb"); idx = index_out.open("w", encoding="utf-8")
            handles.extend([out, idx])
            state = {"part": part, "index_sha256": index_sha, "out": out, "idx": idx,
                     "out_path": out_path, "index_path": index_out, "documents": 0,
                     "tokens": 0, "expected_documents": 0, "reference_hash": hashlib.sha256(),
                     "reference_tokens": 0, "content_hash": hashlib.sha256()}
            part_states.append(state); cursor = 0
            with index_path.open(encoding="utf-8") as source_index:
                for line in source_index:
                    if not line.strip(): continue
                    r = json.loads(line); count = r.get("n_tokens", r.get("token_count"))
                    if type(count) is not int or count <= 0 or r.get("token_start", cursor) != cursor:
                        raise ValueError("index counts or split-local offsets are invalid")
                    cursor += count; state["expected_documents"] += 1
                    uid = str(r.get("doc_id", r.get("uuid", "")))
                    if not uid: raise ValueError("selected source identity missing")
                    row_key = r.get("row_key", "")
                    coordinate_prefix = job.get("row_identity_prefix", job["source_key"]) + ":"
                    if not row_key and job.get("id_encodes_row"):
                        row_key = uid
                    if row_key.startswith(coordinate_prefix) and row_key[len(coordinate_prefix):].isdigit():
                        coordinate = int(row_key[len(coordinate_prefix):]); target = by_number
                    else: coordinate = uid; target = by_uid
                    if skip_prefix and (target is not by_number or coordinate < skip_prefix + job.get("line_number_base", 0)):
                        raise ValueError("text recovery would skip a selected or unproven source row")
                    if disk_selection is not None:
                        disk_selection.add("number" if target is by_number else "uid", coordinate, part_index, uid, r)
                    elif coordinate in target:
                        raise ValueError("duplicate selected source row or ambiguous UID across parts")
                    else: target[coordinate] = (part_index, uid, r)
                    anchor = part.get("anchors", {}).get(uid)
                    if anchor: expected_anchors[(part_index, uid)] = anchor
            if 2 * cursor != part["token_snapshot"]["bytes"]:
                raise ValueError("source index does not exactly exhaust its original token bin")
            state["old_tokens"] = cursor
        if by_number and by_uid:
            raise ValueError("mixed row-number and UID selection requires a separate source job")
        uid_set = set(by_uid); max_row = max(by_number) if by_number else None
        if disk_selection is not None:
            disk_selection.ready()
            by_number = disk_selection.mode == "number"
            max_row = disk_selection.max_row
        source_path = work / "source.data"
        source_sha = download_source(client, bucket, job, source_path)
        matched_anchors = set(); scanned = 0; processed = 0; encoding_seconds = 0.; comparison_seconds = 0.
        next_progress = time.monotonic() + 15
        for batch in _source_batches(source_path, job):
            chosen = []; texts = []
            for number, value in batch:
                scanned += 1
                aliases = {str(value[c]) for c in ("uuid", "source_uuid", "doc_id", "document_id", "id") if value.get(c) is not None}
                if disk_selection is not None: match = disk_selection.find(number, aliases)
                elif by_number: match = by_number.get(number)
                else:
                    hits = aliases & uid_set
                    if len(hits) > 1: raise ValueError("source row matches conflicting selected identities")
                    match = by_uid[next(iter(hits))] if hits else None
                if match is None: continue
                pi, uid, r = match
                unique = number if by_number else uid
                if disk_selection is None:
                    if unique in seen: raise ValueError("selected UID occurs at multiple raw source coordinates")
                    seen.add(unique)
                if by_number and aliases and uid not in aliases and uid not in (r.get("row_key"), job["source_key"] + ":" + str(number)):
                    raise ValueError("source identity differs at indexed coordinate")
                text = _source_text(value, job); raw = text.encode("utf-8", errors="strict")
                if "n_bytes" in r and r["n_bytes"] != len(raw):
                    raise ValueError(f"source text byte count differs from its original index at "
                                     f"{job['source_key']}:{number}: expected {r['n_bytes']}, got {len(raw)}")
                digest = hashlib.sha256(raw).hexdigest()
                anchor_key = (pi, uid)
                if anchor_key in expected_anchors:
                    if expected_anchors[anchor_key] != digest: raise ValueError("source text differs from the frozen source-replay anchor")
                    matched_anchors.add(anchor_key)
                family = r.get("split_family_id", r.get("family_id", value.get("source_uuid", uid)))
                chosen.append((pi, uid, number, digest, family)); texts.append(text)
            tick = time.monotonic(); buffers = encoder.encode_batch_bytes(texts); encoding_seconds += time.monotonic() - tick
            old_buffers = None
            if reference:
                tick = time.monotonic(); old_buffers = reference.encode_batch_bytes(texts); comparison_seconds += time.monotonic() - tick
            for position, ((pi, uid, number, digest, family), data) in enumerate(zip(chosen, buffers)):
                state = part_states[pi]; part = state["part"]; ntokens = len(data) // 2
                if old_buffers is not None:
                    old = old_buffers[position]
                    if len(data) > len(old): raise ValueError("candidate increased document token count")
                    if data != old:
                        new_ids = struct.unpack("<" + "H" * ntokens, data)
                        expanded = [x for token in new_ids for x in expansions.get(token, [token])]
                        expanded_bytes = struct.pack("<" + "H" * len(expanded), *expanded)
                        if expanded_bytes != old: raise ValueError("candidate changed content beyond the declared atom replacements")
                    state["reference_hash"].update(old); state["reference_tokens"] += len(old) // 2
                record = {"doc_id": uid, "row_key": job["source_key"] + ":" + str(number),
                          "source_text_sha256": digest, "validation": part["validation"],
                          "split_family_id": str(family), "token_start": state["tokens"], "n_tokens": ntokens}
                state["idx"].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                state["out"].write(data); state["content_hash"].update(data)
                state["tokens"] += ntokens; state["documents"] += 1; processed += 1
            if time.monotonic() >= next_progress:
                progress = result | {"documents": processed, "content_tokens": sum(s["tokens"] for s in part_states),
                                     "elapsed_seconds": time.monotonic() - start}
                client.put_object(Bucket=bucket, Key=prefix + "/progress/" + job_id + ".json", Body=json.dumps(progress).encode())
                next_progress = time.monotonic() + 15
            if by_number and batch and batch[-1][0] >= max_row: break
        for h in handles: h.close()
        handles.clear()
        if matched_anchors != set(expected_anchors):
            raise ValueError("not every frozen source anchor was found")
        check_source(client, bucket, job)
        outputs = []
        for i, state in enumerate(part_states):
            part = state["part"]
            if state["documents"] != state["expected_documents"]:
                raise ValueError("source scan did not cover every selected document")
            check_object(client, bucket, part["token_key"], part["token_snapshot"])
            ref_sha = state["reference_hash"].hexdigest() if reference else None
            if part.get("reference_bin_sha256") and ref_sha != part["reference_bin_sha256"]:
                raise ValueError("native v1 IDs differ from the completed Python reference file")
            bin_sha = state["content_hash"].hexdigest(); index_sha = file_sha256(state["index_path"])
            if part.get("expected_output_bin_sha256") and bin_sha != part["expected_output_bin_sha256"]:
                raise ValueError("candidate IDs differ from the qualified native output file")
            key = f"{prefix}/candidate-output/{job_id}/{i}/{bin_sha}-{index_sha}"
            outputs.append({"original_index_key": part["index_key"], "original_index_sha256": state["index_sha256"],
                            "validation": part["validation"], "documents": state["documents"],
                            "original_tokens": state["old_tokens"], "tokens": state["tokens"],
                            "reference_tokens": state["reference_tokens"] if reference else None,
                            "reference_bin_sha256": ref_sha,
                            "python_reference_file_matched": bool(part.get("reference_bin_sha256")),
                            "bin": publish(client, bucket, state["out_path"], key + ".uint16"),
                            "index": publish(client, bucket, state["index_path"], key + ".index.jsonl")})
        result.update(status="verified", source_key=job["source_key"], source_sha256=source_sha,
                      documents=processed, source_rows_scanned=scanned,
                      content_tokens=sum(s["tokens"] for s in part_states),
                      encoding_seconds=encoding_seconds, comparison_seconds=comparison_seconds,
                      elapsed_seconds=time.monotonic() - start, outputs=outputs,
                      declared_atom_expansion_verified=bool(comparison),
                      original_holdout_assignments_preserved=True)
        if job.get("text_source_recovery"):
            result.update(text_source_recovery=job["text_source_recovery"],
                          source_sha256_scope="primary raw bytes plus reviewed tail completion")
    except Exception as exc:
        result.update(status="rejected", error=type(exc).__name__ + ": " + str(exc),
                      elapsed_seconds=time.monotonic() - start)
    finally:
        for handle in handles: handle.close()
        if disk_selection is not None: disk_selection.close()
    path = work / "result.json"; path.write_text(json.dumps(result, indent=2) + "\n")
    receipt = publish(client, bucket, path, prefix + "/results/" + job_id + "/" + file_sha256(path) + ".json")
    result["receipt"] = receipt
    client.put_object(Bucket=bucket, Key=prefix + "/progress/" + job_id + ".json", Body=json.dumps(result).encode())
    if result["status"] == "verified":
        client.put_object(Bucket=bucket, Key=prefix + "/completed/" + result["job_fingerprint"] + ".json",
                          Body=json.dumps({"receipt": receipt}).encode())
    if plan.get("clean_work", False):
        # Only discard local files after durable receipt publication. A spot
        # retry validates the receipt or replays this one source object.
        shutil.rmtree(work)
    return result


def run_plan(plan, work_root, profile=None):
    start = time.monotonic(); results = []
    if plan.get("schema") != 1 or type(plan.get("workers")) is not int or plan["workers"] < 1:
        raise ValueError("invalid native corpus plan")
    ids = [j["job_id"] for j in plan["jobs"]]
    if len(ids) != len(set(ids)): raise ValueError("duplicate source job ID")
    worker_plan = {k: v for k, v in plan.items() if k != "jobs"}
    with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
        futures = [pool.submit(convert_job, job, worker_plan, work_root, profile) for job in plan["jobs"]]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps({k: result[k] for k in ("job_id", "status", "documents", "content_tokens", "elapsed_seconds", "error") if k in result}), flush=True)
    summary = {"schema": 1, "complete": len(results) == len(plan["jobs"]),
               "verified_jobs": sum(r["status"] == "verified" for r in results),
               "rejected_jobs": sum(r["status"] == "rejected" for r in results),
               "wall_seconds": time.monotonic() - start, "results": results,
               "training_ready": False}
    path = Path(work_root) / "summary.json"; path.write_text(json.dumps(summary, indent=2) + "\n")
    s3 = s3_client(profile); receipt = publish(s3, plan["bucket"], path,
                                             plan["output_prefix"] + "/results/summary-" + file_sha256(path) + ".json")
    s3.put_object(Bucket=plan["bucket"], Key=plan["output_prefix"] + "/PILOT_RESULT.json",
                  Body=json.dumps(summary | {"receipt": receipt}).encode())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True); parser.add_argument("--work-root", required=True)
    parser.add_argument("--profile")
    args = parser.parse_args(); work = Path(args.work_root).resolve(); repo = Path(__file__).resolve().parents[1]
    if work == repo or repo in work.parents: raise ValueError("source data and results must stay outside Git")
    work.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text()); result = run_plan(plan, work, args.profile)
    if result["rejected_jobs"]: raise SystemExit(1)


if __name__ == "__main__": main()
