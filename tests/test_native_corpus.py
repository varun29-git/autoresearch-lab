import base64
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from botocore.exceptions import ClientError

import pyarrow as pa
import pyarrow.parquet as pq

from production import native_corpus
from production.tokenizer_revision import build_candidate, ATOM_CANDIDATE_SHA256, ATOM_REPLACEMENTS
from terminus.tokenizer65k import Frozen65kTokenizer, TOKENIZER_SHA256

ROOT = Path(__file__).resolve().parents[1]


class MemoryS3:
    def __init__(self):
        self.objects = {}; self.checksums = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[Key] = data; self.checksums[Key] = kwargs.get("ChecksumSHA256")
        return {}

    def head_object(self, Bucket, Key, **kwargs):
        data = self.objects[Key]
        result = {"ETag": '"' + hashlib.md5(data).hexdigest() + '"', "ContentLength": len(data)}
        if self.checksums.get(Key): result["ChecksumSHA256"] = self.checksums[Key]
        return result

    def get_object(self, Bucket, Key, **kwargs):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        if kwargs.get("IfMatch") is not None and kwargs["IfMatch"] != self.head_object(Bucket, Key)["ETag"]:
            raise ValueError("snapshot mismatch")
        return {"Body": BytesIO(self.objects[Key])}


class MultipartPublicationTests(unittest.TestCase):
    def test_composite_checksum_is_distinct_from_file_digest(self):
        data = b"source corpus bytes" * 29
        class S3:
            def __init__(self): self.parts = {}; self.aborted = False
            def create_multipart_upload(self, **kwargs):
                self.metadata = kwargs["Metadata"]; return {"UploadId": "test"}
            def upload_part(self, **kwargs):
                self.parts[kwargs["PartNumber"]] = kwargs["Body"]
                return {"ETag": str(kwargs["PartNumber"]), "ChecksumSHA256": kwargs["ChecksumSHA256"]}
            def complete_multipart_upload(self, **kwargs):
                self.completed = b"".join(self.parts[n] for n in sorted(self.parts))
                self.conditional = kwargs["IfNoneMatch"]
            def head_object(self, **kwargs):
                hashes = b"".join(hashlib.sha256(self.parts[n]).digest() for n in sorted(self.parts))
                return {"ContentLength": len(self.completed), "Metadata": self.metadata,
                        "ChecksumType": "COMPOSITE",
                        "ChecksumSHA256": base64.b64encode(hashlib.sha256(hashes).digest()).decode() + "-" + str(len(self.parts))}
            def abort_multipart_upload(self, **kwargs): self.aborted = True
        s3 = S3()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"; path.write_bytes(data)
            with patch.object(native_corpus, "SINGLE_UPLOAD_LIMIT", 1), patch.object(native_corpus, "MULTIPART_CHUNK_BYTES", 17):
                item = native_corpus.publish(s3, "bucket", path, "key")
        self.assertEqual(s3.completed, data)
        self.assertEqual(s3.conditional, "*")
        self.assertFalse(s3.aborted)
        self.assertEqual(item["sha256"], hashlib.sha256(data).hexdigest())
        head = s3.head_object()
        self.assertTrue(native_corpus._output_matches(head, item))
        self.assertFalse(native_corpus._output_matches(head | {"ChecksumSHA256": "wrong"}, item))
        self.assertFalse(native_corpus._output_matches(head | {"Metadata": {"sha256": "wrong"}}, item))


@unittest.skipUnless(importlib.util.find_spec("_terminus_tokenizer_native"), "native extension required")
class NativeCorpusTests(unittest.TestCase):
    def test_remote_original_requires_commit_and_full_payload_hash(self):
        text = b"original source"; sha = hashlib.sha256(text).hexdigest()
        url = "https://huggingface.co/datasets/org/data/resolve/" + "a" * 40 + "/file.parquet"
        job = {"source_key": url, "source_snapshot": {"bytes": len(text), "sha256": sha}}
        class Response(BytesIO):
            def geturl(self): return "https://cdn.example/file"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            with patch.object(native_corpus, "urlopen", return_value=Response(text)):
                self.assertEqual(native_corpus.download_source(None, "b", job, path), sha)
            with patch.object(native_corpus, "urlopen", return_value=Response(b"wrong")):
                with self.assertRaisesRegex(ValueError, "pinned payload"):
                    native_corpus.download_source(None, "b", job, path)
        job["source_key"] = url.replace("a" * 40, "main")
        with self.assertRaisesRegex(ValueError, "immutable"):
            native_corpus.check_source(None, "b", job)

    def test_source_credentials_do_not_follow_cross_host_redirects(self):
        from urllib.request import Request
        request = Request("https://huggingface.co/file", headers={"Authorization": "Bearer test"})
        handler = native_corpus._SourceRedirect()
        next_request = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example/file")
        self.assertFalse(next_request.has_header("Authorization"))
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example/file")

    def run_case(self, *, duplicate=False, bad_offset=False, bad_reference=False, nul=False,
                 missing_field=False, uid_only=False, duplicate_raw_uid=False, resume=False, corrupt=False,
                 force_disk=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); client = MemoryS3(); base = ROOT / "tokenizer/tokenizer-65k.json"
            payload, _ = build_candidate(base); candidate = root / "candidate.json"; candidate.write_bytes(payload)
            tok = Frozen65kTokenizer(base)
            texts = ["नमस्ते ऋ", "def f():\n    return 1", "ಕನ್ನಡ ಠ"]
            if nul: texts[0] += "\x00"
            values = [{"text": text, "uuid": "row" + str(i), "source_uuid": "family" + str(i)} for i, text in enumerate(texts)]
            if duplicate_raw_uid: values[2]["source_uuid"] = "family0"
            raw = root / "source.parquet"; pq.write_table(pa.Table.from_pylist(values), raw)
            client.put_object(Bucket="test", Key="source.parquet", Body=raw.read_bytes())
            def snap(key):
                head = client.head_object("test", key)
                return {"etag": head["ETag"], "bytes": head["ContentLength"]}
            parts = []
            for part_id, (selected, validation) in enumerate([([0] if duplicate_raw_uid else [0, 2], False), ([0] if duplicate else [1], True)]):
                records = []; data = b""; anchors = {}
                for row in selected:
                    ids = tok.encode(texts[row]); uid = ("family" if uid_only else "row") + str(row)
                    record = {"doc_id": uid, "token_start": len(data) // 2 + int(bad_offset),
                              "n_tokens": len(ids), "n_bytes": len(texts[row].encode())}
                    if not uid_only: record["row_key"] = "source.parquet:" + str(row)
                    records.append(record); data += struct.pack("<" + "H" * len(ids), *ids)
                    anchors[uid] = hashlib.sha256(texts[row].encode()).hexdigest()
                ik = f"part{part_id}.index.jsonl"; tk = f"part{part_id}.uint16"
                client.put_object(Bucket="test", Key=ik, Body=("\n".join(map(json.dumps, records)) + "\n").encode())
                client.put_object(Bucket="test", Key=tk, Body=data)
                parts.append({"validation": validation, "index_key": ik, "token_key": tk,
                              "index_snapshot": snap(ik), "token_snapshot": snap(tk), "anchors": anchors,
                              "reference_bin_sha256": "0" * 64 if bad_reference else hashlib.sha256(data).hexdigest()})
            job = {"job_id": "test", "source_key": "source.parquet", "source_snapshot": snap("source.parquet"), "parts": parts,
                   "text_columns": ["missing" if missing_field else "text"]}
            plan = {"bucket": "test", "output_prefix": "candidate", "tokenizer_path": str(candidate),
                    "force_disk_selection": force_disk,
                    "tokenizer_sha256": ATOM_CANDIDATE_SHA256,
                    "comparison": {"tokenizer_path": str(base), "tokenizer_sha256": TOKENIZER_SHA256,
                                   "expand_target_ids": {str(i): [288 + b for b in c.encode()] for i, c in ATOM_REPLACEMENTS}}}
            with patch.object(native_corpus, "s3_client", lambda profile=None: client):
                result = native_corpus.convert_job(job, plan, root / "work")
                if resume or corrupt:
                    plan["resume_verified"] = True
                    if corrupt:
                        key = result["outputs"][0]["bin"]["uri"].removeprefix("s3://test/")
                        client.checksums[key] = "corrupt"
                        with self.assertRaisesRegex(ValueError, "missing or changed"):
                            native_corpus.convert_job(job, plan, root / "work")
                    else:
                        with patch.object(native_corpus, "_source_batches", side_effect=AssertionError("source reread")):
                            result = native_corpus.convert_job(job, plan, root / "work")
                        self.assertTrue(result["resumed_from_verified_receipt"])
            if result["status"] == "verified":
                self.assertFalse(result["training_accepted"])
                self.assertEqual([o["validation"] for o in result["outputs"]], [False, True])
                self.assertTrue(all(o["python_reference_file_matched"] for o in result["outputs"]))
                self.assertTrue(result["declared_atom_expansion_verified"])
                for output in result["outputs"]:
                    key = output["index"]["uri"].removeprefix("s3://test/")
                    for row in map(json.loads, client.objects[key].splitlines()):
                        self.assertEqual(row["validation"], output["validation"])
                        self.assertEqual(row["split_family_id"], "family" + row["row_key"].rsplit(":", 1)[1])
            return result

    def test_source_once_with_disjoint_train_and_val(self):
        self.assertEqual(self.run_case()["status"], "verified")

    def test_original_source_uuid_alias(self):
        self.assertEqual(self.run_case(uid_only=True)["status"], "verified")

    def test_reject_duplicate_selected_coordinates(self):
        self.assertIn("duplicate selected", self.run_case(duplicate=True)["error"])

    def test_reject_duplicate_raw_uid(self):
        self.assertIn("multiple raw source", self.run_case(uid_only=True, duplicate_raw_uid=True)["error"])

    def test_reject_bad_index(self):
        self.assertIn("offsets", self.run_case(bad_offset=True)["error"])

    def test_reject_reference_disagreement(self):
        self.assertIn("Python reference", self.run_case(bad_reference=True)["error"])

    def test_preserve_nul_without_dropping_text(self):
        self.assertEqual(self.run_case(nul=True)["status"], "verified")

    def test_explicit_source_field_required(self):
        self.assertIn("text columns", self.run_case(missing_field=True)["error"])

    def test_resume_rechecks_receipts_without_reencoding(self):
        self.assertEqual(self.run_case(resume=True)["status"], "verified")

    def test_resume_rejects_changed_durable_output(self):
        self.run_case(corrupt=True)

    def test_disk_selection_preserves_numeric_and_uid_semantics(self):
        for uid in (False, True):
            self.assertEqual(self.run_case(force_disk=True, uid_only=uid)["status"], "verified")
        self.assertIn("duplicate selected", self.run_case(force_disk=True, duplicate=True)["error"])
        self.assertIn("multiple raw source", self.run_case(force_disk=True, uid_only=True, duplicate_raw_uid=True)["error"])
