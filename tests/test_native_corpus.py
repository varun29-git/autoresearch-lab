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


@unittest.skipUnless(importlib.util.find_spec("_terminus_tokenizer_native"), "native extension required")
class NativeCorpusTests(unittest.TestCase):
    def run_case(self, *, duplicate=False, bad_offset=False, bad_reference=False, nul=False,
                 missing_field=False, uid_only=False, duplicate_raw_uid=False, resume=False, corrupt=False):
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

    def test_reject_nul_without_dropping_text(self):
        self.assertIn("NUL", self.run_case(nul=True)["error"])

    def test_explicit_source_field_required(self):
        self.assertIn("text columns", self.run_case(missing_field=True)["error"])

    def test_resume_rechecks_receipts_without_reencoding(self):
        self.assertEqual(self.run_case(resume=True)["status"], "verified")

    def test_resume_rejects_changed_durable_output(self):
        self.run_case(corrupt=True)
