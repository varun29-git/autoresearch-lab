import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from production import native_corpus
from terminus.tokenizer65k import Frozen65kTokenizer
from tests.test_native_corpus import MemoryS3


class RangeS3(MemoryS3):
    def get_object(self, **kwargs):
        result = super().get_object(**kwargs)
        if kwargs.get("Range"):
            start, end = map(int, kwargs["Range"].removeprefix("bytes=").split("-"))
            result["Body"] = BytesIO(result["Body"].read()[start:end + 1])
        return result


class TextRecoveryTests(unittest.TestCase):
    def fixture(self, directory, *, bad_hash=False, skip_selected=False):
        client = RangeS3()
        texts = ["पहला दस्तावेज़", "अंतिम हिन्दी दस्तावेज़"]
        last = texts[1].encode()
        cut = len(last) - 2  # Deliberately split a UTF-8 character.
        primary = texts[0].encode() + b"\n\n" + last[:cut]
        suffix = last[cut:] + b"\n"
        client.put_object(Bucket="b", Key="first.txt", Body=primary)
        client.put_object(Bucket="b", Key="next.txt", Body=suffix + b"\nnext source record\n")
        def snapshot(key):
            h = client.head_object("b", key)
            return {"etag": h["ETag"], "bytes": h["ContentLength"]}
        tok = Frozen65kTokenizer(Path(__file__).resolve().parents[1] / "tokenizer/tokenizer-65k.json")
        parts = []
        for i, text in enumerate(texts):
            ids = tok.encode(text)
            row = i * 2
            record = {"doc_id": f"first.txt:{row}", "row_key": f"first.txt:{row}",
                      "n_bytes": len(text.encode()), "n_tokens": len(ids), "token_start": 0}
            ik, tk = f"{i}.index.jsonl", f"{i}.uint16"
            client.put_object(Bucket="b", Key=ik, Body=(json.dumps(record) + "\n").encode())
            client.put_object(Bucket="b", Key=tk, Body=struct.pack("<" + "H" * len(ids), *ids))
            parts.append({"validation": bool(i), "index_key": ik, "token_key": tk,
                          "index_snapshot": snapshot(ik), "token_snapshot": snapshot(tk)})
        job = {"job_id": "repair", "source_key": "first.txt", "source_snapshot": snapshot("first.txt"),
               "source_format": "text_lines", "line_number_base": 0, "parts": parts,
               "text_source_recovery": {"schema": 1, "unselected_prefix_lines": 1 if skip_selected else 0,
                   "tail_completion": {"source_key": "next.txt", "source_snapshot": snapshot("next.txt"),
                                       "bytes": len(suffix), "sha256": "0" * 64 if bad_hash else hashlib.sha256(suffix).hexdigest()}}}
        return client, job, texts, primary + suffix

    def test_complete_split_utf8_character_without_replacing_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            client, job, texts, expected = self.fixture(directory)
            path = Path(directory) / "source"
            digest = native_corpus.download_source(client, "b", job, path)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
            batches = list(native_corpus._source_batches(path, job))
            self.assertEqual([(n, r["text"]) for b in batches for n, r in b if r["text"]],
                             [(0, texts[0]), (2, texts[1])])

    def test_tail_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            client, job, _, _ = self.fixture(directory, bad_hash=True)
            with self.assertRaisesRegex(ValueError, "reviewed first-line bytes"):
                native_corpus.download_source(client, "b", job, Path(directory) / "source")

    def test_only_unselected_leading_fragment_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"\xb9fragment\n\n" + "नया दस्तावेज़".encode() + b"\n")
            job = {"source_key": "source.txt", "source_format": "text_lines", "line_number_base": 0,
                   "text_source_recovery": {"schema": 1, "unselected_prefix_lines": 2}}
            self.assertEqual(list(native_corpus._source_batches(path, job))[0][0],
                             (2, {"text": "नया दस्तावेज़"}))
            job["text_source_recovery"]["unselected_prefix_lines"] = 0
            with self.assertRaises(UnicodeDecodeError):
                list(native_corpus._source_batches(path, job))

    def test_changed_companion_object_is_rejected_on_source_check(self):
        with tempfile.TemporaryDirectory() as directory:
            client, job, _, _ = self.fixture(directory)
            client.objects["next.txt"] = b"changed"
            with self.assertRaisesRegex(ValueError, "frozen snapshot"):
                native_corpus.check_source(client, "b", job)

    def test_native_conversion_keeps_original_train_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            client, job, _, _ = self.fixture(directory)
            from production.tokenizer_revision import build_candidate, ATOM_CANDIDATE_SHA256
            base = Path(__file__).resolve().parents[1] / "tokenizer/tokenizer-65k.json"
            payload, _ = build_candidate(base)
            candidate = Path(directory) / "tokenizer.json"; candidate.write_bytes(payload)
            plan = {"bucket": "b", "output_prefix": "candidate", "tokenizer_path": str(candidate),
                    "tokenizer_sha256": ATOM_CANDIDATE_SHA256}
            with patch.object(native_corpus, "s3_client", lambda profile=None: client):
                result = native_corpus.convert_job(job, plan, Path(directory) / "work")
                self.assertEqual(result["status"], "verified", result.get("error"))
                self.assertEqual([o["validation"] for o in result["outputs"]], [False, True])
                plan["resume_verified"] = True
                client.objects["next.txt"] = b"changed"
                with self.assertRaisesRegex(ValueError, "frozen snapshot"):
                    native_corpus.convert_job(job, plan, Path(directory) / "work")

    def test_prefix_skip_cannot_remove_a_selected_record(self):
        with tempfile.TemporaryDirectory() as directory:
            client, job, _, _ = self.fixture(directory, skip_selected=True)
            from production.tokenizer_revision import build_candidate, ATOM_CANDIDATE_SHA256
            base = Path(__file__).resolve().parents[1] / "tokenizer/tokenizer-65k.json"
            payload, _ = build_candidate(base)
            candidate = Path(directory) / "tokenizer.json"; candidate.write_bytes(payload)
            plan = {"bucket": "b", "output_prefix": "candidate", "tokenizer_path": str(candidate),
                    "tokenizer_sha256": ATOM_CANDIDATE_SHA256}
            with patch.object(native_corpus, "s3_client", lambda profile=None: client):
                result = native_corpus.convert_job(job, plan, Path(directory) / "work")
            self.assertEqual(result["status"], "rejected")
            self.assertIn("skip a selected", result["error"])


if __name__ == "__main__":
    unittest.main()
