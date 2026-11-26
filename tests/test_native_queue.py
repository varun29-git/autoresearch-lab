import hashlib
from io import BytesIO
import json
import unittest
from unittest.mock import patch

from production import native_queue


class Queue:
    def __init__(self): self.actions = []
    def delete_message(self, **kwargs): self.actions.append(("delete", kwargs))
    def send_message(self, **kwargs): self.actions.append(("failure", kwargs))
    def change_message_visibility(self, **kwargs): self.actions.append(("retry", kwargs))


class NativeQueueTests(unittest.TestCase):
    def run_result(self, result, attempts=1, corrupt=False):
        job = json.dumps({"job_id": "source-1"}).encode()
        class S3:
            def get_object(self, **kwargs): return {"Body": BytesIO(job)}
        body = {"job_id": "source-1", "bucket": "b", "key": "job.json",
                "sha256": "0" * 64 if corrupt else hashlib.sha256(job).hexdigest()}
        message = {"Body": json.dumps(body), "ReceiptHandle": "lease",
                   "Attributes": {"ApproximateReceiveCount": str(attempts)}}
        settings = {"queue_url": "work", "failure_queue_url": "failure", "plan": {}}
        queue = Queue()
        with patch.object(native_queue, "convert_job", return_value=result):
            outcome = native_queue.handle_message(message, settings, queue, S3(), "/tmp")
        return outcome, queue.actions

    def test_acknowledge_only_durable_success(self):
        result, actions = self.run_result({"status": "verified", "content_tokens": 123, "receipt": {"sha256": "verified"}})
        self.assertEqual(result["tokens"], 123)
        self.assertEqual([a[0] for a in actions], ["delete"])

    def test_deterministic_failure_is_durable_before_ack(self):
        result, actions = self.run_result({"status": "rejected", "error": "ValueError: roundtrip", "receipt": {"sha256": "failure"}})
        self.assertEqual(result["status"], "rejected")
        self.assertEqual([a[0] for a in actions], ["failure", "delete"])

    def test_transient_failure_is_retried_then_rejected(self):
        failed = {"status": "rejected", "error": "ReadTimeoutError: read", "receipt": {"sha256": "failure"}}
        self.assertEqual([a[0] for a in self.run_result(failed)[1]], ["retry"])
        self.assertEqual([a[0] for a in self.run_result(failed, attempts=3)[1]], ["failure", "delete"])

    def test_job_identity_mismatch_is_not_acknowledged(self):
        with self.assertRaisesRegex(ValueError, "pinned hash"):
            self.run_result({}, corrupt=True)
