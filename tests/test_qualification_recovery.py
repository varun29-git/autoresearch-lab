"""CPU regression tests for durable backup, verified restore and watchdogs."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from terminus.integrity import file_sha256, write_json_atomic

from production.kimi_h100_5b.recovery import (BackupQueue, LocalDestination, UploadError,
                                              backup_health, restore_checkpoint,
                                              upload_checkpoint, verify_remote)


def make_checkpoint(directory, step="step_00000100.pt", payload_bytes=b"x" * 4096, updates=100):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / step
    payload.write_bytes(payload_bytes)
    receipt = {"schema": 1, "file": step, "sha256": file_sha256(payload),
               "size_bytes": payload.stat().st_size, "contract_sha256": "cd" * 32,
               "state": {"updates": updates, "tokens_seen": updates * 16,
                         "next_micro_step": updates}}
    write_json_atomic(directory / (Path(step).stem + ".json"), receipt)
    return receipt


class UploadRestoreTests(unittest.TestCase):
    def test_upload_verify_restore_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = make_checkpoint(root / "checkpoints")
            dest = LocalDestination(root / "remote")
            result = upload_checkpoint(root / "checkpoints", "step_00000100.pt", dest)
            self.assertEqual(result["status"], "durable")
            self.assertTrue(dest.exists("step_00000100.uploaded.json"))
            remote = verify_remote(dest, "step_00000100.pt", expected_sha256=receipt["sha256"])
            self.assertTrue(remote["durable"])
            restored = restore_checkpoint(dest, "step_00000100.pt", root / "restored",
                                          expected_sha256=receipt["sha256"])
            self.assertEqual(restored["status"], "restored_verified")
            self.assertEqual(restored["state"]["updates"], 100)

    def test_corrupt_upload_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = make_checkpoint(root / "checkpoints")
            dest = LocalDestination(root / "remote")
            upload_checkpoint(root / "checkpoints", "step_00000100.pt", dest)
            (dest.root / "step_00000100.pt").write_bytes(b"tampered")
            remote = verify_remote(dest, "step_00000100.pt", expected_sha256=receipt["sha256"])
            self.assertFalse(remote["durable"])
            with self.assertRaisesRegex(UploadError, "digest mismatch"):
                restore_checkpoint(dest, "step_00000100.pt", root / "restored",
                                   expected_sha256=receipt["sha256"])
            self.assertFalse((root / "restored/step_00000100.pt").exists())

    def test_upload_detects_local_tamper_and_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = make_checkpoint(root / "checkpoints")
            dest = LocalDestination(root / "remote")
            (root / "checkpoints/step_00000100.pt").write_bytes(b"changed after receipt")
            with self.assertRaisesRegex(UploadError, "integrity"):
                upload_checkpoint(root / "checkpoints", "step_00000100.pt", dest)
            (dest.root / "step_00000100.pt").write_bytes(b"foreign content")
            make_checkpoint(root / "checkpoints2", payload_bytes=b"different payload")
            with self.assertRaisesRegex(UploadError, "collision"):
                upload_checkpoint(root / "checkpoints2", "step_00000100.pt", dest)

    def test_interrupted_partials_are_swept_by_age(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = LocalDestination(Path(directory) / "remote")
            stale = dest.root / ".step_x.deadbeef.partial"
            stale.write_bytes(b"interrupted")
            old = time.time() - 7200
            os.utime(stale, (old, old))
            fresh = dest.root / ".step_y caf00.partial.partial"
            fresh.write_bytes(b"recent")
            removed = dest.sweep_partials(older_than_seconds=3600)
            self.assertEqual(removed, [stale.name])
            self.assertTrue(fresh.exists())

    def test_backup_health_watchdog_alarms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root / "checkpoints", step="step_00000100.pt", updates=100)
            make_checkpoint(root / "checkpoints", step="step_00000200.pt",
                            payload_bytes=b"y" * 2048, updates=200)
            dest = LocalDestination(root / "remote")
            upload_checkpoint(root / "checkpoints", "step_00000100.pt", dest)
            health = backup_health(root / "checkpoints", dest, max_pending_bytes=1024,
                                   max_pending_seconds=1, min_free_bytes=0,
                                   now=time.time() + 100)
            self.assertEqual(health["durable"], ["step_00000100.pt"])
            self.assertEqual([item["checkpoint"] for item in health["pending"]],
                             ["step_00000200.pt"])
            self.assertIn("backup_queue_bytes_exceeded", health["alarms"])
            self.assertIn("backup_queue_age_exceeded", health["alarms"])
            health = backup_health(root / "checkpoints", dest, max_pending_bytes=10**9,
                                   max_pending_seconds=10**6, min_free_bytes=10**18)
            self.assertEqual(health["alarms"], ["low_disk_space"])


class BackupQueueTests(unittest.TestCase):
    def test_async_upload_drains_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root / "checkpoints")
            queue = BackupQueue(root / "checkpoints", LocalDestination(root / "remote"))
            queue.enqueue("step_00000100.pt")
            status = queue.close()
            self.assertEqual(status["completed"], ["step_00000100.pt"])
            self.assertEqual(status["failed"], [])

    def test_failed_upload_stays_accountable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir(parents=True)
            (root / "checkpoints/step_00000001.pt").write_bytes(b"orphan without receipt")
            queue = BackupQueue(root / "checkpoints", LocalDestination(root / "remote"))
            queue.enqueue("step_00000001.pt")
            status = queue.close()
            self.assertEqual(status["completed"], [])
            self.assertEqual(len(status["failed"]), 1)
            self.assertIn("missing", status["failed"][0]["error"])
            self.assertTrue((root / "checkpoints/step_00000001.pt").exists())

    def test_duplicate_enqueue_and_closed_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root / "checkpoints")
            queue = BackupQueue(root / "checkpoints", LocalDestination(root / "remote"))
            queue.enqueue("step_00000100.pt")
            queue.enqueue("step_00000100.pt")
            queue.close()
            with self.assertRaises(UploadError):
                queue.enqueue("step_00000100.pt")


if __name__ == "__main__":
    unittest.main()
