"""Bounded CPU queue worker for a pinned native source conversion campaign.

Workers lease one source job at a time, extend the lease while encoding, and
acknowledge it only after durable results. Interrupted jobs return to the queue.
Deterministic source errors go to a failure queue; they never become accepted
data. The VM launcher must provide an independent termination deadline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

import boto3
from botocore.config import Config

from .native_corpus import convert_job, s3_client


def read_pinned_json(client, record):
    response = client.get_object(Bucket=record["bucket"], Key=record["key"])
    payload = response["Body"].read()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("queue job payload differs from its pinned hash")
    return json.loads(payload)


def handle_message(message, settings, sqs, s3, work_root):
    body = json.loads(message["Body"])
    job = read_pinned_json(s3, body)
    if body.get("job_id") != job["job_id"]:
        raise ValueError("queue job ID mismatch")
    stop = threading.Event(); lease_errors = []

    def heartbeat():
        while not stop.wait(45):
            try:
                sqs.change_message_visibility(QueueUrl=settings["queue_url"],
                    ReceiptHandle=message["ReceiptHandle"], VisibilityTimeout=180)
            except Exception as exc:
                # Conversion is idempotent. A lost lease can produce redundant
                # work, but cannot overwrite a different accepted token file.
                lease_errors.append(type(exc).__name__)

    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        result = convert_job(job, settings["plan"], work_root)
        attempts = int(message.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
        if result["status"] == "verified":
            outcome = "verified"
        elif result.get("error", "").startswith(("ValueError:", "UnicodeError:", "UnicodeEncodeError:", "UnicodeDecodeError:")) or attempts >= 3:
            sqs.send_message(QueueUrl=settings["failure_queue_url"], MessageBody=json.dumps({
                "job": body, "result_receipt": result["receipt"], "error": result.get("error"),
                "attempts": attempts, "training_accepted": False}))
            outcome = "rejected"
        else:
            sqs.change_message_visibility(QueueUrl=settings["queue_url"],
                ReceiptHandle=message["ReceiptHandle"], VisibilityTimeout=30)
            return {"job_id": job["job_id"], "status": "retry", "error": result.get("error")}
        sqs.delete_message(QueueUrl=settings["queue_url"], ReceiptHandle=message["ReceiptHandle"])
        return {"job_id": job["job_id"], "status": outcome,
                "tokens": result.get("content_tokens", 0), "lease_errors": lease_errors}
    finally:
        stop.set(); thread.join(timeout=5)


def worker(settings, work_root, slot):
    client = s3_client()
    sqs = boto3.client("sqs", region_name="us-east-1", config=Config(
        connect_timeout=15, read_timeout=35, retries={"max_attempts": 4}))
    work = Path(work_root) / str(slot); work.mkdir(parents=True, exist_ok=True)
    identity = socket.gethostname() + "-" + str(os.getpid())
    counts = {"verified": 0, "rejected": 0, "retry": 0, "tokens": 0}
    consecutive_empty = 0; last_write = 0
    deadline = settings["deadline_unix"]
    while time.time() < deadline - 180:
        response = sqs.receive_message(QueueUrl=settings["queue_url"], MaxNumberOfMessages=1,
            WaitTimeSeconds=20, VisibilityTimeout=180, AttributeNames=["ApproximateReceiveCount"])
        messages = response.get("Messages", [])
        if not messages:
            attrs = sqs.get_queue_attributes(QueueUrl=settings["queue_url"],
                AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
            empty = all(int(v) == 0 for v in attrs.values())
            consecutive_empty = consecutive_empty + 1 if empty else 0
            # Keep workers available while another machine owns a lease. If
            # spot capacity disappears, its messages can still be completed.
            if consecutive_empty >= 6: break
        else:
            consecutive_empty = 0
            try:
                result = handle_message(messages[0], settings, sqs, client, work)
            except Exception as exc:
                # Infrastructure failures retain the message. Queue redrive
                # handles the bounded retry count even if a process crashes.
                sqs.change_message_visibility(QueueUrl=settings["queue_url"],
                    ReceiptHandle=messages[0]["ReceiptHandle"], VisibilityTimeout=30)
                result = {"status": "retry", "error": type(exc).__name__ + ": " + str(exc)}
            counts[result["status"]] += 1; counts["tokens"] += result.get("tokens", 0)
            print(json.dumps({"worker": identity, **result}), flush=True)
        if time.time() - last_write >= 30:
            client.put_object(Bucket=settings["plan"]["bucket"],
                Key=settings["plan"]["output_prefix"] + "/fleet/" + identity + ".json",
                Body=json.dumps({"worker": identity, "counts": counts, "heartbeat_unix": time.time(),
                                 "deadline_unix": deadline}).encode())
            last_write = time.time()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True); parser.add_argument("--work-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(); settings = json.loads(Path(args.settings).read_text())
    work = Path(args.work_root).resolve(); repo = Path(__file__).resolve().parents[1]
    if work == repo or repo in work.parents: raise ValueError("corpus work must stay outside Git")
    if args.workers < 1 or not time.time() < settings["deadline_unix"] <= time.time() + 24 * 3600:
        raise ValueError("worker count/deadline is invalid")
    if not settings["plan"].get("resume_verified") or not settings["plan"].get("clean_work"):
        raise ValueError("queue campaigns require verified resume and local cleanup")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, settings, work, slot) for slot in range(args.workers)]
        print(json.dumps({"workers": [f.result() for f in futures]}), flush=True)


if __name__ == "__main__": main()
