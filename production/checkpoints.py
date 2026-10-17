"""Audit checkpoints; optionally archive and retain them while a run is stopped."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import uuid

from production.artifacts import external_path
from terminus.integrity import file_sha256, fingerprint, fsync_directory, write_json_atomic


def verified_receipt(path, contract_sha):
    receipt = json.loads(path.read_text())
    name = receipt["file"]
    if Path(name).name != name or Path(name).suffix != ".pt":
        raise ValueError("invalid checkpoint receipt filename")
    payload = path.parent / name
    if receipt["contract_sha256"] != contract_sha:
        raise ValueError("checkpoint belongs to another run")
    if payload.stat().st_size != receipt["size_bytes"] or file_sha256(payload) != receipt["sha256"]:
        raise ValueError("checkpoint integrity verification failed")
    return receipt


def audit(run_dir):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    digest = fingerprint(run["contract"])
    if run["contract_sha256"] != digest:
        raise ValueError("run contract digest mismatch")
    valid, invalid = [], []
    directory = run_dir / "checkpoints"
    for path in sorted(directory.glob("step_*.json")):
        try:
            receipt = verified_receipt(path, digest)
            if path.stem != Path(receipt["file"]).stem:
                raise ValueError("receipt filename differs from checkpoint")
            valid.append(receipt)
        except (ValueError, KeyError, OSError) as exc:
            invalid.append({"receipt": path.name, "error": str(exc)})
    valid.sort(key=lambda r: r["state"]["updates"])
    orphans = [p.name for p in directory.glob("step_*.pt") if not p.with_suffix(".json").exists()]
    latest = None
    try:
        latest = verified_receipt(directory / "latest.json", digest)
        if latest not in valid:
            raise ValueError("latest pointer has no matching committed step receipt")
    except (ValueError, KeyError, OSError) as exc:
        invalid.append({"receipt": "latest.json", "error": str(exc)})
    return {"schema": 1, "contract_sha256": digest, "valid": valid, "invalid": invalid,
            "orphans": sorted(orphans), "latest": latest,
            "recovery_checkpoint": valid[-1]["file"] if valid else None,
            "free_bytes": shutil.disk_usage(run_dir).free}


def copy_verified(source, destination, digest):
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError(f"archive content collision: {destination.name}")
        return
    temporary = destination.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if file_sha256(temporary) != digest:
            raise ValueError("archive copy failed verification")
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def maintain(run_dir, *, archive, keep_last=3, protect=(), apply=False):
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("retain at least one checkpoint")
    run_dir, archive = external_path(run_dir), external_path(archive)
    if archive == run_dir or run_dir in archive.parents or archive in run_dir.parents:
        raise ValueError("archive must be separate from the run directory")
    # The trainer uses this same lock. No pruning during an active campaign.
    with (run_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = audit(run_dir)
        if report["invalid"] or report["orphans"]:
            raise ValueError("resolve invalid/orphan checkpoints before retention; use audit for details")
        receipts = report["valid"]
        protected = set(protect) | {r["file"] for r in receipts[-keep_last:]}
        protected.add(report["latest"]["file"])
        if set(protect) - {r["file"] for r in receipts}:
            raise ValueError("protected checkpoint is not in this run")
        prune = [r["file"] for r in receipts if r["file"] not in protected]
        report.update(archive=str(archive), prune=prune, applied=apply)
        if not apply:
            return report
        archive.mkdir(parents=True, exist_ok=True)
        with (archive / ".archive.lock").open("a") as archive_lock:
            fcntl.flock(archive_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_file = run_dir / "run.json"
            copy_verified(run_file, archive / "run.json", file_sha256(run_file))
            destination = archive / "checkpoints"
            destination.mkdir(exist_ok=True)
            # Commit all backups before removing any local checkpoint.
            for receipt in receipts:
                name = receipt["file"]
                copy_verified(run_dir / "checkpoints" / name, destination / name, receipt["sha256"])
                write_json_atomic((destination / name).with_suffix(".json"), receipt)
            write_json_atomic(destination / "latest.json", report["latest"])
            for name in prune:
                path = run_dir / "checkpoints" / name
                path.unlink()
                path.with_suffix(".json").unlink()
            fsync_directory(run_dir / "checkpoints")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--protect", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.archive:
        parser.error("--apply requires --archive")
    output = external_path(args.output, fresh=True)
    result = (maintain(args.run_dir, archive=args.archive, keep_last=args.keep_last,
                       protect=args.protect, apply=args.apply) if args.archive else audit(args.run_dir))
    write_json_atomic(output, result)
    print(json.dumps({"recovery_checkpoint": result["recovery_checkpoint"], "free_bytes": result["free_bytes"],
                      "invalid": result["invalid"], "orphans": result["orphans"]}))
    if result["invalid"] or result["orphans"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
