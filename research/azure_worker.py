"""One H100 experiment, verified Blob archive, then self-deallocation.

Run under a systemd unit on the existing Azure VM. The VM identity needs Blob
Data Contributor on the dedicated container and VM Contributor on this VM.
No deallocation request is issued until a fresh download passes SHA-256 review.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path, PurePosixPath

from .artifacts import create, verify
from .loop import status, trial
from .protocol import campaign, read_json, sha256, write_json


def command(*args: str) -> str:
    result = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"{args[0]} command failed ({result.returncode}): "
                           f"{result.stderr.strip()[-2000:]}")
    return result.stdout.strip()


def archive_and_verify(root: Path, *, account: str, container: str,
                       blob_prefix: str, verification_parent: Path,
                       run=command) -> dict:
    """Verify a new, unique Blob prefix by downloading every recorded file."""
    root = root.resolve()
    receipt_path = root / "artifact_receipt.json"
    create(root, receipt_path)
    receipt = read_json(receipt_path)
    names = sorted([*receipt["files"], receipt_path.name])
    prefix = blob_prefix.strip("/")
    if not prefix or any(part in {"", ".", ".."} for part in PurePosixPath(prefix).parts):
        raise ValueError("Blob prefix must be a nonempty safe path")
    run("az", "storage", "blob", "upload-batch", "--account-name", account,
        "--destination", container, "--destination-path", prefix,
        "--source", str(root), "--auth-mode", "login", "--overwrite", "false",
        "--max-connections", "8", "--no-progress", "--output", "none")
    listed = json.loads(run("az", "storage", "blob", "list", "--account-name", account,
                            "--container-name", container, "--prefix", prefix + "/",
                            "--auth-mode", "login", "--query", "[].name", "--output", "json"))
    expected = {f"{prefix}/{name}" for name in names}
    if set(listed) != expected:
        raise ValueError(f"Blob archive names differ: missing={sorted(expected - set(listed))}, "
                         f"extra={sorted(set(listed) - expected)}")
    verification_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terminus-blob-verify-", dir=verification_parent) as scratch:
        scratch = Path(scratch)
        downloaded = scratch / "download"
        downloaded.mkdir()
        trusted_receipt = scratch / "trusted" / receipt_path.name
        trusted_receipt.parent.mkdir()
        shutil.copy2(receipt_path, trusted_receipt)
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive path: {name}")
            destination = downloaded.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            run("az", "storage", "blob", "download", "--account-name", account,
                "--container-name", container, "--name", f"{prefix}/{name}",
                "--file", str(destination), "--auth-mode", "login",
                "--overwrite", "true", "--max-connections", "8",
                "--no-progress", "--output", "none")
        result = verify(downloaded, trusted_receipt)
    return {**result, "blob_prefix": prefix, "receipt_sha256": sha256(receipt_path)}


def run_cycle(args, *, run=command, run_trial=trial,
              archive=archive_and_verify) -> dict:
    """The only successful path to deallocation goes through verified archive."""
    settings = campaign(args.campaign)
    campaign_root = Path(settings["results_root"]) / settings["tag"]
    trial_dir = campaign_root / "trials" / args.trial_id
    result = None
    failure = None
    run("az", "login", "--identity", "--output", "none")
    run("az", "storage", "container", "show", "--account-name", args.blob_account,
        "--name", args.blob_container, "--auth-mode", "login", "--output", "none")
    telemetry_path = campaign_root / "cycles" / f"{args.cycle_id}.gpu_dmon.log"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_process = None
    telemetry_handle = None
    if not args.archive_only and shutil.which("nvidia-smi"):
        telemetry_handle = telemetry_path.open("w", encoding="utf-8")
        try:
            telemetry_process = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "pucm", "-d", "10", "-o", "DT"],
                stdout=telemetry_handle, stderr=subprocess.STDOUT)
        except Exception:
            telemetry_handle.close()
            raise
    try:
        if not args.archive_only:
            try:
                result = run_trial(args.campaign, args.trial_id, args.stage,
                                   qat_rationale=args.qat_rationale,
                                   max_seconds=args.trial_budget_seconds)
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
    finally:
        if telemetry_process is not None:
            telemetry_process.terminate()
            try:
                telemetry_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                telemetry_process.kill()
                telemetry_process.wait()
        if telemetry_handle is not None:
            telemetry_handle.close()
    archive_root = trial_dir if trial_dir.is_dir() else campaign_root / "controller_failures" / args.cycle_id
    archive_root.mkdir(parents=True, exist_ok=True)
    if telemetry_path.is_file():
        shutil.copy2(telemetry_path, archive_root / f"gpu_dmon_{args.stage}.log")
    if shutil.which("nvidia-smi"):
        try:
            (archive_root / f"gpu_identity_{args.stage}.log").write_text(
                run("nvidia-smi", "-L") + "\n", encoding="utf-8")
        except Exception as exc:
            (archive_root / f"gpu_identity_{args.stage}.error.log").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    context = archive_root / "archive_context"
    context.mkdir(exist_ok=True)
    for name in ("campaign_lock.json", "ledger.jsonl"):
        source = campaign_root / name
        if source.is_file():
            shutil.copy2(source, context / name)
    if args.unit:
        try:
            journal = run("sudo", "-n", "journalctl", "--unit", args.unit,
                          "--no-pager", "--output", "short-iso")
        except Exception as exc:
            journal = f"journal unavailable: {type(exc).__name__}: {exc}"
        (archive_root / f"worker_{args.stage}.log").write_text(journal + "\n", encoding="utf-8")
    try:
        write_json(archive_root / f"diagnostic_status_{args.stage}.json",
                   status(args.campaign, trial_id=args.trial_id))
    except Exception as exc:
        (archive_root / f"diagnostic_status_{args.stage}.error.log").write_text(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", encoding="utf-8")
    write_json(archive_root / "cycle_result.json", {
        "cycle_id": args.cycle_id, "trial_id": args.trial_id, "stage": args.stage,
        "trial_result": result, "worker_failure": failure,
        "archive_only_recovery": args.archive_only,
        "note": "VM self-deallocation follows only after independent Blob download and hash verification"})
    verified = archive(archive_root, account=args.blob_account,
                       container=args.blob_container, blob_prefix=args.blob_prefix,
                       verification_parent=args.verification_parent, run=run)
    # Keep the verification proof locally on the persistent disk as well. The
    # remote archive was already verified; this file is a controller convenience.
    write_json(campaign_root / "cycles" / f"{args.cycle_id}.verified.json", verified)
    run("az", "vm", "deallocate", "--resource-group", args.resource_group,
        "--name", args.vm_name, "--no-wait", "--output", "none")
    return {"verified": verified, "trial_result": result, "worker_failure": failure,
            "deallocation_requested": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--stage", choices=("screen", "full", "qat_probe"), required=True)
    parser.add_argument("--qat-rationale")
    parser.add_argument("--trial-budget-seconds", type=float, required=True)
    parser.add_argument("--archive-only", action="store_true",
                        help="recover an inactive failed cycle without re-running training")
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--vm-name", required=True)
    parser.add_argument("--blob-account", required=True)
    parser.add_argument("--blob-container", required=True)
    parser.add_argument("--blob-prefix", required=True)
    parser.add_argument("--verification-parent", type=Path, required=True)
    print(json.dumps(run_cycle(parser.parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
