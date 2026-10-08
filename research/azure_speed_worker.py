"""One speed-campaign S0 session on the H100, verified Blob archive, then self-deallocation.

Same discipline as research.azure_worker: no deallocation request is issued
until a fresh download passes SHA-256 review; a failed archive leaves the VM
running for repair. The trial body is the speed campaign's predeclared S0
plan (research.speed_s0), not the architecture runner.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import traceback
from pathlib import Path

from .azure_worker import archive_and_verify
from .speed_s0 import run_plan


def command(*args: str) -> str:
    result = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"{args[0]} command failed ({result.returncode}): "
                           f"{result.stderr.strip()[-2000:]}")
    return result.stdout.strip()


def run_cycle(args, *, run=command, archive=archive_and_verify) -> dict:
    """The only successful path to deallocation goes through verified archive."""
    trial_dir = Path(args.results_root) / args.cycle_id
    result = None
    failure = None
    run("az", "login", "--identity", "--output", "none")
    run("az", "storage", "container", "show", "--account-name", args.blob_account,
        "--name", args.blob_container, "--auth-mode", "login", "--output", "none")
    telemetry_path = trial_dir / "gpu_dmon.log"
    trial_dir.mkdir(parents=True, exist_ok=True)
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
                result = run_plan(speed_root=args.speed_root, data_root=args.data_root,
                                  results_root=args.results_root, cycle_id=args.cycle_id,
                                  plan_path=args.s0_plan,
                                  budget_seconds=args.trial_budget_seconds)
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
    archive_root = trial_dir
    if shutil.which("nvidia-smi"):
        try:
            (archive_root / "gpu_identity.log").write_text(
                run("nvidia-smi", "-L") + "\n", encoding="utf-8")
        except Exception as exc:
            (archive_root / "gpu_identity.error.log").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    context = archive_root / "archive_context"
    context.mkdir(exist_ok=True)
    for name in ("campaign.lock.json", "campaign.lock.sha256", "optimizer_recipe.json",
                 "schedule.json", "manifest.json"):
        source = Path(args.speed_root) / name
        if source.is_file():
            shutil.copy2(source, context / name)
    shutil.copy2(args.s0_plan, context / "s0_plan.json")
    if args.unit:
        try:
            journal = run("sudo", "-n", "journalctl", "--unit", args.unit,
                          "--no-pager", "--output", "short-iso")
        except Exception as exc:
            journal = f"journal unavailable: {type(exc).__name__}: {exc}"
        (archive_root / "worker.log").write_text(journal + "\n", encoding="utf-8")
    with (archive_root / "cycle_result.json").open("w", encoding="utf-8") as handle:
        json.dump({"cycle_id": args.cycle_id, "trial_id": "s0", "campaign": "stack-004-speed",
                   "s0_result": result, "worker_failure": failure,
                   "archive_only_recovery": args.archive_only,
                   "note": "VM self-deallocation follows only after independent Blob download and hash verification"},
                  handle, indent=1, sort_keys=True)
        handle.write("\n")
    verified = archive(archive_root, account=args.blob_account,
                       container=args.blob_container, blob_prefix=args.blob_prefix,
                       verification_parent=args.verification_parent, run=run)
    (Path(args.results_root) / f"{args.cycle_id}.verified.json").write_text(
        json.dumps(verified, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    run("az", "vm", "deallocate", "--resource-group", args.resource_group,
        "--name", args.vm_name, "--no-wait", "--output", "none")
    return {"verified": verified, "s0_result": result, "worker_failure": failure,
            "deallocation_requested": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--s0-plan", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--vm-name", required=True)
    parser.add_argument("--blob-account", required=True)
    parser.add_argument("--blob-container", required=True)
    parser.add_argument("--blob-prefix", required=True)
    parser.add_argument("--verification-parent", required=True)
    parser.add_argument("--trial-budget-seconds", type=float, required=True)
    parser.add_argument("--archive-only", action="store_true",
                        help="recover an inactive failed cycle without re-running the plan")
    print(json.dumps(run_cycle(parser.parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
