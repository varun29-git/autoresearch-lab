"""Start the existing Azure H100 for one committed trial, then wait for safe shutdown.

This command runs on the inexpensive research controller. The systemd worker on
the VM archives and independently verifies its files before requesting Azure
deallocation. A failed archive leaves the VM running for recovery.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .protocol import ROOT, assert_clean_git, policy, read_json, sha256


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SAFE_REMOTE_PATH = re.compile(r"/[A-Za-z0-9_./-]+\Z")


class ForeignWorkloadError(RuntimeError):
    """Another boot-started GPU job owns the VM; never power it off automatically."""


def command(*args: str) -> str:
    result = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"{args[0]} command failed ({result.returncode}): "
                           f"{result.stderr.strip()[-2000:]}")
    return result.stdout.strip()


def config(path: Path) -> dict:
    value = read_json(path)
    required = ("resource_group", "vm_name", "ssh_user", "remote_repo",
                "remote_python", "remote_campaign", "remote_data_root",
                "remote_results_root", "persistent_mount", "blob_account",
                "blob_container", "blob_prefix", "verification_parent",
                "local_control_root", "max_vm_hours")
    for field in required:
        if not value.get(field):
            raise ValueError(f"Azure cycle config needs {field}")
    for field in ("resource_group", "vm_name", "ssh_user", "blob_account",
                  "blob_container"):
        if not SAFE_NAME.fullmatch(value[field]):
            raise ValueError(f"unsafe Azure cycle {field}")
    for field in ("remote_repo", "remote_python", "remote_campaign", "remote_data_root",
                  "remote_results_root", "persistent_mount", "verification_parent"):
        if not SAFE_REMOTE_PATH.fullmatch(value[field]) or ".." in Path(value[field]).parts:
            raise ValueError(f"unsafe Azure cycle {field}")
    mount = Path(value["persistent_mount"])
    for field in ("remote_campaign", "remote_data_root", "remote_results_root",
                  "verification_parent"):
        if not Path(value[field]).is_relative_to(mount):
            raise ValueError(f"{field} must be on the declared persistent mount")
    if mount.is_relative_to(Path("/mnt")) or mount.is_relative_to(Path("/tmp")):
        raise ValueError("Azure temporary resource disks cannot hold campaign data/results")
    value["local_control_root"] = str(Path(value["local_control_root"]).expanduser().resolve())
    if Path(value["local_control_root"]).is_relative_to(ROOT):
        raise ValueError("local_control_root must stay outside the Git checkout")
    if float(value["max_vm_hours"]) <= 0:
        raise ValueError("max_vm_hours must be positive")
    prefix_parts = value["blob_prefix"].strip("/").split("/")
    if not prefix_parts or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
                               or part in {".", ".."} for part in prefix_parts):
        raise ValueError("blob_prefix needs safe, nonempty path components")
    if not isinstance(value.get("remote_env", {}), dict):
        raise ValueError("remote_env must be an object")
    for key, item in value.get("remote_env", {}).items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) or not isinstance(item, str) or "\n" in item:
            raise ValueError(f"invalid remote_env setting: {key}")
    return value


def power_state(settings: dict, *, run=command) -> str:
    raw = run("az", "vm", "get-instance-view", "--resource-group", settings["resource_group"],
              "--name", settings["vm_name"], "--query",
              "instanceView.statuses[?starts_with(code, 'PowerState/')].code | [0]", "--output", "tsv")
    if not raw.startswith("PowerState/"):
        raise RuntimeError(f"could not establish Azure VM power state: {raw!r}")
    return raw


def append_event(path: Path, value: dict) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ledger(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def budget_state(events: list[dict], max_hours: float) -> dict:
    requested, aborted, started, closed = set(), set(), {}, set()
    archive_confirmed, pretrial_failures = set(), set()
    spent = 0.0
    for item in events:
        if item["event"] == "start_requested":
            requested.add(item["cycle_id"])
        elif item["event"] == "start_aborted":
            aborted.add(item["cycle_id"])
        elif item["event"] == "vm_started":
            started[item["cycle_id"]] = item["time_unix"]
        elif item["event"] == "vm_deallocated":
            cycle_id = item["cycle_id"]
            if cycle_id not in started or cycle_id in closed:
                raise ValueError(f"cycle ledger has unmatched deallocation: {cycle_id}")
            spent += max(0.0, item["time_unix"] - started[cycle_id]) / 3600
            closed.add(cycle_id)
            if item.get("reason") == "pretrial_failure":
                pretrial_failures.add(cycle_id)
        elif item["event"] == "archive_confirmed":
            archive_confirmed.add(item["cycle_id"])
    return {"spent_vm_hours_estimate": spent,
            "remaining_vm_hours_estimate": max_hours - spent,
            "unresolved_start_requests": sorted(requested - set(started) - aborted),
            "unclosed_cycles": sorted(set(started) - closed),
            "unconfirmed_archives": sorted(closed - archive_confirmed - pretrial_failures)}


def ssh_options(settings: dict) -> list[str]:
    result = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
              "-o", "StrictHostKeyChecking=accept-new"]
    if settings.get("ssh_identity_file"):
        result.extend(["-i", str(Path(settings["ssh_identity_file"]).expanduser())])
    return result


def ssh(settings: dict, host: str, script: str, *, run=command) -> str:
    target = f"{settings['ssh_user']}@{host}"
    return run("ssh", *ssh_options(settings), target, f"bash -lc {shlex.quote(script)}")


def public_ip(settings: dict, *, run=command) -> str:
    address = run("az", "vm", "show", "--resource-group", settings["resource_group"],
                  "--name", settings["vm_name"], "--show-details", "--query", "publicIps",
                  "--output", "tsv")
    if not re.fullmatch(r"[0-9A-Fa-f:.]+", address):
        raise RuntimeError("VM needs a reachable public IP or a private-network controller")
    return address


def wait_for_ssh(settings: dict, *, run=command, timeout=600) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            host = public_ip(settings, run=run)
            if ssh(settings, host, "true", run=run) == "":
                return host
        except RuntimeError:
            pass
        time.sleep(5)
    raise TimeoutError("H100 started but SSH was not reachable within ten minutes")


def stage_commit(settings: dict, host: str, commit: str, *, run=command) -> None:
    remote_repo = settings["remote_repo"]
    bundle_remote = remote_repo + ".incoming.bundle"
    with tempfile.TemporaryDirectory(prefix="terminus-azure-bundle-") as directory:
        bundle = Path(directory) / "candidate.bundle"
        run("git", "bundle", "create", str(bundle), "HEAD")
        ssh(settings, host, f"mkdir -p {shlex.quote(str(Path(remote_repo).parent))}", run=run)
        run("scp", *ssh_options(settings), str(bundle),
            f"{settings['ssh_user']}@{host}:{bundle_remote}")
    script = ("set -euo pipefail; "
              f"if [ ! -d {shlex.quote(remote_repo + '/.git')} ]; then "
              f"git init -q {shlex.quote(remote_repo)}; fi; "
              f"test -z \"$(git -C {shlex.quote(remote_repo)} status --porcelain)\"; "
              f"git -C {shlex.quote(remote_repo)} fetch -q {shlex.quote(bundle_remote)} HEAD; "
              f"git -C {shlex.quote(remote_repo)} checkout -q --detach {shlex.quote(commit)}; "
              f"test \"$(git -C {shlex.quote(remote_repo)} rev-parse HEAD)\" = {shlex.quote(commit)}; "
              f"rm -f {shlex.quote(bundle_remote)}")
    ssh(settings, host, script, run=run)


def remote_preflight(settings: dict, host: str, trial_id: str, stage: str,
                     *, run=command) -> None:
    mount = settings["persistent_mount"]
    ssh(settings, host, "nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100",
        run=run)
    foreign = ssh(settings, host,
                  "nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                  "| grep -E '^[0-9]+' || true", run=run)
    if foreign.strip():
        raise ForeignWorkloadError(f"other GPU processes started on H100: {foreign.strip()}")
    services = ssh(settings, host,
                   "systemctl list-units 'terminus-*' --state=running "
                   "--plain --no-legend --no-pager", run=run)
    if services.strip():
        raise ForeignWorkloadError(f"older Terminus services started on H100: {services.strip()}")
    contract_check = (
        "from pathlib import Path; from research.protocol import campaign; "
        f"c=campaign(Path({settings['remote_campaign']!r})); "
        f"assert Path(c['data_root']).resolve()==Path({settings['remote_data_root']!r}); "
        f"assert Path(c['results_root']).resolve()==Path({settings['remote_results_root']!r})")
    checks = ["set -euo pipefail", f"mountpoint -q {shlex.quote(mount)}",
              f"test -d {shlex.quote(settings['remote_data_root'])}",
              f"test -d {shlex.quote(settings['remote_results_root'])}",
              f"test -f {shlex.quote(settings['remote_campaign'])}",
              f"test -x {shlex.quote(settings['remote_python'])}",
              "command -v az >/dev/null", "sudo -n true",
              "az login --identity --output none",
              f"az storage container show --account-name {shlex.quote(settings['blob_account'])} "
              f"--name {shlex.quote(settings['blob_container'])} --auth-mode login --output none",
              f"cd {shlex.quote(settings['remote_repo'])}",
              f"{shlex.quote(settings['remote_python'])} -c {shlex.quote(contract_check)}"]
    environment = settings.get("remote_env", {})
    for directory in environment.get("PYTHONPATH", "").split(":"):
        if directory:
            checks.append(f"test -d {shlex.quote(directory)}")
    if environment.get("DG_JIT_NVCC_COMPILER"):
        checks.append(f"test -x {shlex.quote(environment['DG_JIT_NVCC_COMPILER'])}")
    if not (trial_id == "baseline" and stage == "screen"):
        integrity_check = (
            "from pathlib import Path; "
            "from research.protocol import campaign; "
            "from terminus.data import TokenManifest, TokenShard; "
            "from terminus.integrity import file_sha256; "
            f"c=campaign(Path({settings['remote_campaign']!r})); "
            "m=TokenManifest.load(Path(c['manifest'])); "
            "assert all(s.sha256 and file_sha256(TokenShard(s, c['data_root']).path) == "
            "s.sha256.lower() for s in m.shards), 'prepared shard checksum changed'")
        checks.append(f"{shlex.quote(settings['remote_python'])} -c "
                      f"{shlex.quote(integrity_check)}")
    ssh(settings, host, "; ".join(checks), run=run)
    if trial_id == "baseline" and stage == "screen":
        script = (f"cd {shlex.quote(settings['remote_repo'])}; "
                  f"{shlex.quote(settings['remote_python'])} prepare.py "
                  f"--campaign {shlex.quote(settings['remote_campaign'])}")
        ssh(settings, host, script, run=run)


def launch_worker(settings: dict, host: str, *, trial_id: str, stage: str,
                  cycle_id: str, qat_rationale: str | None,
                  trial_budget_seconds: float, archive_only: bool = False,
                  run=command) -> str:
    unit = f"terminus-ar-{cycle_id}"
    args = [settings["remote_python"], "-m", "research.azure_worker",
            "--campaign", settings["remote_campaign"], "--trial-id", trial_id,
            "--stage", stage, "--cycle-id", cycle_id, "--unit", unit,
            "--resource-group", settings["resource_group"], "--vm-name", settings["vm_name"],
            "--blob-account", settings["blob_account"],
            "--blob-container", settings["blob_container"],
            "--blob-prefix", f"{settings['blob_prefix'].strip('/')}/{cycle_id}",
            "--verification-parent", settings["verification_parent"],
            "--trial-budget-seconds", str(trial_budget_seconds)]
    if qat_rationale:
        args.extend(["--qat-rationale", qat_rationale])
    if archive_only:
        args.append("--archive-only")
    script = shlex.join(["sudo", "-n", "systemd-run", f"--unit={unit}",
                         f"--uid={settings['ssh_user']}",
                         f"--working-directory={settings['remote_repo']}",
                         "--property=Restart=no", "--property=TimeoutStopSec=180",
                         f"--setenv=HOME=/home/{settings['ssh_user']}",
                         "--setenv=PYTHONUNBUFFERED=1",
                         *(f"--setenv={key}={value}" for key, value in
                           settings.get("remote_env", {}).items()),
                         *args])
    ssh(settings, host, script, run=run)
    return unit


def confirm_archive(settings: dict, cycle_id: str, trial_id: str, *, run=command) -> dict:
    """Controller-side proof that the deallocated worker published its receipt."""
    prefix = f"{settings['blob_prefix'].strip('/')}/{cycle_id}"
    destination = Path(settings["local_control_root"]) / "receipts" / cycle_id
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("artifact_receipt.json", "cycle_result.json"):
        run("az", "storage", "blob", "download", "--account-name", settings["blob_account"],
            "--container-name", settings["blob_container"], "--name", f"{prefix}/{name}",
            "--file", str(destination / name), "--auth-mode", "login",
            "--overwrite", "true", "--no-progress", "--output", "none")
    receipt = read_json(destination / "artifact_receipt.json")
    result = read_json(destination / "cycle_result.json")
    expected = receipt.get("files", {}).get("cycle_result.json", {}).get("sha256")
    if result.get("cycle_id") != cycle_id or result.get("trial_id") != trial_id or \
            expected != sha256(destination / "cycle_result.json"):
        raise ValueError("deallocated VM archive receipt does not match this cycle")
    return {"blob_prefix": prefix, "receipt_sha256": sha256(destination / "artifact_receipt.json"),
            "trial_result": result.get("trial_result"),
            "worker_failure": result.get("worker_failure")}


def fetch_archive(settings: dict, cycle_id: str, *, include_checkpoints: bool = False,
                  run=command) -> dict:
    """Download verified analysis files while the H100 is deallocated."""
    receipt_path = (Path(settings["local_control_root"]) / "receipts" / cycle_id /
                    "artifact_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError("fetch needs a controller-confirmed archive receipt")
    receipt = read_json(receipt_path)
    prefix = f"{settings['blob_prefix'].strip('/')}/{cycle_id}"
    destination = Path(settings["local_control_root"]) / "analysis" / cycle_id
    files = receipt["files"]
    selected = {name: info for name, info in files.items()
                if include_checkpoints or not ("checkpoints" in Path(name).parts and
                                               Path(name).suffix in {".pt", ".safetensors"})}
    for name, info in sorted(selected.items()):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe archive path: {name}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256(target) == info["sha256"]:
            continue
        run("az", "storage", "blob", "download", "--account-name", settings["blob_account"],
            "--container-name", settings["blob_container"], "--name", f"{prefix}/{name}",
            "--file", str(target), "--auth-mode", "login", "--overwrite", "true",
            "--no-progress", "--output", "none")
        if sha256(target) != info["sha256"]:
            raise ValueError(f"downloaded analysis file differs from receipt: {name}")
    return {"analysis_path": str(destination), "files": len(selected),
            "checkpoints_included": include_checkpoints}


def request_stop(settings: dict, cycle_id: str, *, run=command) -> dict:
    """Signal only the active trial child; the worker still archives and shuts down."""
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z[0-9a-f]{8}", cycle_id):
        raise ValueError("invalid Azure cycle ID")
    state = power_state(settings, run=run)
    if state == "PowerState/deallocated":
        return {"cycle_id": cycle_id, "vm_power_state": state, "signal_sent": False}
    host = wait_for_ssh(settings, run=run)
    unit = f"terminus-ar-{cycle_id}"
    pid = ssh(settings, host,
              shlex.join(["sudo", "-n", "systemctl", "show", unit,
                          "--property=MainPID", "--value"]), run=run)
    if not pid.isdecimal() or int(pid) <= 0:
        return {"cycle_id": cycle_id, "vm_power_state": state,
                "signal_sent": False, "note": "worker has no active main PID; inspect archive/shutdown state"}
    script = (f"if sudo -n pgrep -P {pid} -f 'terminus.train|terminus.preflight|research.evaluate' "
              f">/dev/null; then sudo -n pkill -TERM -P {pid} -f "
              "'terminus.train|terminus.preflight|research.evaluate'; echo signalled; "
              "else echo no_trial_child; fi")
    result = ssh(settings, host, script, run=run)
    return {"cycle_id": cycle_id, "vm_power_state": state,
            "signal_sent": "signalled" in result,
            "note": "the worker must still archive, verify and deallocate; do not stop it"}


def repair_cycle(settings: dict, old_cycle_id: str, *, run=command) -> dict:
    """Recover a failed archive without rerunning the experiment or rebooting."""
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z[0-9a-f]{8}", old_cycle_id):
        raise ValueError("invalid Azure cycle ID")
    root = Path(settings["local_control_root"])
    log_path = root / "azure_cycles.jsonl"
    events = ledger(log_path)
    starts = [row for row in events if row.get("cycle_id") == old_cycle_id and
              row["event"] == "start_requested"]
    launched = any(row.get("cycle_id") == old_cycle_id and row["event"] == "worker_started"
                   for row in events)
    if len(starts) != 1 or not launched:
        raise ValueError("repair needs one previously started experiment worker")
    if any(row.get("cycle_id") == old_cycle_id and row["event"] == "vm_deallocated"
           for row in events):
        raise ValueError("VM is already deallocated; reconcile its Blob receipt instead")
    if power_state(settings, run=run) != "PowerState/running":
        raise RuntimeError("archive repair needs the still-running H100")
    host = wait_for_ssh(settings, run=run)
    old_unit = f"terminus-ar-{old_cycle_id}"
    state = ssh(settings, host, shlex.join(["sudo", "-n", "systemctl", "show", old_unit,
                                           "--property=ActiveState", "--value"]), run=run)
    if state.strip() not in {"inactive", "failed"}:
        raise RuntimeError(f"old worker is still active; do not archive a changing trial: {state}")
    foreign = ssh(settings, host,
                  "nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                  "| grep -E '^[0-9]+' || true", run=run)
    if foreign.strip():
        raise ForeignWorkloadError(f"GPU process still active during repair: {foreign.strip()}")
    recovery_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + secrets.token_hex(4)
    original = starts[0]
    unit = launch_worker(settings, host, trial_id=original["trial_id"],
                         stage=original["stage"], cycle_id=recovery_id,
                         qat_rationale=None, trial_budget_seconds=0,
                         archive_only=True, run=run)
    append_event(log_path, {"event": "archive_repair_started", "cycle_id": old_cycle_id,
                            "recovery_cycle_id": recovery_id, "unit": unit,
                            "time_unix": time.time()})
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        time.sleep(20)
        if power_state(settings, run=run) == "PowerState/deallocated":
            append_event(log_path, {"event": "vm_deallocated", "cycle_id": old_cycle_id,
                                    "time_unix": time.time(), "recovery_cycle_id": recovery_id})
            archive = confirm_archive(settings, recovery_id, original["trial_id"], run=run)
            append_event(log_path, {"event": "archive_confirmed", "cycle_id": old_cycle_id,
                                    "time_unix": time.time(),
                                    "recovery_cycle_id": recovery_id,
                                    "receipt_sha256": archive["receipt_sha256"]})
            analysis = fetch_archive(settings, recovery_id, run=run)
            return {"repaired_cycle_id": old_cycle_id, "archive_cycle_id": recovery_id,
                    **archive, "analysis": analysis, "deallocated": True}
        try:
            state = ssh(settings, host, shlex.join(["sudo", "-n", "systemctl", "show", unit,
                                                   "--property=ActiveState", "--property=Result",
                                                   "--no-pager"]), run=run)
        except RuntimeError:
            continue
        if "ActiveState=failed" in state:
            raise RuntimeError(f"archive repair failed; VM retained: {state}")
    raise TimeoutError("archive repair timed out; VM retained")


def reconcile_cycle(settings: dict, cycle_id: str, *, run=command) -> dict:
    """Recover controller bookkeeping after a disconnect; never infer Blob safety."""
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z[0-9a-f]{8}", cycle_id):
        raise ValueError("invalid Azure cycle ID")
    log_path = Path(settings["local_control_root"]) / "azure_cycles.jsonl"
    events = ledger(log_path)
    starts = [row for row in events if row.get("cycle_id") == cycle_id and
              row["event"] == "start_requested"]
    if len(starts) != 1:
        raise ValueError("controller has no unique start record for this cycle")
    if power_state(settings, run=run) != "PowerState/deallocated":
        raise RuntimeError("VM is not deallocated; inspect worker or use repair")
    if not any(row.get("cycle_id") == cycle_id and row["event"] == "worker_started"
               for row in events):
        raise RuntimeError("no worker-start record; manually inspect pretrial state")
    repairs = [row["recovery_cycle_id"] for row in events if row.get("cycle_id") == cycle_id
               and row["event"] == "archive_repair_started"]
    archive_id = repairs[-1] if repairs else cycle_id
    archive = confirm_archive(settings, archive_id, starts[0]["trial_id"], run=run)
    if not any(row.get("cycle_id") == cycle_id and row["event"] == "vm_deallocated"
               for row in events):
        append_event(log_path, {"event": "vm_deallocated", "cycle_id": cycle_id,
                                "time_unix": time.time(), "reconciled": True,
                                "archive_cycle_id": archive_id})
    if not any(row.get("cycle_id") == cycle_id and row["event"] == "archive_confirmed"
               for row in events):
        append_event(log_path, {"event": "archive_confirmed", "cycle_id": cycle_id,
                                "time_unix": time.time(),
                                "receipt_sha256": archive["receipt_sha256"]})
    try:
        analysis = fetch_archive(settings, archive_id, run=run)
    except Exception as exc:
        analysis = {"error": f"{type(exc).__name__}: {exc}"}
    return {"cycle_id": cycle_id, "archive_cycle_id": archive_id,
            **archive, "analysis": analysis, "deallocated": True,
            "note": "reconciled VM-on time is conservative if shutdown preceded this check"}


def run_cycle(settings: dict, *, trial_id: str, stage: str,
              qat_rationale: str | None = None, run=command) -> dict:
    if any("YOUR_PERSISTENT_DISK" in settings.get(field, "")
           for field in ("remote_campaign", "remote_data_root", "remote_results_root",
                         "persistent_mount", "verification_parent")) or \
            settings.get("local_control_root", "").startswith("/absolute/path/"):
        raise ValueError("replace Azure-cycle example paths before starting the H100")
    if not SAFE_NAME.fullmatch(trial_id) or stage not in {"screen", "full", "qat_probe"}:
        raise ValueError("invalid trial id or stage")
    if (stage == "qat_probe") != bool(qat_rationale):
        raise ValueError("QAT probes need a rationale; other stages cannot have one")
    commit = assert_clean_git()
    root = Path(settings["local_control_root"])
    log_path = root / "azure_cycles.jsonl"
    budget = budget_state(ledger(log_path), float(settings["max_vm_hours"]))
    if (budget["unresolved_start_requests"] or budget["unclosed_cycles"] or
            budget["unconfirmed_archives"]):
        raise RuntimeError(f"reconcile unfinished Azure cycles before starting: {budget}")
    if budget["remaining_vm_hours_estimate"] <= 0.5:
        raise RuntimeError("Azure VM-on budget is exhausted or lacks archive reserve")
    if power_state(settings, run=run) != "PowerState/deallocated":
        raise RuntimeError("refusing to take over an H100 that was already running")
    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + secrets.token_hex(4)
    requested = time.time()
    append_event(log_path, {"event": "start_requested", "cycle_id": cycle_id,
                            "trial_id": trial_id, "stage": stage, "commit": commit,
                            "time_unix": requested})
    try:
        run("az", "vm", "start", "--resource-group", settings["resource_group"],
            "--name", settings["vm_name"], "--output", "none")
    except Exception:
        if power_state(settings, run=run) == "PowerState/deallocated":
            append_event(log_path, {"event": "start_aborted", "cycle_id": cycle_id,
                                    "time_unix": time.time()})
        raise
    started = requested
    append_event(log_path, {"event": "vm_started", "cycle_id": cycle_id,
                            "time_unix": started, "ready_unix": time.time()})
    worker_launched = False
    try:
        host = wait_for_ssh(settings, run=run)
        stage_commit(settings, host, commit, run=run)
        remote_preflight(settings, host, trial_id, stage, run=run)
        archive_reserve = float(settings.get("archive_reserve_seconds", 1800))
        trial_budget = (budget["remaining_vm_hours_estimate"] * 3600 -
                        (time.time() - started) - archive_reserve)
        if trial_budget < 300:
            raise RuntimeError("VM budget cannot cover another trial plus verified archive")
        unit = launch_worker(settings, host, trial_id=trial_id, stage=stage,
                             cycle_id=cycle_id, qat_rationale=qat_rationale,
                             trial_budget_seconds=trial_budget, run=run)
        worker_launched = True
        append_event(log_path, {"event": "worker_started", "cycle_id": cycle_id,
                                "unit": unit, "time_unix": time.time()})
        deadline = time.monotonic() + policy()["max_trial_seconds"] + 7200
        while time.monotonic() < deadline:
            time.sleep(20)
            if power_state(settings, run=run) == "PowerState/deallocated":
                stopped = time.time()
                append_event(log_path, {"event": "vm_deallocated", "cycle_id": cycle_id,
                                        "time_unix": stopped})
                archive = confirm_archive(settings, cycle_id, trial_id, run=run)
                append_event(log_path, {"event": "archive_confirmed", "cycle_id": cycle_id,
                                        "time_unix": time.time(),
                                        "receipt_sha256": archive["receipt_sha256"]})
                try:
                    analysis = fetch_archive(settings, cycle_id, run=run)
                except Exception as exc:
                    analysis = {"error": f"{type(exc).__name__}: {exc}",
                                "retry": f"python -m research.azure_cycle fetch --config <config> --cycle-id {cycle_id}"}
                return {"cycle_id": cycle_id, "trial_id": trial_id, "stage": stage,
                        "commit": commit, "vm_on_hours_estimate": (stopped - started) / 3600,
                        **archive,
                        "analysis": analysis,
                        "deallocated": True}
            try:
                state = ssh(settings, host, shlex.join(["sudo", "-n", "systemctl", "show", unit,
                                                       "--property=ActiveState", "--property=Result",
                                                       "--property=ExecMainStatus", "--no-pager"]), run=run)
            except RuntimeError:
                # SSH can disappear while Azure processes the worker's shutdown.
                continue
            if "ActiveState=failed" in state or ("ActiveState=inactive" in state and
                                                  "Result=success" not in state):
                raise RuntimeError(f"worker stopped before verified shutdown: {state}")
        raise TimeoutError("worker exceeded the trial + archive deadline; VM retained for recovery")
    except Exception as exc:
        if isinstance(exc, ForeignWorkloadError):
            append_event(log_path, {"event": "foreign_gpu_workload", "cycle_id": cycle_id,
                                    "time_unix": time.time(), "note": str(exc)})
        elif not worker_launched:
            # No trial/checkpoint has begun. Avoid paying for an idle preflight VM.
            run("az", "vm", "deallocate", "--resource-group", settings["resource_group"],
                "--name", settings["vm_name"], "--output", "none")
            append_event(log_path, {"event": "vm_deallocated", "cycle_id": cycle_id,
                                    "time_unix": time.time(), "reason": "pretrial_failure"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "run", "fetch", "stop",
                                             "repair", "reconcile"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--stage", choices=("screen", "full", "qat_probe"), default="screen")
    parser.add_argument("--qat-rationale")
    parser.add_argument("--cycle-id")
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    settings = config(args.config)
    if args.command == "inspect":
        result = {"vm_power_state": power_state(settings),
                  "budget": budget_state(ledger(Path(settings["local_control_root"]) /
                                                "azure_cycles.jsonl"),
                                         float(settings["max_vm_hours"]))}
    elif args.command == "fetch":
        if not args.cycle_id:
            parser.error("fetch needs --cycle-id")
        result = fetch_archive(settings, args.cycle_id,
                               include_checkpoints=args.include_checkpoints)
    elif args.command == "stop":
        if not args.cycle_id:
            parser.error("stop needs --cycle-id")
        result = request_stop(settings, args.cycle_id)
    elif args.command == "repair":
        if not args.cycle_id:
            parser.error("repair needs --cycle-id")
        result = repair_cycle(settings, args.cycle_id)
    elif args.command == "reconcile":
        if not args.cycle_id:
            parser.error("reconcile needs --cycle-id")
        result = reconcile_cycle(settings, args.cycle_id)
    else:
        if not args.trial_id:
            parser.error("run needs --trial-id")
        result = run_cycle(settings, trial_id=args.trial_id, stage=args.stage,
                           qat_rationale=args.qat_rationale)
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
