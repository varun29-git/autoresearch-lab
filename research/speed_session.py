"""One authorized, continuous Azure H100 speed-research session.

Kimi owns experiment selection. This small controller owns the five GPU-hour
allocation: it starts the VM once, leaves it on between experiments, and puts
an independent cutoff timer on the VM. Records stay on the persistent disk.
Nothing in this module starts a VM until ``open`` is explicitly invoked.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

from .azure_cycle import (SAFE_NAME, SAFE_REMOTE_PATH, append_event, command,
                          ledger, power_state, public_ip, ssh, stage_commit,
                          wait_for_ssh)
from .protocol import ROOT, assert_clean_git, read_json

CAP_SECONDS = 5 * 3600
SHUTDOWN_MARGIN_SECONDS = 120


def config(path: Path) -> dict:
    value = read_json(path)
    for name in ("resource_group", "vm_name", "ssh_user", "remote_repo",
                 "remote_results_root", "persistent_mount", "local_control_root"):
        if not value.get(name):
            raise ValueError(f"speed session config needs {name}")
    for name in ("resource_group", "vm_name", "ssh_user"):
        if not SAFE_NAME.fullmatch(value[name]):
            raise ValueError(f"unsafe {name}")
    for name in ("remote_repo", "remote_results_root", "persistent_mount"):
        if not SAFE_REMOTE_PATH.fullmatch(value[name]) or ".." in Path(value[name]).parts:
            raise ValueError(f"unsafe {name}")
    mount = Path(value["persistent_mount"])
    if not Path(value["remote_results_root"]).is_relative_to(mount):
        raise ValueError("results must be on the persistent mount")
    if mount.is_relative_to(Path("/mnt")) or mount.is_relative_to(Path("/tmp")):
        raise ValueError("the Azure temporary disk is not persistent")
    root = Path(value["local_control_root"]).expanduser().resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("the session ledger must be outside Git")
    value["local_control_root"] = str(root)
    return value


def usage(events: list[dict], now: float) -> dict:
    """Conservatively count VM-on seconds, including an unfinished allocation."""
    used, active = 0.0, None
    for event in events:
        if event["event"] == "allocated":
            if active is not None:
                raise ValueError("overlapping speed allocations")
            active = event
        elif event["event"] == "deallocated":
            if active is None or event["session_id"] != active["session_id"]:
                raise ValueError("unmatched speed deallocation")
            used += max(0.0, event["time_unix"] - active["time_unix"])
            active = None
    if active is not None:
        used += max(0.0, now - active["time_unix"])
    return {"used_seconds": used, "remaining_seconds": max(0.0, CAP_SECONDS - used),
            "active_session_id": active["session_id"] if active else None}


def log_path(settings: dict) -> Path:
    return Path(settings["local_control_root"]) / "speed_allocations.jsonl"


def observe(settings: dict, *, run=command, now=time.time) -> dict:
    path = log_path(settings)
    events = ledger(path)
    state = usage(events, now())
    power = power_state(settings, run=run)
    if state["active_session_id"] and power == "PowerState/deallocated":
        append_event(path, {"event": "deallocated", "session_id": state["active_session_id"],
                            "time_unix": now(), "reason": "observed_off"})
        state = usage(ledger(path), now())
    events = ledger(path)
    last_ready = next((event["setup_seconds"] for event in reversed(events)
                       if event["event"] == "ready"), None)
    last_shutdown = next((event.get("shutdown_seconds") for event in reversed(events)
                          if event["event"] == "deallocated"), None)
    return {**state, "vm_power_state": power,
            "last_boot_ready_seconds": last_ready,
            "last_shutdown_seconds": last_shutdown,
            "cutoff_margin_seconds": SHUTDOWN_MARGIN_SECONDS}


def cutoff_command(settings: dict, session_id: str, seconds: int) -> str:
    """A transient systemd timer survives an SSH or agent-process failure."""
    unit = f"terminus-speed-cutoff-{session_id}"
    login = shlex.join(["az", "login", "--identity", "--output", "none"])
    deallocate = shlex.join(["az", "vm", "deallocate", "--resource-group",
                             settings["resource_group"], "--name", settings["vm_name"],
                             "--output", "none"])
    body = f"for attempt in 1 2 3; do {login} && {deallocate} && exit 0; sleep 10; done; exit 1"
    return shlex.join(["sudo", "-n", "systemd-run", f"--unit={unit}",
                       f"--on-active={seconds}s", "--timer-property=AccuracySec=1s",
                       "/bin/bash", "-lc", body])


def open_session(settings: dict, *, run=command, now=time.time,
                 connect=wait_for_ssh, remote=ssh) -> dict:
    state = observe(settings, run=run, now=now)
    if state["active_session_id"] or state["vm_power_state"] != "PowerState/deallocated":
        raise RuntimeError("the research VM is already allocated or running")
    if state["remaining_seconds"] <= SHUTDOWN_MARGIN_SECONDS + 300:
        raise RuntimeError("less than one useful short experiment remains in the five-hour cap")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + secrets.token_hex(3)
    path = log_path(settings)
    requested_at = now()
    append_event(path, {"event": "allocated", "session_id": session_id,
                        "time_unix": requested_at})
    try:
        run("az", "vm", "start", "--resource-group", settings["resource_group"],
            "--name", settings["vm_name"], "--output", "none")
        host = connect(settings, run=run)
        # This is a storage check, not a model/production preflight.
        remote(settings, host,
               f"mountpoint -q {shlex.quote(settings['persistent_mount'])} && "
               f"test -d {shlex.quote(settings['remote_results_root'])}", run=run)
        remaining = usage(ledger(path), now())["remaining_seconds"]
        timer_seconds = int(remaining - SHUTDOWN_MARGIN_SECONDS)
        if timer_seconds <= 0:
            raise RuntimeError("VM startup consumed the remaining research budget")
        remote(settings, host, cutoff_command(settings, session_id, timer_seconds), run=run)
        remote(settings, host,
               shlex.join(["sudo", "-n", "systemctl", "is-active",
                           f"terminus-speed-cutoff-{session_id}.timer"]), run=run)
        append_event(path, {"event": "ready", "session_id": session_id,
                            "time_unix": now(), "setup_seconds": max(0, now() - requested_at)})
    except Exception:
        # If setup fails, release the VM rather than bill idle time.
        try:
            run("az", "vm", "deallocate", "--resource-group", settings["resource_group"],
                "--name", settings["vm_name"], "--output", "none")
            append_event(path, {"event": "deallocated", "session_id": session_id,
                                "time_unix": now(), "reason": "startup_failure"})
        finally:
            raise
    return {**observe(settings, run=run, now=now), "ssh_target": f"{settings['ssh_user']}@{host}"}


def close_session(settings: dict, *, reason: str, run=command, now=time.time) -> dict:
    state = observe(settings, run=run, now=now)
    if not state["active_session_id"]:
        return state
    if reason not in {"goal", "budget", "idle"}:
        raise ValueError("close reason must be goal, budget, or idle")
    shutdown_started = now()
    run("az", "vm", "deallocate", "--resource-group", settings["resource_group"],
        "--name", settings["vm_name"], "--output", "none")
    if power_state(settings, run=run) != "PowerState/deallocated":
        raise RuntimeError("Azure has not confirmed deallocation")
    append_event(log_path(settings), {"event": "deallocated",
                                      "session_id": state["active_session_id"],
                                      "time_unix": now(), "reason": reason,
                                      "shutdown_seconds": max(0, now() - shutdown_started)})
    return observe(settings, run=run, now=now)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("open", "status", "stage", "exec", "close"):
        action = actions.add_parser(name)
        action.add_argument("--config", type=Path, required=True)
        if name == "close":
            action.add_argument("--reason", choices=("goal", "budget", "idle"), required=True)
        if name == "exec":
            action.add_argument("remote_command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    settings = config(args.config)
    if args.action == "open":
        result = open_session(settings)
    elif args.action == "close":
        result = close_session(settings, reason=args.reason)
    else:
        result = observe(settings)
        if args.action in {"stage", "exec"}:
            if not result["active_session_id"] or result["remaining_seconds"] <= SHUTDOWN_MARGIN_SECONDS:
                raise RuntimeError("no usable authorized speed session remains")
            host = public_ip(settings)
            if args.action == "stage":
                commit = assert_clean_git()
                os.chdir(ROOT)
                stage_commit(settings, host, commit)
                result = {**result, "staged_commit": commit}
            else:
                command_words = args.remote_command
                if command_words and command_words[0] == "--":
                    command_words = command_words[1:]
                if not command_words:
                    parser.error("exec needs a remote command after --")
                result = {**result, "remote_output": ssh(settings, host,
                           shlex.join(command_words))}
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
