"""Read local operating health; never launch, signal, or stop a campaign."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time


def last_event(path):
    if not path.exists():
        return None
    # Read a bounded tail and discard a partial first line.
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - 1024 * 1024))
        if size > 1024 * 1024:
            handle.readline()
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def status(run_dir, *, min_free_bytes, max_idle_seconds, now=None):
    if not math.isfinite(min_free_bytes) or not math.isfinite(max_idle_seconds) or min_free_bytes < 0 or max_idle_seconds <= 0:
        raise ValueError("invalid operating thresholds")
    root = Path(run_dir)
    event = last_event(root / "events.jsonl")
    incident = last_event(root / "incidents.jsonl")
    free = shutil.disk_usage(root).free
    idle = (time.time() if now is None else now) - event["time_unix"] if event else None
    alerts = []
    if free < min_free_bytes:
        alerts.append("low_disk_space")
    if event is None:
        alerts.append("no_training_events")
    elif event.get("event") not in {"complete", "stopped"} and idle > max_idle_seconds:
        alerts.append("training_log_stale")
    if incident and (event is None or incident["time_unix"] >= event["time_unix"]):
        alerts.append("unresolved_incident")
    return {"alerts": alerts, "last_event": event, "last_incident": incident,
            "free_bytes": free, "seconds_since_last_event": idle,
            "scope": "local logs and disk only; no process liveness or cluster certification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-free-gb", type=float, required=True)
    parser.add_argument("--max-idle-seconds", type=float, required=True)
    args = parser.parse_args()
    result = status(args.run_dir, min_free_bytes=args.min_free_gb * 1e9, max_idle_seconds=args.max_idle_seconds)
    print(json.dumps(result, allow_nan=False))
    if result["alerts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
