"""Campaign allocation ledger: one append-only account for the 50-hour cap.

The ledger charges allocated wall-clock time regardless of whether the GPU is
busy: an allocation opens when the VM starts billing and closes when it is
deallocated. Every trial, probe, evaluation, checkpoint, retry and idle gap
happens inside an allocation interval. Activities are annotations for the
handback; the charged quantity is always the allocation interval.

Resume never resets elapsed time: the ledger file is the clock. A malformed
ledger fails closed. An individual trainer time limit is not a budget
control; this is.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from . import campaign


class BudgetExhausted(RuntimeError):
    """Remaining allocation cannot cover declared work plus the reserve."""


class LedgerError(RuntimeError):
    """The ledger is missing, malformed or internally inconsistent."""


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


class Ledger:
    def __init__(self, path, *, cap_hours: float = campaign.MAX_ALLOCATED_HOURS, clock=time.time):
        if cap_hours <= 0 or not math.isfinite(cap_hours) or cap_hours > campaign.MAX_ALLOCATED_HOURS:
            raise ValueError("cap must be positive and cannot exceed the 50-hour allocation")
        self.path = Path(path)
        self.cap_hours = float(cap_hours)
        self.clock = clock
        self._records = self._read()

    def _read(self) -> list:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerError(f"malformed ledger line {line_number}: {exc}") from exc
                if (record.get("record") not in ("allocation_open", "allocation_close", "activity")
                        or not isinstance(record.get("time_unix"), (int, float))
                        or not math.isfinite(record["time_unix"])):
                    raise LedgerError(f"invalid ledger record at line {line_number}")
                records.append(record)
        opened = sum(1 for r in records if r["record"] == "allocation_open")
        closed = sum(1 for r in records if r["record"] == "allocation_close")
        if not 0 <= opened - closed <= 1:
            raise LedgerError("ledger allocation intervals are unbalanced")
        return records

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(record)

    def open_allocation(self, *, note: str = "") -> None:
        if self.allocation_open:
            raise LedgerError("an allocation interval is already open")
        self._append({"record": "allocation_open", "time_unix": self.clock(), "note": note})

    def close_allocation(self, *, note: str = "") -> None:
        if not self.allocation_open:
            raise LedgerError("no allocation interval is open")
        self._append({"record": "allocation_close", "time_unix": self.clock(), "note": note})

    def log_activity(self, kind: str, detail: dict | None = None) -> None:
        if not kind:
            raise ValueError("activity kind is required")
        self._append({"record": "activity", "time_unix": self.clock(),
                      "kind": kind, "detail": detail or {}})

    @property
    def allocation_open(self) -> bool:
        opened = closed = False
        for record in self._records:
            if record["record"] == "allocation_open":
                opened, closed = True, False
            elif record["record"] == "allocation_close":
                closed = True
        return opened and not closed

    def consumed_seconds(self, *, now: float | None = None) -> float:
        total, start = 0.0, None
        for record in self._records:
            if record["record"] == "allocation_open":
                start = record["time_unix"]
            elif record["record"] == "allocation_close":
                if start is None:
                    raise LedgerError("allocation closed before it opened")
                total += record["time_unix"] - start
                start = None
        if start is not None:
            total += (now if now is not None else self.clock()) - start
        if total < 0:
            raise LedgerError("ledger time moved backwards")
        return total

    def consumed_hours(self, *, now: float | None = None) -> float:
        return self.consumed_seconds(now=now) / 3600.0

    def remaining_hours(self, *, now: float | None = None) -> float:
        return self.cap_hours - self.consumed_hours(now=now)

    def require(self, needed_hours: float, *, label: str,
                reserve_hours: float = campaign.EMERGENCY_RESERVE_HOURS,
                now: float | None = None) -> dict:
        """Fail closed unless declared work plus the emergency reserve fits."""
        needed_hours = _finite_nonnegative("needed_hours", needed_hours)
        reserve_hours = _finite_nonnegative("reserve_hours", reserve_hours)
        remaining = self.remaining_hours(now=now)
        status = {"label": label, "needed_hours": needed_hours,
                  "reserve_hours": reserve_hours,
                  "remaining_hours": remaining,
                  "fits": needed_hours + reserve_hours <= remaining}
        if not status["fits"]:
            raise BudgetExhausted(
                f"{label}: {needed_hours:.4f} h of work plus {reserve_hours:.4f} h reserve "
                f"exceeds {remaining:.4f} h remaining allocation")
        return status

    def status(self, *, now: float | None = None) -> dict:
        consumed = self.consumed_hours(now=now)
        return {"schema": 1, "cap_hours": self.cap_hours,
                "consumed_hours": consumed,
                "remaining_hours": self.cap_hours - consumed,
                "allocation_open": self.allocation_open,
                "records": len(self._records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--cap-hours", type=float, default=campaign.MAX_ALLOCATED_HOURS)
    command = parser.add_subparsers(dest="command", required=True)
    for name in ("open", "close", "status"):
        command.add_parser(name)
    activity = command.add_parser("activity")
    activity.add_argument("--kind", required=True)
    activity.add_argument("--detail", default="{}")
    require = command.add_parser("require")
    require.add_argument("--hours", type=float, required=True)
    require.add_argument("--label", required=True)
    require.add_argument("--reserve-hours", type=float, default=campaign.EMERGENCY_RESERVE_HOURS)
    args = parser.parse_args()

    ledger = Ledger(args.ledger, cap_hours=args.cap_hours)
    try:
        if args.command == "open":
            ledger.open_allocation()
        elif args.command == "close":
            ledger.close_allocation()
        elif args.command == "activity":
            ledger.log_activity(args.kind, json.loads(args.detail))
        elif args.command == "require":
            print(json.dumps(ledger.require(args.hours, label=args.label,
                                            reserve_hours=args.reserve_hours),
                             indent=2, sort_keys=True))
            return 0
    except (BudgetExhausted, LedgerError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(ledger.status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
