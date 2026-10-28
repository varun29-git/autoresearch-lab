"""Print/save a CPU-only campaign forecast; never prepare or launch training."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_TOKENS = 2_097_152
PROXY_WIDTH = 768
FULL_WIDTH = 2304
MAX_ALLOCATED_HOURS = 50.0
# The historical anchor remains separate from the production-.003 hypothesis.
TRIALS = (
    ("H", 0.017, 0.006),
    ("R0", 0.017, 0.009),
    ("R1", 0.017, 0.00255),
    ("R2", 0.0082, 0.009),
    ("R3", 0.0082, 0.00255),
)


def positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def accumulation(world_size: int, microbatch: int,
                 global_batch: int = GLOBAL_BATCH_TOKENS) -> int:
    micro_tokens = (positive_int("world_size", world_size)
                    * positive_int("microbatch", microbatch) * SEQUENCE_LENGTH)
    steps, remainder = divmod(positive_int("global_batch", global_batch), micro_tokens)
    if remainder or steps < 1:
        raise ValueError("global_batch must be a positive multiple of microstep tokens")
    return steps


def schedule(total_updates: int, warmup_updates: int, decay_updates: int,
             global_batch: int = GLOBAL_BATCH_TOKENS) -> dict:
    for name, value in (("total_updates", total_updates),
                        ("warmup_updates", warmup_updates),
                        ("decay_updates", decay_updates), ("global_batch", global_batch)):
        positive_int(name, value)
    stable_updates = total_updates - warmup_updates - decay_updates
    if stable_updates <= 0:
        raise ValueError("warmup and decay must leave a positive stable segment")
    return {
        "schedule": "wsd", "decay_style": "cosine", "final_lr_fraction": 0.1,
        "global_batch_tokens": global_batch,
        "total_tokens": total_updates * global_batch,
        "total_updates": total_updates,
        "warmup_tokens": warmup_updates * global_batch,
        "stable_tokens": stable_updates * global_batch,
        "decay_tokens": decay_updates * global_batch,
        "warmup_updates": warmup_updates, "stable_updates": stable_updates,
        "decay_updates": decay_updates,
        "boundary_updates": [warmup_updates, total_updates - decay_updates, total_updates],
        "boundary_tokens": [warmup_updates * global_batch,
                            (total_updates - decay_updates) * global_batch,
                            total_updates * global_batch],
    }


def finite_number(name: str, value: float, *, allow_zero: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def make_plan(*, proxy_tps: float = 200_000, full_tps: float = 40_000,
              reserve_hours: float = 8, cap_hours: float = MAX_ALLOCATED_HOURS,
              include_optional_batch: bool = False, proxy_microbatch: int = 4) -> dict:
    proxy_tps = finite_number("proxy_tps", proxy_tps)
    full_tps = finite_number("full_tps", full_tps)
    reserve_hours = finite_number("reserve_hours", reserve_hours, allow_zero=True)
    cap_hours = finite_number("cap_hours", cap_hours)
    if cap_hours > MAX_ALLOCATED_HOURS:
        raise ValueError("cap_hours cannot exceed the user's 50-hour allocation")
    if type(include_optional_batch) is not bool:
        raise ValueError("include_optional_batch must be boolean")
    proxy_schedule = schedule(2384, 238, 477)
    full_schedule = schedule(480, 48, 96)
    proxy_count = len(TRIALS) + int(include_optional_batch)
    one_proxy_hours = proxy_schedule["total_tokens"] / proxy_tps / 3600
    all_proxy_hours = one_proxy_hours * proxy_count
    full_hours = full_schedule["total_tokens"] / full_tps / 3600
    total_hours = all_proxy_hours + full_hours + reserve_hours
    if not math.isfinite(total_hours):
        raise ValueError("rates produce an unrepresentable forecast; check their units")
    full_time_available = cap_hours - reserve_hours - all_proxy_hours
    return {
        "schema": 1,
        "status": "planning_only_not_launch_clearance",
        "protocol": "production/kimi_h100_5b/RUNBOOK.md",
        "allocated_gpus_assumed": 1,
        "input_rates_are_not_measurement_receipts": True,
        "inputs": {"proxy_tokens_per_second": proxy_tps,
                   "full_tokens_per_second": full_tps,
                   "setup_check_contingency_hours": reserve_hours,
                   "allocated_hour_cap": cap_hours},
        "proxy_execution_candidate": {
            "world_size": 1, "microbatch_sequences": proxy_microbatch,
            "sequence_length": SEQUENCE_LENGTH,
            "accumulation_steps": accumulation(1, proxy_microbatch),
            "fit_and_speed_measured": False,
        },
        "proxy_schedule": proxy_schedule,
        "mandatory_proxy_trials_in_order": [
            {"id": name, "proxy_muon_lr": muon, "proxy_adamw_lr": adamw,
             "transferred_muon_lr": muon,
             "transferred_adamw_lr": float(Decimal(str(adamw)) * PROXY_WIDTH / FULL_WIDTH)}
            for name, muon, adamw in TRIALS
        ],
        "width_transfer": {"proxy_width": PROXY_WIDTH, "full_width": FULL_WIDTH,
                           "adamw_multiplier": PROXY_WIDTH / FULL_WIDTH,
                           "muon_and_small_group_multiplier": 1},
        "optional_batch_trial": {
            "included_in_forecast": include_optional_batch,
            "default_status": "deferred",
            "schedule": schedule(4768, 476, 954, GLOBAL_BATCH_TOKENS // 2),
            "requires_separate_measured_rate_if_execution_differs": True,
        },
        "full_confirmation": {"number_of_candidates": 1, "selection": "pending",
                              "fresh_full_size_initialization": True,
                              "schedule": full_schedule},
        "forecast_hours": {
            "one_proxy": one_proxy_hours, "all_proxies": all_proxy_hours,
            "full_confirmation": full_hours, "reserve": reserve_hours,
            "total": total_hours, "headroom": cap_hours - total_hours,
            "fits_arithmetic_cap": total_hours <= cap_hours,
            "minimum_full_tps_at_other_inputs": (
                full_schedule["total_tokens"] / full_time_available / 3600
                if full_time_available > 0 else None),
        },
        "limitations": [
            "No generated trainer config, manifest, runtime or native gate validation.",
            "No live allocation ledger, watchdog, cloud action or training launch.",
            "Rates must include ordinary cadence overhead or charge it separately without double-counting.",
            "The default proxy rate is user-reported; the default full-size rate is assumed.",
            "All billed GPU quantities and setup, probes, evaluation, retries and idle time count.",
            "Valid-target counts must be audited against nominal sequence-slot batch arithmetic.",
        ],
    }


def save_plan(plan: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise ValueError("save generated forecasts outside this repository")
    payload = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-tps", type=float, default=200_000)
    parser.add_argument("--full-tps", type=float, default=40_000)
    parser.add_argument("--reserve-hours", type=float, default=8)
    parser.add_argument("--cap-hours", type=float, default=MAX_ALLOCATED_HOURS)
    parser.add_argument("--proxy-microbatch", type=int, default=4)
    parser.add_argument("--include-optional-batch", action="store_true",
                        help="Forecast the deferred sixth proxy; does not authorize it")
    parser.add_argument("--output", type=Path, help="Fresh external forecast path")
    args = parser.parse_args()
    try:
        result = make_plan(proxy_tps=args.proxy_tps, full_tps=args.full_tps,
                           reserve_hours=args.reserve_hours, cap_hours=args.cap_hours,
                           include_optional_batch=args.include_optional_batch,
                           proxy_microbatch=args.proxy_microbatch)
        if args.output:
            save_plan(result, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["forecast_hours"]["fits_arithmetic_cap"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
