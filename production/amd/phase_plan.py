"""Allocate dense/QAT tokens and time; emits a budget plan, never a run config.

The recommended starting allocation is 60% dense / 40% ternary by tokens.
Both phase rates are required inputs and are not certified by this calculator.
See QAT_ALLOCATION.md. AMD ternary execution remains disabled pending its port.
"""
import argparse
from fractions import Fraction
import json
import math

from production.artifacts import external_path
from terminus.integrity import write_json_atomic


RECOMMENDED_QAT_PERCENT = 40
MIXED_RUN_RESERVE_GPU_HOURS = 480


def plan(*, dense_rate, qat_rate, qat_percent=RECOMMENDED_QAT_PERCENT,
         gpu_hours=5760, reserve_gpu_hours=MIXED_RUN_RESERVE_GPU_HOURS,
         duty_cycle=.9, gpus=8, global_batch_tokens=2_097_152):
    for name, value in {"dense_rate": dense_rate, "qat_rate": qat_rate,
                        "gpu_hours": gpu_hours, "duty_cycle": duty_cycle}.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (isinstance(reserve_gpu_hours, bool) or not isinstance(reserve_gpu_hours, (float, int))
            or not math.isfinite(reserve_gpu_hours) or not 0 <= reserve_gpu_hours < gpu_hours):
        raise ValueError("reserve_gpu_hours must fit inside the allocation")
    if duty_cycle > 1 or type(gpus) is not int or gpus < 1:
        raise ValueError("invalid duty_cycle or GPU count")
    if type(qat_percent) is not int or not 0 < qat_percent < 100:
        raise ValueError("qat_percent must be an integer from 1 to 99")
    if type(global_batch_tokens) is not int or global_batch_tokens <= 0 or global_batch_tokens % 4096:
        raise ValueError("global_batch_tokens must be a positive multiple of 4096")

    fraction = Fraction(qat_percent, 100)
    qat_units = fraction.numerator
    dense_units = fraction.denominator - qat_units
    # One group contains complete global updates in the exact selected ratio.
    group_dense = dense_units * global_batch_tokens
    group_qat = qat_units * global_batch_tokens
    group_productive_gpu_seconds = group_dense / dense_rate + group_qat / qat_rate
    available_seconds = (gpu_hours - reserve_gpu_hours) * 3600 * duty_cycle
    if not math.isfinite(available_seconds) or not math.isfinite(group_productive_gpu_seconds):
        raise ValueError("nonfinite phase accounting")
    capacity = available_seconds / group_productive_gpu_seconds
    if not math.isfinite(capacity):
        raise ValueError("nonfinite phase capacity")
    groups = math.floor(capacity)
    if groups < 1:
        raise ValueError("allocation cannot fit one complete update group at this split")
    dense_tokens, qat_tokens = groups * group_dense, groups * group_qat
    try:
        finite_tokens = math.isfinite(float(dense_tokens + qat_tokens))
    except OverflowError:
        finite_tokens = False
    if not finite_tokens:
        raise ValueError("nonfinite token capacity")

    def phase(tokens, rate):
        productive_hours = tokens / rate / 3600
        allocated_hours = productive_hours / duty_cycle
        return {"tokens": tokens, "updates": tokens // global_batch_tokens,
                "input_tokens_per_second_per_gpu": rate,
                "productive_gpu_hours": productive_hours, "allocated_gpu_hours": allocated_hours,
                "node_days_including_assumed_overhead": allocated_hours / gpus / 24}

    dense, qat = phase(dense_tokens, dense_rate), phase(qat_tokens, qat_rate)
    used = reserve_gpu_hours + dense["allocated_gpu_hours"] + qat["allocated_gpu_hours"]
    total = dense_tokens + qat_tokens
    return {"schema": 1, "status": "arithmetic_plan_not_a_training_configuration",
            "rate_evidence": "caller_inputs_unverified", "amd_qat_ready": False,
            "allocation_gpu_hours": gpu_hours, "gpus": gpus, "reserve_gpu_hours": reserve_gpu_hours,
            "reserve_node_days": reserve_gpu_hours / gpus / 24, "duty_cycle": duty_cycle,
            "qat_token_percent": qat_percent, "global_batch_tokens": global_batch_tokens,
            "total_tokens": total, "dense": dense, "qat": qat,
            "effective_productive_tokens_per_second_per_gpu":
                total / (dense_tokens / dense_rate + qat_tokens / qat_rate),
            "qat_fraction_of_production_time": qat["allocated_gpu_hours"] /
                (dense["allocated_gpu_hours"] + qat["allocated_gpu_hours"]),
            "total_allocated_gpu_hours_used": used,
            "rounding_headroom_gpu_hours": max(0., gpu_hours - used),
            "total_node_days": used / gpus / 24,
            "transition_at_global_update": dense["updates"],
            "note": "Reserve includes qualification of both phases. Duty cycle is applied once; rates must already include real DDP and optimizer work. No native QAT capability or quality is certified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-rate", type=float, required=True)
    parser.add_argument("--qat-rate", type=float, required=True)
    parser.add_argument("--qat-percent", type=int, default=RECOMMENDED_QAT_PERCENT)
    parser.add_argument("--gpu-hours", type=float, default=5760)
    parser.add_argument("--reserve-gpu-hours", type=float, default=MIXED_RUN_RESERVE_GPU_HOURS)
    parser.add_argument("--duty-cycle", type=float, default=.9)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--global-batch-tokens", type=int, default=2_097_152)
    parser.add_argument("--output", help="optional fresh JSON path outside Git")
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = plan(**args)
    if output:
        write_json_atomic(external_path(output, fresh=True), result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
