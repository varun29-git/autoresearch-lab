"""5,760 GPU-hour accounting; input rates must come from measured whole updates."""
import argparse
import json
import math


def forecast(*, dense_rate, target_tokens=1_500_000_000_000, gpu_hours=5760,
             gpus=8, reserve_gpu_hours=0, duty_cycle=1., qat_tokens=0, qat_rate=None):
    for name, value in dict(dense_rate=dense_rate, target_tokens=target_tokens, gpu_hours=gpu_hours,
                            gpus=gpus, duty_cycle=duty_cycle).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if type(gpus) is not int or duty_cycle > 1:
        raise ValueError("invalid GPU count or duty cycle")
    if (not math.isfinite(reserve_gpu_hours) or not 0 <= reserve_gpu_hours < gpu_hours
            or not math.isfinite(qat_tokens) or not 0 <= qat_tokens <= target_tokens):
        raise ValueError("reserve and QAT tokens must fit the budget")
    if qat_tokens and (qat_rate is None or not math.isfinite(qat_rate) or qat_rate <= 0):
        raise ValueError("nonzero QAT allocation requires a measured QAT rate")
    seconds = (gpu_hours - reserve_gpu_hours) * 3600 * duty_cycle
    qat_seconds = qat_tokens / qat_rate if qat_tokens else 0.
    available_dense = max(0., seconds - qat_seconds)
    dense_tokens = target_tokens - qat_tokens
    required_seconds = dense_tokens / dense_rate + qat_seconds
    needed_hours = reserve_gpu_hours + required_seconds / 3600 / duty_cycle
    return {"schema": 1, "status": "arithmetic_projection_only", "gpu_hours": gpu_hours,
            "gpus": gpus, "allocation_wall_days": gpu_hours / gpus / 24,
            "reserve_gpu_hours": reserve_gpu_hours, "duty_cycle": duty_cycle,
            "target_tokens": target_tokens, "qat_tokens": qat_tokens, "qat_rate": qat_rate,
            "dense_rate_per_gpu": dense_rate,
            "projected_dense_tokens_after_qat": available_dense * dense_rate,
            "required_dense_rate_per_gpu": dense_tokens / available_dense if available_dense else None,
            "gross_required_average_rate_per_gpu": target_tokens / (gpu_hours * 3600),
            "gpu_hours_required": needed_hours, "wall_days_required": needed_hours / gpus / 24,
            "fits_assumptions": required_seconds <= seconds,
            "note": "Use an actual-cluster per-GPU rate. Duty cycle excludes tests/setup already charged to reserve; do not double-count communication or QAT."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense-rate", type=float, required=True)
    p.add_argument("--target-tokens", type=int, default=1_500_000_000_000)
    p.add_argument("--gpu-hours", type=float, default=5760)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--reserve-gpu-hours", type=float, default=0.)
    p.add_argument("--duty-cycle", type=float, default=1.)
    p.add_argument("--qat-tokens", type=int, default=0)
    p.add_argument("--qat-rate", type=float)
    print(json.dumps(forecast(**vars(p.parse_args())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
