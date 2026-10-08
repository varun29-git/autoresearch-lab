import argparse
import json
from pathlib import Path

from .physics import HARDWARE, budget_targets, compute_floor, projection
from .protocol import (analyze_screen, append_receipt, make_lock, read_json,
                       validate_candidate_config, verify_anchor)


def main():
    parser = argparse.ArgumentParser(description="Stack-004 speed research: offline tools; no GPU/cloud launcher")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="compute targets and optimistic mixed-precision bounds")
    plan.add_argument("--hardware", choices=HARDWARE, default="h100_sxm")
    plan.add_argument("--unavailable-fraction", type=float, default=0.1)
    plan.add_argument("--dense-tps", type=float)
    plan.add_argument("--qat-tps", type=float)
    plan.add_argument("--qat-fraction", type=float)
    plan.add_argument("--scaling-efficiency", type=float)
    anchor = sub.add_parser("verify-anchor", help="verify winning source from Git, optionally the working tree")
    anchor.add_argument("--repo", default=".")
    anchor.add_argument("--checkout", action="store_true")
    config = sub.add_parser("check-config", help="check a fully materialized candidate model config")
    config.add_argument("--config", required=True)
    config.add_argument("--phase", required=True, choices=["dense", "qat"])
    lock = sub.add_parser("lock", help="freeze a completed campaign template outside Git")
    lock.add_argument("--campaign", required=True)
    lock.add_argument("--output", required=True)
    analyze = sub.add_parser("analyze", help="analyze paired native full-update screen records")
    analyze.add_argument("--lock", required=True)
    analyze.add_argument("--evidence", required=True)
    append = sub.add_parser("append", help="append an external trial receipt without rewriting history")
    append.add_argument("--ledger", required=True)
    append.add_argument("--receipt", required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = {"targets_before_overhead": budget_targets(),
                      "targets_after_overhead": budget_targets(unavailable_fraction=args.unavailable_fraction),
                      "optimistic_compute_bound": compute_floor(hardware=args.hardware),
                      "research_gpus": 1, "production_gpus": 32,
                      "research_budget": "separate; human must set a cap before launching"}
            rates = [args.dense_tps, args.qat_tps, args.qat_fraction, args.scaling_efficiency]
            if any(v is not None for v in rates):
                if any(v is None for v in rates):
                    raise ValueError("projection requires dense-tps, qat-tps, qat-fraction and scaling-efficiency together")
                result["projection"] = projection(dense_tps=args.dense_tps, qat_tps=args.qat_tps,
                    qat_fraction=args.qat_fraction, scaling_efficiency=args.scaling_efficiency,
                    unavailable_fraction=args.unavailable_fraction)
        elif args.command == "verify-anchor":
            result = verify_anchor(args.repo, checkout=args.checkout)
        elif args.command == "check-config":
            result = validate_candidate_config(read_json(args.config), phase=args.phase)
        elif args.command == "lock":
            result = make_lock(read_json(args.campaign))
            with Path(args.output).open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.write("\n")
        elif args.command == "analyze":
            result = analyze_screen(read_json(args.evidence), read_json(args.lock))
        else:
            result = append_receipt(args.ledger, read_json(args.receipt))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
