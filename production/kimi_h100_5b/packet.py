"""Generate the external launch packet (QUAL_ROOT) for the human handoff.

Creates the EVIDENCE.md layout with the frozen campaign contract, forecast,
source/tokenizer identity, frozen selection margins, honest gate receipts
(passed only with an actual result; otherwise unrun or
blocked_by_missing_input) and the exact ordered launch commands. Candidate
configs and horizon manifests are generated separately once a verified
source token manifest exists; this packet names precisely what is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from production.artifacts import external_path
from terminus.integrity import file_sha256, write_json_atomic

from . import campaign, plan


def _git_identity() -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(campaign.REPOSITORY), *args],
                              capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty_files": [line for line in git("status", "--porcelain").splitlines() if line]}


def _gate(name: str, status: str, detail: str, remaining: list) -> dict:
    return {"schema": 1, "gate": name, "status": status, "detail": detail,
            "remaining": remaining}


def _launch_text(qual_root: Path) -> str:
    return f"""# Human launch runbook — bounded H100 proxy qualification

This packet grants no launch by itself. The human provisions the Azure H100
VM, starts and stops all paid work, and records the launch below.

QUAL_ROOT = {qual_root}

## Environment inputs to fill before any launch (never commit values)

- Azure VM SKU, billed GPU quantity per clock hour, usable HBM
- DATA_ROOT (prepared shards) and durable backup destination (S3 URI or path)
- DG_JIT_NVCC_COMPILER and FA3_CUDA_HOME (required by runtime capture)
- Ambient cloud credentials stay in the VM environment, never in files

## Ordered native gates (RUNBOOK section 4)

1. VM/hardware identity; record the actual billing unit:
   `nvidia-smi -q > $QUAL_ROOT/identity/nvidia_smi.txt`
2. Runtime capture:  `python -m production.runtime --output $QUAL_ROOT/identity/runtime.json`
   Verify later with: `python -m production.runtime --verify $QUAL_ROOT/identity/runtime.json`
3. Open the allocation ledger (starts the 50 h clock; never reset on resume):
   `python -m production.kimi_h100_5b.budget --ledger $QUAL_ROOT/budget.jsonl open`
4. Data: fetch/tokenize/verify shards per DATA_SOURCE_MAP.md; build the source
   manifest; create the staging receipt:
   `python -m production.amd.staging --config <cfg> --manifest <src> --data-root $DATA_ROOT --output $QUAL_ROOT/identity/staging.json`
   (staging is hardware-agnostic CPU verification reused for H100.)
5. Horizon views and candidate configs:
   `python -m production.kimi_h100_5b.horizons view --source-manifest <src> --total-tokens 4999610368 --microbatch 4 --output $QUAL_ROOT/manifests/proxy_5b.json`
   `for T in H R0 R1 R2 R3; do python -m production.kimi_h100_5b.builder --trial $T --microbatch 4 --manifest $QUAL_ROOT/manifests/proxy_5b.json --output $QUAL_ROOT/configs/$T.json; done`
6. Native preflight per geometry (proxy and full-size):
   `python -m terminus.preflight --config $QUAL_ROOT/configs/H.json --output $QUAL_ROOT/gates/preflight_proxy.json`
7. Bounded fit/rate probes (probe runs never count as candidate progress):
   `python -m production.kimi_h100_5b.runner --config $QUAL_ROOT/configs/H.json --manifest $QUAL_ROOT/manifests/proxy_5b.json --data-root $DATA_ROOT --output-dir $QUAL_ROOT/probes/fit_H --trial H --probe fit_H --stop-after-updates 20 --runtime-report $QUAL_ROOT/identity/runtime.json --staging-receipt $QUAL_ROOT/identity/staging.json --preflight-report $QUAL_ROOT/gates/preflight_proxy.json --source-receipt $QUAL_ROOT/identity/source.json --ledger $QUAL_ROOT/budget.jsonl --stage-forecast-hours 1.0`
8. Recovery drill with the chosen destination; compare restored state.
9. Reforecast with measured cadence-inclusive rates; update forecast.json.
10. Trials in order H, R0, R1, R2, R3 (example for H):
   `python -m production.kimi_h100_5b.runner --config $QUAL_ROOT/configs/H.json --manifest $QUAL_ROOT/manifests/proxy_5b.json --data-root $DATA_ROOT --output-dir $QUAL_ROOT/trials/H --trial H --runtime-report $QUAL_ROOT/identity/runtime.json --staging-receipt $QUAL_ROOT/identity/staging.json --preflight-report $QUAL_ROOT/gates/preflight_proxy.json --source-receipt $QUAL_ROOT/identity/source.json --ledger $QUAL_ROOT/budget.jsonl --stage-forecast-hours <measured> --backup-destination <dest> --reference-probes --replay-max-weight-delta <frozen> --replay-max-moment-delta <frozen>`
11. Selection: `python -m production.kimi_h100_5b.selection compare --qual-root $QUAL_ROOT --margins $QUAL_ROOT/evaluation/margins.json --output $QUAL_ROOT/reports/selection.json`
12. Single transfer + one confirmation:
   `python -m production.kimi_h100_5b.selection transfer --selection $QUAL_ROOT/reports/selection.json --output $QUAL_ROOT/reports/transfer.json`
   `python -m production.kimi_h100_5b.builder --confirm-from <winner> --microbatch 1 --manifest $QUAL_ROOT/manifests/confirmation_1b.json --output $QUAL_ROOT/confirmation/config.json`
   then run the confirmation like step 10 with --confirmation-from <winner>.
13. Handback per EVIDENCE.md; close the ledger only when the VM is deallocated:
   `python -m production.kimi_h100_5b.budget --ledger $QUAL_ROOT/budget.jsonl close`

## Stop / failure / resume

- Stop gracefully: send SIGINT/SIGTERM to the runner; it checkpoints at the
  next cadence boundary and exits. The ledger keeps counting while the VM bills.
- Hard failure: the attempt stays failed; the last valid durable checkpoint is
  under trials/<id>/checkpoints with upload markers; repairs need a new run identity.
- Resume: add `--resume $QUAL_ROOT/trials/<id>/checkpoints/step_XXXXXXXX.pt`;
  the contract, world size, data and runtime receipts must verify unchanged,
  and the ledger clock is never reset.
- Process exit does not stop billing; deallocate the VM to close the ledger.
"""


def create_packet(qual_root, *, tokenizer_path=None, proxy_tps: float = 200_000,
                  full_tps: float = 40_000, reserve_hours: float = 8) -> dict:
    root = external_path(qual_root)
    if (root / "campaign.json").exists():
        raise FileExistsError("campaign packet already exists; do not overwrite a frozen packet")
    for name in ("identity", "manifests", "configs", "evaluation", "gates", "probes",
                 "trials", "confirmation", "reports", "launch"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for trial_id in campaign.TRIAL_ORDER:
        (root / "trials" / trial_id).mkdir(exist_ok=True)

    forecast = plan.make_plan(proxy_tps=proxy_tps, full_tps=full_tps,
                              reserve_hours=reserve_hours)
    git = _git_identity()
    campaign_doc = {
        "schema": 1,
        "status": "prepared_not_launched",
        "protocol": "production/kimi_h100_5b/RUNBOOK.md",
        "trials_in_order": list(campaign.TRIAL_ORDER),
        "trial_lrs": campaign.TRIAL_LRS,
        "fixed_optimizer": campaign.FIXED_OPTIMIZER,
        "proxy_schedule": campaign.PROXY_SCHEDULE,
        "confirmation_schedule": campaign.CONFIRMATION_SCHEDULE,
        "global_batch_tokens": campaign.GLOBAL_BATCH_TOKENS,
        "world_size": campaign.WORLD_SIZE,
        "cap_hours": campaign.MAX_ALLOCATED_HOURS,
        "emergency_reserve_hours": campaign.EMERGENCY_RESERVE_HOURS,
        "selection_margins": campaign.SELECTION_MARGINS,
        "optional_batch_trial": "deferred_by_default",
        "git": git,
        "human_launch_record": None,
    }
    write_json_atomic(root / "campaign.json", campaign_doc)
    write_json_atomic(root / "forecast.json", forecast)
    write_json_atomic(root / "identity" / "source.json",
                      {"schema": 1, "source": campaign.qualification_source_hashes(),
                       "git": git})
    write_json_atomic(root / "evaluation" / "margins.json", campaign.SELECTION_MARGINS)

    if tokenizer_path is not None:
        tokenizer = Path(tokenizer_path).resolve()
        if not tokenizer.is_file():
            raise FileNotFoundError(f"tokenizer file not found: {tokenizer}")
        write_json_atomic(root / "identity" / "tokenizer.json", {
            "schema": 1, "path": str(tokenizer), "sha256": file_sha256(tokenizer),
            "note": "verified against the freeze record before manifest creation"})

    gates = [
        _gate("offline_contract", "blocked_by_missing_input",
              "builder/validator implemented and CPU-tested; needs the source token manifest "
              "to generate horizon views and candidate configs",
              ["verified source token manifest with exact 4,999,610,368-token usable horizon"]),
        _gate("data_evaluation", "blocked_by_missing_input",
              "no training shards or evaluation records are present in this environment; "
              "tokenizer identity recorded from the available frozen file",
              ["fetched/verified shards per DATA_SOURCE_MAP.md", "staging receipt",
               "frozen held-out identities and the reserved final holdout"]),
        _gate("runtime_native", "blocked_by_missing_input",
              "Azure H100 is currently deallocated; runtime/preflight receipts need the live VM",
              ["VM SKU/HBM/billing identity", "runtime capture", "native preflight per geometry",
               "loss/masking/gradient/optimizer reference checks", "FP8 cache verification"]),
        _gate("execution_fit", "blocked_by_missing_input",
              "fit and cadence-inclusive rates are unmeasured at both geometries",
              ["bounded full-update fit probes", "peak HBM >= 15% headroom",
               "measured proxy/full-size rates for the reforecast"]),
        _gate("recovery_storage", "blocked_by_missing_input",
              "backup destination unknown; drill needs native checkpoints",
              ["durable destination", "upload/restore drill with subsequent-step comparison",
               "free-space and lag thresholds"]),
        _gate("budget_launch", "blocked_by_missing_input",
              "allocation not open; Azure billing unit unconfirmed",
              ["allocation open record", "measured reforecast fitting 50 h",
               "explicit human launch record"]),
    ]
    for gate in gates:
        write_json_atomic(root / "gates" / f"{gate['gate']}.json", gate)
    (root / "launch" / "LAUNCH.md").write_text(_launch_text(root), encoding="utf-8")
    return {"schema": 1, "qual_root": str(root), "status": "prepared_not_launched",
            "gates": {gate["gate"]: gate["status"] for gate in gates},
            "forecast_fits_arithmetic_cap": forecast["forecast_hours"]["fits_arithmetic_cap"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qual-root", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--proxy-tps", type=float, default=200_000)
    parser.add_argument("--full-tps", type=float, default=40_000)
    parser.add_argument("--reserve-hours", type=float, default=8)
    args = parser.parse_args()
    result = create_packet(args.qual_root, tokenizer_path=args.tokenizer,
                           proxy_tps=args.proxy_tps, full_tps=args.full_tps,
                           reserve_hours=args.reserve_hours)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
