"""Complete-trial comparison, margins-based selection and the single transfer.

Selection uses only complete healthy trials, matched frozen held-out
identities and the margins frozen before trials. A tiny single-seed advantage
is reported as a tie with an explicitly provisional choice; an incomplete
matrix is reported as incomplete. Nothing here reruns training or invents
significance: document/bootstrap uncertainty belongs to the evaluation
records and addresses evaluation noise only, not seed variance or selection
bias.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from production.artifacts import external_path

from . import campaign


class SelectionError(RuntimeError):
    """The comparison cannot support a selection claim."""


def _read_events(trial_dir) -> list:
    path = Path(trial_dir) / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def aggregate_nll(languages: dict) -> float:
    """Token-weighted token-mean NLL over per-domain validation results."""
    loss_sum = sum(result["loss"] * result["tokens"] for result in languages.values())
    tokens = sum(result["tokens"] for result in languages.values())
    if tokens <= 0:
        raise SelectionError("validation results contain no tokens")
    return loss_sum / tokens


def load_trial_summary(trial_dir, *, boundary_updates=campaign.PROXY_BOUNDARY_UPDATES) -> dict:
    """Completion status and boundary held-out NLL for one trial directory."""
    events = _read_events(trial_dir)
    terminal = next((event for event in reversed(events)
                     if event.get("event") in ("complete", "stopped")), None)
    incidents_path = Path(trial_dir) / "incidents.jsonl"
    incidents = []
    if incidents_path.is_file():
        incidents = [json.loads(line) for line in incidents_path.read_text().splitlines() if line.strip()]
    validations = {}
    for event in events:
        if event.get("event") == "validation" and event.get("updates") in boundary_updates:
            languages = event.get("languages") or {}
            validations[event["updates"]] = {
                "aggregate": aggregate_nll(languages),
                "per_domain": {name: result["loss"] for name, result in languages.items()},
                "tokens": sum(result["tokens"] for result in languages.values()),
            }
    completed = (terminal is not None and terminal.get("event") == "complete"
                 and terminal.get("updates") == max(boundary_updates))
    return {
        "completed": completed,
        "failed": bool(incidents),
        "incidents": [{"event": item.get("event"), "error": item.get("error")}
                      for item in incidents],
        "terminal": terminal,
        "boundary_validation": validations,
        "missing_boundaries": [u for u in boundary_updates if u not in validations],
    }


def compare_trials(summaries: dict, margins: dict,
                   *, boundary_updates=campaign.PROXY_BOUNDARY_UPDATES) -> dict:
    """Margins-based selection over complete healthy trials, or an honest status."""
    final_update = max(boundary_updates)
    pre_decay_update = sorted(boundary_updates)[-2]
    eligible, excluded = {}, {}
    for trial_id, summary in summaries.items():
        if summary["failed"]:
            excluded[trial_id] = "failed attempt; see incidents"
        elif not summary["completed"]:
            excluded[trial_id] = "incomplete trial"
        elif summary["missing_boundaries"]:
            excluded[trial_id] = f"missing boundary validations {summary['missing_boundaries']}"
        else:
            eligible[trial_id] = summary
    if len(eligible) < len(summaries):
        return {"schema": 1, "status": "incomplete", "eligible": sorted(eligible),
                "excluded": excluded,
                "note": "no winner is claimed from an incomplete comparison"}
    final_nll = {trial: summary["boundary_validation"][final_update]["aggregate"]
                 for trial, summary in eligible.items()}
    pre_decay_nll = {trial: summary["boundary_validation"][pre_decay_update]["aggregate"]
                     for trial, summary in eligible.items()}
    superiority = margins["aggregate_superiority_nll"]
    domain_margin = margins["domain_noninferiority_nll"]
    ranked = sorted(final_nll, key=final_nll.get)
    best, second = ranked[0], ranked[1]
    pairwise = {f"{best}_minus_{other}": final_nll[best] - final_nll[other]
                for other in ranked[1:]}
    domains = sorted({domain for summary in eligible.values()
                      for domain in summary["boundary_validation"][final_update]["per_domain"]})
    domain_excess = {}
    for domain in domains:
        values = {trial: summary["boundary_validation"][final_update]["per_domain"].get(domain)
                  for trial, summary in eligible.items()}
        others = [value for trial, value in values.items() if trial != best and value is not None]
        if values[best] is not None and others:
            domain_excess[domain] = values[best] - min(others)
    superiority_gap = final_nll[second] - final_nll[best]
    selected = (superiority_gap >= superiority
                and all(excess <= domain_margin for excess in domain_excess.values()))
    report = {
        "schema": 1,
        "final_update": final_update,
        "pre_decay_update": pre_decay_update,
        "final_aggregate_nll": final_nll,
        "pre_decay_aggregate_nll": pre_decay_nll,
        "pairwise_final_differences": pairwise,
        "domain_excess_of_best": domain_excess,
        "superiority_gap": superiority_gap,
        "margins": dict(margins),
    }
    if selected:
        report.update({"status": "selected", "selection": best, "provisional": True,
                       "note": "single-seed proxy selection: a provisional transfer choice, "
                               "not a trillion-token optimum"})
    else:
        report.update({"status": "tie", "selection": best, "provisional": True,
                       "note": "candidates are not separated under the frozen margins; the "
                               "provisional choice is the best aggregate and is not a proven ranking"})
    return report


def transfer_confirmation(selection: dict) -> dict:
    """The single width-transfer record for the selected recipe."""
    if selection.get("status") not in ("selected", "tie") or not selection.get("selection"):
        raise SelectionError("transfer requires a completed comparison")
    trial = selection["selection"]
    if trial not in campaign.TRIAL_LRS:
        raise SelectionError(f"unknown selected trial: {trial!r}")
    return {
        "schema": 1,
        "selected_trial": trial,
        "selection_status": selection["status"],
        "proxy_muon_lr": campaign.TRIAL_LRS[trial]["muon_lr"],
        "proxy_adamw_lr": campaign.TRIAL_LRS[trial]["adamw_lr"],
        "transfer_multiplier": f"{campaign.PROXY_WIDTH}/{campaign.FULL_WIDTH}",
        "transferred_muon_lr": campaign.TRIAL_LRS[trial]["muon_lr"],
        "transferred_adamw_lr": campaign.transferred_adamw_lr(trial),
        "unchanged": {"norm_lr": campaign.FIXED_OPTIMIZER["norm_lr"],
                      "qk_norm_lr": campaign.FIXED_OPTIMIZER["qk_norm_lr"],
                      "weight_decay": campaign.FIXED_OPTIMIZER["weight_decay"],
                      "embedding_weight_decay": campaign.FIXED_OPTIMIZER["embedding_weight_decay"],
                      "momentum": campaign.FIXED_OPTIMIZER["momentum"],
                      "betas": campaign.FIXED_OPTIMIZER["betas"],
                      "eps": campaign.FIXED_OPTIMIZER["eps"]},
        "note": "transfer applied exactly once; hyperparameters only, never weights",
    }


def main() -> int:
    from terminus.integrity import write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare", help="Compare complete trials and select or tie")
    compare.add_argument("--qual-root", required=True)
    compare.add_argument("--margins", required=True)
    compare.add_argument("--output", required=True)
    transfer = sub.add_parser("transfer", help="Emit the single transfer record")
    transfer.add_argument("--selection", required=True)
    transfer.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "compare":
        root = Path(external_path(args.qual_root))
        margins = json.loads(Path(args.margins).read_text())
        summaries = {}
        for trial_id in campaign.TRIAL_ORDER:
            trial_dir = root / "trials" / trial_id
            summaries[trial_id] = load_trial_summary(trial_dir)
        report = compare_trials(summaries, margins)
        report["trials"] = {trial: {"completed": s["completed"], "failed": s["failed"],
                                    "boundary_validation": s["boundary_validation"]}
                            for trial, s in summaries.items()}
        write_json_atomic(external_path(args.output, fresh=True), report)
        print(json.dumps({"status": report["status"],
                          "selection": report.get("selection")}, sort_keys=True))
        return 0 if report["status"] != "incomplete" else 2
    selection = json.loads(Path(args.selection).read_text())
    record = transfer_confirmation(selection)
    write_json_atomic(external_path(args.output, fresh=True), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
