"""Compare matched evaluation records; no quality threshold is invented."""
import argparse
import json
import math
from pathlib import Path

from production.artifacts import external_path
from terminus.integrity import file_sha256, write_json_atomic
from terminus.task_eval import bootstrap_ci


def records(report):
    path = Path(report["records_path"])
    if file_sha256(path) != report["records_sha256"]:
        raise ValueError("evaluation record digest mismatch")
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["task"], row["item_id"], row["language"], row["role"])
        if key in result or not math.isfinite(row["loss"]):
            raise ValueError("invalid or duplicate evaluation record")
        result[key] = row
    if not result:
        raise ValueError("evaluation has no records")
    return result


def compare(baseline, candidate, *, max_loss_increase=None):
    for key in ("metric", "evaluator_sha256", "tokenizer_sha256", "task_files", "max_items_per_task", "include_final"):
        if key not in baseline or baseline[key] != candidate.get(key):
            raise ValueError(f"evaluations are not comparable: {key}")
    before, after = records(baseline), records(candidate)
    if set(before) != set(after):
        raise ValueError("evaluation item coverage differs")
    tasks, languages = {}, {}
    for key, row in before.items():
        if row["output_tokens"] != after[key]["output_tokens"]:
            raise ValueError("scored token counts differ")
        delta = after[key]["loss"] - row["loss"]
        tasks.setdefault(key[0], []).append(delta)
        languages.setdefault(key[2], []).append(delta)
    per_task = {task: bootstrap_ci(values) for task, values in sorted(tasks.items())}
    per_language = {lang: bootstrap_ci(values) for lang, values in sorted(languages.items())}
    if max_loss_increase is not None and (not math.isfinite(max_loss_increase) or max_loss_increase < 0):
        raise ValueError("loss tolerance must be finite and nonnegative")
    result = {"schema": 1, "metric": "paired_reference_nll_increase_nats_per_token",
              "items": len(before), "per_task": per_task, "per_language": per_language,
              "max_loss_increase": max_loss_increase, "status": "comparison_only"}
    if max_loss_increase is not None:
        # A pooled average cannot hide a language/task regression.
        result["status"] = ("passed" if all(value["high"] <= max_loss_increase
                             for value in [*per_task.values(), *per_language.values()]) else "failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max-loss-increase", type=float)
    args = parser.parse_args()
    output = external_path(args.output, fresh=True)
    report = compare(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text()),
                     max_loss_increase=args.max_loss_increase)
    write_json_atomic(output, report)
    print(json.dumps(report))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
