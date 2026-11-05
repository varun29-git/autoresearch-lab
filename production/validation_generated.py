"""Score saved completions against frozen math, QA and code task records.

No model runs here. Code execution requires an explicit switch and should use
the dedicated isolated evaluation worker described in the readiness contract.
"""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import re
import unicodedata

from production.artifacts import external_path
from terminus.integrity import file_sha256, write_json_atomic


def normalize_answer(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(ch for ch in value if not unicodedata.category(ch).startswith("P"))
    return " ".join(value.split())


def qa_scores(completion, answers):
    predicted = normalize_answer(completion).split()
    scores = []
    for answer in answers:
        target = normalize_answer(answer).split()
        overlap = sum((Counter(predicted) & Counter(target)).values())
        f1 = 2 * overlap / (len(predicted) + len(target)) if predicted and target else float(predicted == target)
        scores.append((float(predicted == target), f1))
    if not scores:
        raise ValueError("QA item has no reference answer")
    return {"exact_match": max(x[0] for x in scores), "token_f1": max(x[1] for x in scores)}


def numeric_answer(value):
    text = unicodedata.normalize("NFKC", value).replace(",", "")
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    number = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    matches = re.findall(r"[-+]?" + number + r"(?:/" + number + r")?", text)
    if not matches:
        return None
    try:
        pieces = matches[-1].split("/")
        result = Decimal(pieces[0])
        if len(pieces) == 2:
            denominator = Decimal(pieces[1])
            if not denominator:
                return None
            result /= denominator
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def code_completion(value):
    # Preserve indentation of plain continuations. Markdown extraction is
    # permitted only for an actual fenced block, never a guessed function.
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", value, re.DOTALL)
    return fence.group(1) if fence else value


def score_rows(items, generations, *, execute_code=False):
    if not items:
        raise ValueError("frozen evaluation has no items")
    ids = [row["id"] for row in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate frozen task ID")
    generated = {}
    for row in generations:
        if row["id"] in generated or not isinstance(row.get("completion"), str):
            raise ValueError("duplicate or invalid generated completion")
        generated[row["id"]] = row["completion"]
    if set(generated) != set(ids):
        raise ValueError("generation coverage differs from frozen task items")
    results, groups = [], {}
    for row in items:
        completion = generated[row["id"]]
        record = {"id": row["id"], "task": row["task"], "language": row["language"], "role": row["role"]}
        if row["kind"] == "math":
            answer = numeric_answer(completion)
            expected = [numeric_answer(x) for x in row["answers"]]
            if any(x is None for x in expected):
                raise ValueError("invalid frozen math answer")
            record.update(exact_match=float(answer is not None and answer in expected),
                          parsed_answer=str(answer) if answer is not None else None)
        elif row["kind"] == "qa":
            record.update(qa_scores(completion, row["answers"]))
        elif row["kind"] == "code":
            if not execute_code:
                raise ValueError("code scoring requires explicit execution on the evaluation worker")
            from terminus.codeexec import score_code_item
            result = score_code_item(row["input"] + code_completion(completion), row["test"])
            record.update(pass_at_1=float(result["passed"]), execution_status=result["status"])
        else:
            raise ValueError("unknown frozen task kind")
        results.append(record)
        key = row["task"] + "/" + row["language"]
        groups.setdefault(key, []).append(record)
    summary = {}
    for name, records in groups.items():
        metrics = {k for r in records for k in ("exact_match", "token_f1", "pass_at_1") if k in r}
        summary[name] = {"items": len(records), **{
            k: sum(r[k] for r in records) / len(records) for k in metrics}}
    return {"summary": summary, "items": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-manifest", "generations", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--role", choices=("development", "final"), default="development")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--execute-code", action="store_true")
    args = parser.parse_args()
    if args.role == "final" and not args.include_final:
        parser.error("final role needs --include-final")
    manifest_path = external_path(args.task_manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "terminus_generated_evaluation_v1":
        raise ValueError("unsupported generated-task manifest")
    items = []
    for entry in manifest["tasks"].values():
        if entry["role"] != args.role:
            continue
        path = (manifest_path.parent / entry["file"]).resolve()
        if manifest_path.parent not in path.parents or file_sha256(path) != entry["sha256"]:
            raise ValueError("task file identity mismatch")
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if len(rows) != entry["rows"] or any(r["role"] != args.role for r in rows):
            raise ValueError("task file role or length mismatch")
        items.extend(rows)
    generated_path = external_path(args.generations)
    generations = [json.loads(x) for x in generated_path.read_text().splitlines()]
    result = score_rows(items, generations, execute_code=args.execute_code)
    result.update(task_manifest_sha256=file_sha256(manifest_path),
                  generations_sha256=file_sha256(generated_path), evaluator_sha256=file_sha256(__file__),
                  model_execution_provenance_verified=False)
    output = external_path(args.output, fresh=True)
    write_json_atomic(output, result)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
