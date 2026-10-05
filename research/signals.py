"""Read-only architecture signals from the protected trainer's existing logs."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def _records(path: Path):
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _maximum(old, value):
    value = _finite(value)
    return max(old, value) if old is not None and value is not None else value if old is None else old


def _minimum(old, value):
    value = _finite(value)
    return min(old, value) if old is not None and value is not None else value if old is None else old


def _nonfinite_fractions(item):
    if isinstance(item, dict):
        if "nonfinite_fraction" in item:
            yield item["nonfinite_fraction"]
        for value in item.values():
            yield from _nonfinite_fractions(value)


def _probe_numbers(item, prefix=""):
    """Preserve each named tensor statistic, excluding sample-size metadata."""
    for name, value in item.items():
        if name in {"values", "key_positions", "query_heads"}:
            continue
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from _probe_numbers(value, path)
        elif value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            yield path, _finite(value)


def _record_metric(metrics: dict, name: str, tokens: int, value: float | None) -> None:
    row = metrics.setdefault(name, {"count": 0, "missing": 0, "first": None,
                                    "last": None, "min": None, "max": None,
                                    "sum": 0.0})
    if value is None:
        row["missing"] += 1
        return
    point = {"tokens_seen": tokens, "value": value}
    if row["first"] is None:
        row["first"] = point
    row["last"] = point
    if row["min"] is None or value < row["min"]["value"]:
        row["min"] = point
    if row["max"] is None or value > row["max"]["value"]:
        row["max"] = point
    row["count"] += 1
    row["sum"] += value


def _metric_summary(metrics: dict) -> dict:
    return {name: {key: value for key, value in row.items() if key != "sum"} |
                  {"mean": row["sum"] / row["count"] if row["count"] else None}
            for name, row in sorted(metrics.items())}


def _phase_summary(train_events: list[dict], through_tokens: int | None) -> dict:
    if not train_events:
        return {}
    horizon = through_tokens or max(item["tokens_seen"] for item in train_events)
    phases = {name: [] for name in ("early", "middle", "late")}
    for item in train_events:
        fraction = item["tokens_seen"] / horizon if horizon else 1.0
        name = "early" if fraction <= 1 / 3 else "middle" if fraction <= 2 / 3 else "late"
        phases[name].append(item)
    result = {}
    for name, items in phases.items():
        if not items:
            result[name] = {"logged_updates": 0}
            continue
        def median(field):
            values = [_finite(item.get(field)) for item in items]
            values = [value for value in values if value is not None]
            return statistics.median(values) if values else None
        clips = [_finite(item.get("grad_clip_coefficient")) for item in items]
        valid_clips = [value for value in clips if value is not None]
        memories = [_finite(item.get("peak_memory_bytes")) for item in items]
        memories = [value for value in memories if value is not None]
        reserved = [_finite(item.get("reserved_memory_bytes")) for item in items]
        reserved = [value for value in reserved if value is not None]
        result[name] = {"logged_updates": len(items),
                        "median_train_loss": median("loss"),
                        "median_tokens_per_second": median("tokens_per_second"),
                        "median_gradient_norm_before_clip": median("grad_norm_before_clip"),
                        "gradient_clip_fraction": (sum(value < 1 - 1e-6 for value in valid_clips) /
                                                   len(valid_clips) if valid_clips else None),
                        "min_gradient_clip_coefficient": min(valid_clips) if valid_clips else None,
                        "median_lr_scale": median("lr_scale"),
                        "median_optimizer_group_lr_min": median("optimizer_group_lr_min"),
                        "median_optimizer_group_lr_max": median("optimizer_group_lr_max"),
                        "median_quantization_strength": median("quantization_strength"),
                        "max_peak_memory_bytes": max(memories) if memories else None,
                        "max_reserved_memory_bytes": max(reserved) if reserved else None}
    return result


def summarize_run(run_dir: Path, *, through_tokens: int | None = None) -> dict:
    """Keep diagnostic samples separate from hard numerical failures."""
    run_dir = Path(run_dir)
    speeds, curve, incident_types, train_events, probe_metrics = [], [], {}, [], {}
    clip_count = train_count = stability_count = 0
    peak_memory = peak_gradient = max_nonfinite = max_residual = None
    max_attention_logit = min_attention_entropy = max_fp8_error = max_update_ratio = None
    for event in _records(run_dir / "events.jsonl"):
        tokens = event.get("tokens_seen", 0)
        if through_tokens is not None and tokens > through_tokens:
            continue
        if event.get("event") == "train":
            train_events.append(event)
            train_count += 1
            coefficient = _finite(event.get("grad_clip_coefficient"))
            clip_count += coefficient is not None and coefficient < 1 - 1e-6
            speed = _finite(event.get("tokens_per_second"))
            if speed is not None and speed > 0:
                speeds.append(speed)
            peak_memory = _maximum(peak_memory, event.get("peak_memory_bytes"))
            peak_gradient = _maximum(peak_gradient, event.get("grad_norm_before_clip"))
        elif event.get("event") == "validation":
            loss = _finite((event.get("languages") or {}).get("en", {}).get("loss"))
            if loss is not None and loss > 0:
                curve.append((tokens, loss))
    for record in _records(run_dir / "stability.jsonl"):
        if through_tokens is not None and record.get("tokens_seen", 0) > through_tokens:
            continue
        stability_count += 1
        probes = record.get("probes") or {}
        for name, value in _probe_numbers(probes):
            _record_metric(probe_metrics, name, record.get("tokens_seen", 0), value)
        for fraction in _nonfinite_fractions(probes):
            max_nonfinite = _maximum(max_nonfinite, fraction)
        for name, probe in probes.items():
            if not isinstance(probe, dict):
                continue
            if name.endswith(".residual"):
                max_residual = _maximum(max_residual, probe.get("rms"))
            elif name.endswith(".attention_sample"):
                max_attention_logit = _maximum(max_attention_logit,
                                               (probe.get("logits") or {}).get("max_abs"))
                min_attention_entropy = _minimum(min_attention_entropy, probe.get("min_entropy"))
            elif name.endswith(".fp8_input"):
                max_fp8_error = _maximum(max_fp8_error, probe.get("relative_rms_error"))
            elif name.endswith(".parameter_sample"):
                max_update_ratio = _maximum(max_update_ratio, probe.get("update_to_weight_rms"))
    for incident in _records(run_dir / "incidents.jsonl"):
        if through_tokens is not None and (incident.get("state") or {}).get("tokens_seen", 0) > through_tokens:
            continue
        name = str(incident.get("event", "unknown"))
        incident_types[name] = incident_types.get(name, 0) + 1
    curve = sorted(dict(curve).items())
    numerical_failure = (incident_types.get("nonfinite_loss", 0) > 0
                         or incident_types.get("gradient_failure", 0) > 0
                         or (max_nonfinite is not None and max_nonfinite > 0))
    return {
        "validation_curve": [{"tokens_seen": tokens, "en_loss": loss} for tokens, loss in curve],
        "best_en_validation_loss": min((loss for _, loss in curve), default=None),
        "last_en_validation_loss": curve[-1][1] if curve else None,
        "late_validation_regression": (curve[-1][1] - min(loss for _, loss in curve)) if curve else None,
        "median_logged_train_tokens_per_second": statistics.median(speeds) if speeds else None,
        "train_phases": _phase_summary(train_events, through_tokens),
        "peak_memory_bytes": peak_memory,
        "logged_train_updates": train_count,
        "logged_gradient_clip_fraction": clip_count / train_count if train_count else None,
        "max_logged_gradient_norm": peak_gradient,
        "sampled_stability_updates": stability_count,
        "max_sampled_nonfinite_fraction": max_nonfinite,
        "max_sampled_residual_rms": max_residual,
        "max_sampled_attention_logit_abs": max_attention_logit,
        "min_sampled_attention_entropy": min_attention_entropy,
        "max_sampled_fp8_relative_rms_error": max_fp8_error,
        "max_sampled_update_to_weight_rms": max_update_ratio,
        "incidents": incident_types,
        "probe_metrics": _metric_summary(probe_metrics),
        "numerical_failure_or_sampled_nonfinite": numerical_failure,
        "note": "gradient/activation statistics are sampled; no universal safe thresholds are assumed",
    }


def compare_signals(candidate: dict, baseline: dict) -> dict:
    """Positive AUC improvement means faster optimization at equal tokens."""
    a = {item["tokens_seen"]: item["en_loss"] for item in candidate["validation_curve"]}
    b = {item["tokens_seen"]: item["en_loss"] for item in baseline["validation_curve"]}
    tokens = sorted(a.keys() & b.keys())
    if len(tokens) >= 2:
        area_a = sum((a[left] + a[right]) * (right - left) / 2
                     for left, right in zip(tokens, tokens[1:]))
        area_b = sum((b[left] + b[right]) * (right - left) / 2
                     for left, right in zip(tokens, tokens[1:]))
        auc_gain = (area_b - area_a) / area_b if area_b > 0 else None
    else:
        auc_gain = None
    speed_a, speed_b = (candidate["median_logged_train_tokens_per_second"],
                        baseline["median_logged_train_tokens_per_second"])
    memory_a, memory_b = candidate["peak_memory_bytes"], baseline["peak_memory_bytes"]
    candidate_metrics, baseline_metrics = candidate["probe_metrics"], baseline["probe_metrics"]
    probe_differences = {}
    for name in sorted(candidate_metrics.keys() & baseline_metrics.keys()):
        a_last = candidate_metrics[name]["last"]
        b_last = baseline_metrics[name]["last"]
        if a_last is None or b_last is None:
            continue
        matched = a_last["tokens_seen"] == b_last["tokens_seen"]
        probe_differences[name] = {
            "candidate_last": a_last, "baseline_last": b_last,
            "same_last_sample_tokens": matched,
            "last_difference": a_last["value"] - b_last["value"] if matched else None,
            "last_ratio": a_last["value"] / b_last["value"] if matched and b_last["value"] else None,
            "interpretation": "direction only; inspect function, scale, and loss before calling this beneficial or harmful"}
    return {"matched_validation_points": len(tokens),
            "english_loss_curve_auc_improvement": auc_gain,
            "logged_training_speed_ratio": speed_a / speed_b if speed_a and speed_b else None,
            "peak_memory_ratio": memory_a / memory_b if memory_a and memory_b else None,
            "probe_differences": probe_differences,
            "candidate_only_probes": sorted(candidate_metrics.keys() - baseline_metrics.keys()),
            "baseline_only_probes": sorted(baseline_metrics.keys() - candidate_metrics.keys()),
            "candidate": candidate, "baseline": baseline,
            "note": "quality, learning efficiency, stability and cost are separate axes; no hidden weighted winner"}


def compare_checkpoint_diagnostics(candidate: dict, baseline: dict) -> dict:
    """Expose every measured position and optional development-task difference."""
    positions_a = candidate.get("position_buckets") or {}
    positions_b = baseline.get("position_buckets") or {}
    if positions_a.keys() != positions_b.keys():
        raise ValueError("candidate and baseline position domains differ")
    positions = {}
    for domain in positions_a:
        if positions_a[domain].keys() != positions_b[domain].keys():
            raise ValueError(f"candidate and baseline position buckets differ for {domain}")
        positions[domain] = {}
        for bucket in positions_a[domain]:
            a, b = positions_a[domain][bucket]["loss"], positions_b[domain][bucket]["loss"]
            positions[domain][bucket] = {"candidate_loss": a, "baseline_loss": b,
                                         "relative_improvement": (b - a) / b if b else None,
                                         "candidate_tokens": positions_a[domain][bucket]["tokens"],
                                         "baseline_tokens": positions_b[domain][bucket]["tokens"]}
    tasks_a, tasks_b = candidate.get("tasks"), baseline.get("tasks")
    if bool(tasks_a) != bool(tasks_b):
        raise ValueError("candidate and baseline development-task availability differs")
    task_differences = None
    if tasks_a:
        per_task_a = tasks_a["summary"]["per_task"]
        per_task_b = tasks_b["summary"]["per_task"]
        if per_task_a.keys() != per_task_b.keys():
            raise ValueError("candidate and baseline development tasks differ")
        a_items = {(item["task"], item["item_id"]): item for item in tasks_a["records"]}
        b_items = {(item["task"], item["item_id"]): item for item in tasks_b["records"]}
        if a_items.keys() != b_items.keys() or len(a_items) != len(tasks_a["records"]) or len(b_items) != len(tasks_b["records"]):
            raise ValueError("development task items must match uniquely")
        task_differences = {
            "per_task": {name: {"candidate_mean_loss": per_task_a[name]["mean"],
                                 "baseline_mean_loss": per_task_b[name]["mean"],
                                 "relative_improvement": (
                                     (per_task_b[name]["mean"] - per_task_a[name]["mean"]) /
                                     per_task_b[name]["mean"] if per_task_b[name]["mean"] else None),
                                 "items": per_task_a[name]["n"],
                                 "candidate_summary": per_task_a[name],
                                 "baseline_summary": per_task_b[name]}
                         for name in sorted(per_task_a)},
            "per_item": [{"task": task, "item_id": item_id,
                          "candidate_loss": a_items[(task, item_id)]["loss"],
                          "baseline_loss": b_items[(task, item_id)]["loss"],
                          "loss_difference": (a_items[(task, item_id)]["loss"] -
                                              b_items[(task, item_id)]["loss"]),
                          "candidate_record": a_items[(task, item_id)],
                          "baseline_record": b_items[(task, item_id)]}
                         for task, item_id in sorted(a_items)],
            "note": "answer-only teacher-forced loss, not generated-answer accuracy"}
    return {"position_buckets": positions, "development_tasks": task_differences}
