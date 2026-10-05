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


def summarize_run(run_dir: Path, *, through_tokens: int | None = None) -> dict:
    """Keep diagnostic samples separate from hard numerical failures."""
    run_dir = Path(run_dir)
    speeds, curve, incident_types = [], [], {}
    clip_count = train_count = stability_count = 0
    peak_memory = peak_gradient = max_nonfinite = max_residual = None
    max_attention_logit = min_attention_entropy = max_fp8_error = max_update_ratio = None
    for event in _records(run_dir / "events.jsonl"):
        tokens = event.get("tokens_seen", 0)
        if through_tokens is not None and tokens > through_tokens:
            continue
        if event.get("event") == "train":
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
    return {"matched_validation_points": len(tokens),
            "english_loss_curve_auc_improvement": auc_gain,
            "logged_training_speed_ratio": speed_a / speed_b if speed_a and speed_b else None,
            "peak_memory_ratio": memory_a / memory_b if memory_a and memory_b else None,
            "candidate": candidate, "baseline": baseline,
            "note": "quality, learning efficiency, stability and cost are separate axes; no hidden weighted winner"}
