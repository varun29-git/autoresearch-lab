"""Local dashboard for one Terminus training directory.

Reads the files the trainer and the supervisor already write, once when the
page opens and again when Refresh is pressed. It does not poll, start
training, change a run, or invent a loss alarm. Bind address is loopback;
reach a remote host with an SSH tunnel.

    python -m production.amd.dashboard --run /path/to/output
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


# Selected balanced schedule. Captions say so. These are not measured rates.
GLOBAL_BATCH = 2_097_152
WARMUP_TOKENS = 1_006_632_960
DECAY_START_TOKENS = 718_257_104_812
DENSE_SCORED = 800_000_317_356
QAT_SCORED = 199_999_999_453
HEARTBEAT_STALE_SECONDS = 900
DISK_FLOOR_BYTES = 32 * 1024**3
SERIES_POINTS = 480
LANGUAGE_ORDER = ("en", "hi", "bn", "gu", "ta", "kn", "mr", "te")
# Tokenized benchmark catalog. Scores stay empty until a validation event
# records them. Reference tasks are not on the every-stop scorer.
BENCHMARKS = (
    {"task": "arc-challenge", "kind": "choice", "cadence": "every validation stop", "languages": "en", "development": 299, "final": 1172},
    {"task": "arc-easy", "kind": "choice", "cadence": "every validation stop", "languages": "en", "development": 570, "final": 2376},
    {"task": "belebele", "kind": "choice", "cadence": "every validation stop", "languages": "bn, en, gu, hi, kn, mr, ta, te", "development": 7200, "final": 0},
    {"task": "boolq", "kind": "choice", "cadence": "every validation stop", "languages": "en", "development": 3270, "final": 0},
    {"task": "global-mmlu", "kind": "choice", "cadence": "every validation stop", "languages": "bn, en, hi, te", "development": 1140, "final": 56159},
    {"task": "hellaswag", "kind": "choice", "cadence": "every validation stop", "languages": "en", "development": 10042, "final": 0},
    {"task": "include", "kind": "choice", "cadence": "every validation stop", "languages": "bn, hi, ta, te", "development": 90, "final": 2193},
    {"task": "mmlu", "kind": "choice", "cadence": "every validation stop", "languages": "en", "development": 1531, "final": 14042},
    {"task": "piqa", "kind": "choice", "cadence": "every validation stop", "languages": "en", "development": 1838, "final": 0},
    {"task": "winogrande", "kind": "choice", "cadence": "every validation stop", "languages": "en", "development": 1267, "final": 0},
    {"task": "gsm8k", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 0, "final": 1319},
    {"task": "humaneval", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 0, "final": 164},
    {"task": "indic-humaneval", "kind": "reference", "cadence": "sparse checkpoint", "languages": "bn, en, gu, hi, kn, mr, ta, te", "development": 0, "final": 1312},
    {"task": "indic-mbpp", "kind": "reference", "cadence": "sparse checkpoint", "languages": "bn, en, gu, hi, kn, mr, ta, te", "development": 0, "final": 4000},
    {"task": "indicgenbench-flores", "kind": "reference", "cadence": "sparse checkpoint", "languages": "bn, en, gu, hi, kn, mr, ta, te", "development": 13958, "final": 14168},
    {"task": "indicgenbench-xorqa", "kind": "reference", "cadence": "sparse checkpoint", "languages": "bn, en, gu, hi, kn, mr, ta, te", "development": 3995, "final": 4303},
    {"task": "indicqa", "kind": "reference", "cadence": "sparse checkpoint", "languages": "bn, gu, hi, kn, mr, ta, te", "development": 0, "final": 8509},
    {"task": "math", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 0, "final": 5000},
    {"task": "mbpp", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 90, "final": 500},
    {"task": "mgsm", "kind": "reference", "cadence": "sparse checkpoint", "languages": "bn, en, te", "development": 0, "final": 750},
    {"task": "scalingbench/bbh", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 6450, "final": 0},
    {"task": "scalingbench/human_eval", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 164, "final": 0},
    {"task": "scalingbench/math", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 5000, "final": 0},
    {"task": "scalingbench/mbpp", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 974, "final": 0},
    {"task": "scalingbench/mmlu", "kind": "reference", "cadence": "sparse checkpoint", "languages": "en", "development": 14042, "final": 0},
    {"task": "milu", "kind": "unavailable", "cadence": "not tokenized", "languages": "—", "development": None, "final": None},
    {"task": "facebook/flores", "kind": "unavailable", "cadence": "not tokenized", "languages": "—", "development": None, "final": None},
)

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Terminus run</title>
<style>
  :root {
    --bg: #f4f1ea; --ink: #1c1915; --muted: #5e584e; --line: #d9d2c5;
    --card: #fffdf8; --accent: #1f4d3a; --bad: #8c2f2f; --warn: #8a5a12;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.45 "Iowan Old Style", Palatino, Georgia, serif; }
  header { display: flex; justify-content: space-between; gap: 24px;
    align-items: flex-end; padding: 28px 32px 16px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 24px; font-weight: 600; margin: 0; letter-spacing: -0.02em; }
  h2 { font-size: 18px; font-weight: 600; margin: 0 0 10px; }
  p, .note { color: var(--muted); margin: 4px 0 0; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }
  main { padding: 20px 32px 48px; display: grid; gap: 28px; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button, label { font: 13px/1 ui-sans-serif, system-ui, sans-serif; }
  button { background: var(--ink); color: var(--bg); border: 0; padding: 8px 12px; cursor: pointer; }
  button.secondary { background: transparent; color: var(--ink); border: 1px solid var(--line); }
  .pill { font: 12px/1 ui-sans-serif, system-ui, sans-serif; letter-spacing: 0.04em;
    text-transform: uppercase; padding: 6px 8px; border: 1px solid var(--line); }
  .pill.bad { color: var(--bad); border-color: var(--bad); }
  .pill.warn { color: var(--warn); border-color: var(--warn); }
  .pill.ok { color: var(--accent); border-color: var(--accent); }
  .banner { padding: 12px 14px; border: 1px solid var(--bad); color: var(--bad);
    font: 14px/1.4 ui-sans-serif, system-ui, sans-serif; }
  .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px 24px; }
  .stat b { display: block; font-size: 22px; font-weight: 600; letter-spacing: -0.03em; }
  .stat span { display: block; margin-top: 2px; }
  .charts { display: grid; grid-template-columns: 1.4fr 1fr; gap: 22px; }
  .chart { background: var(--card); border: 1px solid var(--line); padding: 12px 12px 6px; }
  svg { width: 100%; height: 160px; display: block; }
  table { width: 100%; border-collapse: collapse; font: 13px/1.35 ui-sans-serif, system-ui, sans-serif; }
  th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 550; }
  .split { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 28px; }
  .mono { font-variant-numeric: tabular-nums; }
  @media (max-width: 980px) {
    .stats, .charts, .split { grid-template-columns: 1fr; }
    header, main { padding-left: 16px; padding-right: 16px; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>Terminus run</h1>
    <p id="path"></p>
  </div>
  <div class="row">
    <span id="pill" class="pill">waiting</span>
    <button id="refresh" type="button">Refresh</button>
    <span class="note" id="age"></span>
  </div>
</header>
<main id="main"></main>
<script>
const main = document.getElementById("main");
const pill = document.getElementById("pill");

function num(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Math.abs(value) >= 1e9) return (value / 1e9).toFixed(digits === undefined ? 3 : digits) + "B";
  if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(digits === undefined ? 2 : digits) + "M";
  return Number(value).toLocaleString(undefined, {maximumFractionDigits: digits === undefined ? 4 : digits});
}
function pct(value) {
  if (value === null || value === undefined) return "—";
  return (value * 100).toFixed(2) + "%";
}
function when(unix) {
  if (!unix) return "—";
  return new Date(unix * 1000).toLocaleString();
}
function ago(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 90) return Math.round(seconds) + "s ago";
  if (seconds < 5400) return Math.round(seconds / 60) + "m ago";
  return (seconds / 3600).toFixed(1) + "h ago";
}
function esc(value) {
  return String(value ?? "").replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}
function chart(title, unit, points, caption) {
  if (!points || points.length < 2) return "";
  const ys = points.map(p => p.y).filter(y => y !== null && Number.isFinite(y));
  if (ys.length < 2) return "";
  const labelMin = Math.min(...ys), labelMax = Math.max(...ys);
  let min = labelMin, max = labelMax;
  if (min === max) { min -= 1; max += 1; }
  const w = 640, h = 160, pad = 8;
  const coords = points.filter(p => Number.isFinite(p.y)).map((p, i, arr) => {
    const x = pad + (i / Math.max(1, arr.length - 1)) * (w - pad * 2);
    const y = pad + (1 - (p.y - min) / (max - min)) * (h - pad * 2);
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");
  return `<section class="chart"><h2>${esc(title)}</h2>
    <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}">
      <polyline fill="none" stroke="#1f4d3a" stroke-width="1.6" points="${coords}"></polyline>
    </svg>
    <p class="note">${esc(unit)}. Latest ${num(points[points.length - 1].y, 4)}, range ${num(labelMin, 4)} to ${num(labelMax, 4)}. ${esc(caption)}</p>
  </section>`;
}
function table(headers, rows) {
  if (!rows.length) return "";
  return `<table><thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>
    ${rows.map(row => `<tr>${row.map(cell => `<td class="mono">${esc(cell)}</td>`).join("")}</tr>`).join("")}
  </tbody></table>`;
}
function render(data) {
  document.getElementById("path").textContent = data.run;
  pill.textContent = data.status.replaceAll("_", " ");
  pill.className = "pill " + (data.status === "failed" || data.status === "stale" ? "bad"
    : data.status === "clipping" || data.status === "low_disk" ? "warn"
    : data.status === "running" || data.status === "complete" ? "ok" : "");
  document.getElementById("age").textContent = "Read " + when(data.generated_unix);
  const train = data.latest_train || {};
  const progress = data.progress || {};
  const started = data.status !== "waiting";
  const banners = (data.banners || []).map(text => `<div class="banner">${esc(text)}</div>`).join("");
  const stats = started ? [
    ["Scored tokens", num(progress.tokens_seen, 3), progress.fraction_label || ""],
    ["Updates", num(progress.updates, 0), progress.band || ""],
    ["Learning-rate scale", num(train.lr_scale, 4), "Warmup, stable, and decay multiplier"],
    ["Group LR", num(train.optimizer_group_lr_min, 6) + " – " + num(train.optimizer_group_lr_max, 6), "Smallest and largest optimizer group"],
    ["Quantization", num(train.quantization_strength, 2), "Dense stage stays at 0. QAT is 1."],
    ["Valid targets", num(train.valid_targets, 0), "Must equal the planned update"],
  ] : [];
  const charts = [
    chart("Training loss", "Mean NLL of each logged update", data.series.loss, ""),
    chart("Gradient norm before clipping", "Global L2 norm", data.series.grad_norm, ""),
    chart("Clip coefficient", "1 = no clip", data.series.clip, ""),
    chart("Scored tokens per second", "Labels / second since the previous log", data.series.rate, ""),
    chart("Data wait", "Seconds the update waited on the next batch", data.series.data_wait, ""),
    chart("Peak GPU memory", "GiB allocated on the logging rank", data.series.memory, ""),
  ].filter(Boolean).join("");
  const validations = data.validations || [];
  const lastVal = validations[validations.length - 1];
  let validation = "";
  if (lastVal && Object.keys(lastVal.languages || {}).length) {
    const langRows = Object.keys(lastVal.languages).sort().map(lang => {
      const row = lastVal.languages[lang];
      return [lang, num(row.loss, 4), num(row.perplexity, 3), num(row.tokens, 0)];
    });
    const mixture = lastVal.mixture || {};
    validation = `<section><h2>Latest validation, update ${lastVal.updates}</h2>
      <p class="note">${esc(when(lastVal.time_unix))}. Mixture loss ${num(mixture.loss, 4)}, perplexity ${num(mixture.perplexity, 3)}. The mixture is the integer language-weight mean.</p>
      ${table(["Language", "Loss", "Perplexity", "Tokens"], langRows)}
      ${chart("Mixture validation loss", "Weighted mean NLL at each validation stop", data.series.val_loss, "")}
    </section>`;
  }
  const benchRows = (data.benchmarks || []).map(row => [
    row.task, row.languages, row.cadence,
    row.development == null ? "—" : num(row.development, 0),
    row.final == null ? "—" : num(row.final, 0),
    row.state === "scored" ? pct(row.accuracy) : (row.state === "not_tokenized" ? "not tokenized" : "waiting"),
    row.state === "scored" ? num(row.mean_margin, 4) : "—",
    row.state === "scored" ? `${row.correct}/${row.items}` : "—",
  ]);
  const benchmarks = `<section><h2>Benchmarks</h2>
    <p class="note">Choice rows fill from the development split on each validation stop. Final-role items stay off that stop. Reference rows are tokenized and stay waiting until a score is written. MILU and facebook/flores were not tokenized.</p>
    ${table(["Task", "Languages", "When", "Development", "Final", "Accuracy", "Mean margin", "Correct"], benchRows)}
  </section>`;
  const stability = data.stability;
  let stabilityHtml = "";
  if (stability && (stability.worst.length || stability.layers.length)) {
    stabilityHtml = `<section><h2>Stability probe, update ${stability.updates}</h2>
      <p class="note">First microbatch of that update. The probe does not change weights, gradients, or RNG. ${esc(when(stability.time_unix))}.</p>
      ${table(["Probe", "Signal", "Value"], stability.worst.map(row => [row.name, row.signal, num(row.value, 5)]))}
      ${table(["Layer", "Residual RMS", "Attention entropy", "Max attention probability"], stability.layers.map(row => [row.layer, num(row.residual_rms, 4), num(row.entropy, 3), num(row.max_probability, 4)]))}
    </section>`;
  }
  const incidentRows = data.incidents || [];
  const incidents = incidentRows.length ? `<section><h2>Incidents</h2>
    <p class="note">A nonfinite loss, a target-count mismatch, a gradient failure, or a run failure aborts the update.</p>
    ${table(["When", "Event", "Detail"], incidentRows.map(row => [when(row.time_unix), row.event, row.detail]))}
  </section>` : "";
  const ops = data.operations || {};
  const opsHtml = started ? `<section><h2>Checkpoint, disk, and allocation</h2>
    <div class="stats">
      <div class="stat"><b>${esc(ops.checkpoint_file || "—")}</b><span class="note">Latest checkpoint, update ${num(ops.checkpoint_updates, 0)}, ${num((ops.checkpoint_bytes || 0) / (1024**3), 2)} GiB</span></div>
      <div class="stat"><b>${num((ops.free_bytes || 0) / (1024**3), 1)} GiB</b><span class="note">Free on the run volume. Default floor is 32 GiB.</span></div>
      <div class="stat"><b>${num(ops.remaining_gpu_hours, 1)}</b><span class="note">GPU-hours left on the heartbeat budget</span></div>
      <div class="stat"><b>${ago(ops.heartbeat_age_seconds)}</b><span class="note">Heartbeat. Supervisor default stale limit is 900s.</span></div>
    </div>
    <p class="note">${esc(ops.heartbeat_status || "No operations heartbeat yet.")} ${esc(ops.supervision_line || "")}</p>
  </section>` : "";
  const waiting = started ? "" : `<p class="note">Waiting for the training directory. Refresh reads events.jsonl, stability.jsonl, incidents.jsonl, heartbeat.json, and checkpoints/latest.json when you ask. Scores stay blank until those files exist.</p>`;
  const logged = charts ? `<section><h2>Logged updates</h2><p class="note">${esc(data.series_caption || "")}</p><div class="charts">${charts}</div></section>` : "";
  const headline = stats.length ? `<section class="stats">${stats.map(item => `<div class="stat"><b>${esc(item[1])}</b><span class="note">${esc(item[0])}. ${esc(item[2])}</span></div>`).join("")}</section>` : "";
  main.innerHTML = banners + waiting + headline + logged + validation + benchmarks + stabilityHtml + incidents + opsHtml;
}
async function load() {
  const response = await fetch("/api/status", {cache: "no-store"});
  render(await response.json());
}
document.getElementById("refresh").onclick = load;
load();
</script>
</body>
</html>
"""


def _loads(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _jsonl(path):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = _loads(line)
            if row is not None:
                rows.append(row)
    return rows


def _last_json(path, window=1 << 20):
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size == 0:
        return None
    with path.open("rb") as handle:
        while True:
            handle.seek(max(0, size - window))
            raw = handle.read()
            if size > window:
                raw = raw.split(b"\n", 1)[-1]
            lines = [line for line in raw.splitlines() if line.strip()]
            if lines:
                row = _loads(lines[-1].decode("utf-8", "replace"))
                if row is not None or window >= size or window >= 64 << 20:
                    return row
            if window >= size or window >= 64 << 20:
                return None
            window *= 4


def _downsample(rows, limit):
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    chosen = [rows[min(len(rows) - 1, int(index * step))] for index in range(limit - 1)]
    chosen.append(rows[-1])
    return chosen


def _point(row, key, scale=1.0):
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return {"x": row.get("updates"), "y": value / scale}


def _walk(node, prefix, found):
    if not isinstance(node, dict):
        return
    signals = {
        "nonfinite_fraction": "nonfinite fraction",
        "relative_rms_error": "FP8 relative RMS",
        "mean_entropy": "attention entropy",
        "max_probability": "max attention probability",
        "update_to_weight_rms": "update / weight RMS",
        "rms": "RMS",
    }
    for key, label in signals.items():
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            found.append({"name": prefix or key, "signal": label, "value": value, "key": key})
    for key, child in node.items():
        if isinstance(child, dict):
            _walk(child, f"{prefix}.{key}" if prefix else key, found)


def _stability_view(row):
    if not row:
        return None
    found = []
    _walk(row.get("probes") or {}, "", found)
    def worst(key, reverse):
        rows = [item for item in found if item["key"] == key]
        rows.sort(key=lambda item: item["value"], reverse=reverse)
        return rows[:5]
    layers = []
    probes = row.get("probes") or {}
    for index in range(24):
        name = f"layer_{index:02d}"
        residual = (probes.get(f"{name}.residual") or {}).get("rms")
        attention = probes.get(f"{name}.attention_sample") or {}
        if residual is None and not attention:
            continue
        layers.append({"layer": name, "residual_rms": residual,
                       "entropy": attention.get("mean_entropy"),
                       "max_probability": attention.get("max_probability")})
    finite = [item["value"] for item in found if item["key"] == "nonfinite_fraction"]
    fp8 = [item["value"] for item in found if item["key"] == "relative_rms_error"]
    entropy = [item["value"] for item in found if item["key"] == "mean_entropy"]
    ratio = [item["value"] for item in found if item["key"] == "update_to_weight_rms"]
    highlights = worst("nonfinite_fraction", True) + worst("relative_rms_error", True)
    highlights += worst("update_to_weight_rms", True)
    low_entropy = worst("mean_entropy", False)
    return {"time_unix": row.get("time_unix"), "updates": (row.get("updates")),
            "summary": {"max_nonfinite_fraction": max(finite) if finite else None,
                        "max_fp8_relative_rms": max(fp8) if fp8 else None,
                        "min_attention_entropy": min(entropy) if entropy else None,
                        "max_update_to_weight": max(ratio) if ratio else None},
            "worst": (highlights + low_entropy)[:12], "layers": layers}


def _incident_detail(row):
    if row.get("event") == "run_failure":
        oom = " CUDA out of memory." if row.get("cuda_oom") else ""
        return f"{row.get('error_type', '')}: {row.get('error', '')}.{oom}".strip()
    if row.get("event") == "token_count_mismatch":
        return f"valid targets {row.get('valid_targets')} against planned {row.get('update_tokens')}"
    if row.get("error"):
        return str(row.get("error"))
    state = row.get("state") or {}
    if state.get("updates") is not None:
        return f"update {state.get('updates')}, tokens {state.get('tokens_seen')}"
    return ""


def _benchmark_rows(scores):
    scores = scores if isinstance(scores, dict) else {}
    rows = []
    known = set()
    for spec in BENCHMARKS:
        known.add(spec["task"])
        score = scores.get(spec["task"])
        row = dict(spec)
        if (spec["kind"] != "unavailable" and isinstance(score, dict)
                and isinstance(score.get("accuracy"), (int, float)) and math.isfinite(score["accuracy"])):
            row.update(state="scored", accuracy=score.get("accuracy"),
                       mean_margin=score.get("mean_margin"), correct=score.get("correct"),
                       items=score.get("items"))
        elif spec["kind"] == "unavailable":
            row["state"] = "not_tokenized"
        else:
            row["state"] = "waiting"
        rows.append(row)
    for task, score in sorted(scores.items()):
        if task in known or not isinstance(score, dict):
            continue
        rows.append({"task": task, "kind": score.get("kind"), "cadence": "recorded",
                     "languages": "—", "development": None, "final": None, "state": "scored",
                     "accuracy": score.get("accuracy"), "mean_margin": score.get("mean_margin"),
                     "correct": score.get("correct"), "items": score.get("items")})
    return rows


def _band(tokens, strength):
    if not isinstance(tokens, (int, float)):
        return "No scored tokens yet"
    if isinstance(strength, (int, float)) and strength >= 1:
        return "QAT, strength 1"
    if tokens < WARMUP_TOKENS:
        return "Warmup, selected schedule"
    if tokens >= DENSE_SCORED:
        return "Dense scored budget reached"
    if tokens >= DECAY_START_TOKENS:
        return "Decay, final 10% of dense slots"
    return "Stable, selected schedule"


def summarize(run, supervision=None, now=None):
    run = Path(run)
    now = time.time() if now is None else now
    events = _jsonl(run / "events.jsonl")
    incidents = _jsonl(run / "incidents.jsonl")
    startup = next((row for row in events if row.get("event") == "startup"), None)
    trains = [row for row in events if row.get("event") == "train"]
    validations = []
    for row in events:
        if row.get("event") != "validation":
            continue
        languages = {key: value for key, value in (row.get("languages") or {}).items() if isinstance(value, dict)}
        validations.append({"time_unix": row.get("time_unix"), "updates": row.get("updates"),
                            "tokens_seen": row.get("tokens_seen"), "mixture": row.get("mixture"),
                            "languages": languages, "benchmarks": row.get("benchmarks") or {}})
    terminal = next((row for row in reversed(events) if row.get("event") in ("complete", "stopped")), None)
    latest = trains[-1] if trains else None
    stability = _stability_view(_last_json(run / "stability.jsonl"))
    heartbeat = None
    heart_path = run / "heartbeat.json"
    if heart_path.is_file():
        try:
            heartbeat = json.loads(heart_path.read_text())
        except json.JSONDecodeError:
            heartbeat = None
    checkpoint = None
    latest_path = run / "checkpoints" / "latest.json"
    if latest_path.is_file():
        try:
            checkpoint = json.loads(latest_path.read_text())
        except json.JSONDecodeError:
            checkpoint = None
    supervision_rows = []
    supervision_line = ""
    if supervision:
        root = Path(supervision)
        supervision_rows = _jsonl(root / "events.jsonl")[-8:]
        watchdog = _jsonl(root / "watchdog.jsonl")[-3:]
        if (root / "provider_stopped.json").is_file():
            supervision_line = "Provider stop is verified."
        elif watchdog:
            supervision_line = "Last watchdog reason: " + str(watchdog[-1].get("reason"))
        elif supervision_rows:
            supervision_line = "Last supervisor event: " + str(supervision_rows[-1].get("event"))
    usage = shutil.disk_usage(run if run.exists() else run.parent)
    heart_age = None if not heartbeat or not isinstance(heartbeat.get("time_unix"), (int, float)) else now - heartbeat["time_unix"]
    budget = (heartbeat or {}).get("budget") or {}
    failure = next((row for row in reversed(incidents) if row.get("event") == "run_failure"), None)
    oom = bool(failure and failure.get("cuda_oom"))
    banners = []
    if failure:
        banners.append(_incident_detail(failure) or "Run failure recorded.")
    if oom:
        banners.append("The failure is a CUDA out-of-memory error. The documented fallback is a new run at microbatch 8 or 4 with the same global batch.")
    if heart_age is not None and heart_age > HEARTBEAT_STALE_SECONDS and (terminal is None):
        banners.append(f"Operations heartbeat is {int(heart_age)} seconds old. The supervisor treats 900 seconds as stale.")
    started = bool(latest or startup or heartbeat)
    if started and usage.free < DISK_FLOOR_BYTES:
        banners.append(f"Free space on the run volume is {usage.free / 1024**3:.1f} GiB. The operations default floor is 32 GiB.")
    tokens = (latest or {}).get("tokens_seen")
    strength = (latest or {}).get("quantization_strength")
    if terminal and terminal.get("event") == "complete":
        status = "complete"
    elif failure and (latest is None or (failure.get("time_unix") or 0) >= (latest.get("time_unix") or 0)):
        status = "failed"
    elif heart_age is not None and heart_age > HEARTBEAT_STALE_SECONDS and terminal is None:
        status = "stale"
    elif terminal and terminal.get("event") == "stopped":
        status = "stopped"
    elif started and usage.free < DISK_FLOOR_BYTES and latest:
        status = "low_disk"
    elif latest and isinstance(latest.get("grad_clip_coefficient"), (int, float)) and latest["grad_clip_coefficient"] < 1:
        status = "clipping"
    elif latest or startup:
        status = "running"
    else:
        status = "waiting"
    dense_fraction = None if not isinstance(tokens, (int, float)) or strength == 1 else tokens / DENSE_SCORED
    qat_fraction = None
    if strength == 1 and isinstance(tokens, (int, float)):
        qat_fraction = max(0.0, tokens - DENSE_SCORED) / QAT_SCORED
    fraction = qat_fraction if qat_fraction is not None else dense_fraction
    fraction_label = ""
    if fraction is not None:
        which = "QAT scored budget" if qat_fraction is not None else "dense scored budget"
        fraction_label = f"{fraction * 100:.2f}% of the assembled {which}"
    series = {
        "loss": _downsample([item for item in (_point(row, "loss") for row in trains) if item], SERIES_POINTS),
        "grad_norm": _downsample([item for item in (_point(row, "grad_norm_before_clip") for row in trains) if item], SERIES_POINTS),
        "clip": _downsample([item for item in (_point(row, "grad_clip_coefficient") for row in trains) if item], SERIES_POINTS),
        "rate": _downsample([item for item in (_point(row, "tokens_per_second") for row in trains) if item], SERIES_POINTS),
        "data_wait": _downsample([item for item in (_point(row, "data_wait_seconds") for row in trains) if item], SERIES_POINTS),
        "memory": _downsample([item for item in (_point(row, "peak_memory_bytes", 1024**3) for row in trains) if item], SERIES_POINTS),
        "val_loss": [item for item in ({"x": row["updates"], "y": (row.get("mixture") or {}).get("loss")} for row in validations) if isinstance(item["y"], (int, float))],
    }
    checkpoint_state = (checkpoint or {}).get("state") or {}
    return {
        "run": str(run),
        "generated_unix": now,
        "status": status,
        "banners": banners,
        "startup": {"parameters": ((startup or {}).get("counts") or {}).get("trainable"),
                    "world_size": (startup or {}).get("world_size"),
                    "time_unix": (startup or {}).get("time_unix")} if startup else None,
        "latest_train": latest,
        "progress": {"tokens_seen": tokens, "updates": (latest or {}).get("updates"),
                     "content_tokens_seen": (latest or {}).get("content_tokens_seen"),
                     "input_slots_seen": (latest or {}).get("input_slots_seen"),
                     "band": _band(tokens, strength), "fraction_label": fraction_label},
        "series": series,
        "series_caption": "Source: events.jsonl. Loss and data wait are the logged update, not a 12-update average. Rate covers the time since the previous log.",
        "plan": {"global_batch": GLOBAL_BATCH, "warmup_tokens": WARMUP_TOKENS,
                 "decay_start_tokens": DECAY_START_TOKENS, "dense_scored": DENSE_SCORED,
                 "qat_scored": QAT_SCORED, "warmup_updates": WARMUP_TOKENS // GLOBAL_BATCH},
        "validations": validations[-40:],
        "benchmarks": _benchmark_rows(validations[-1]["benchmarks"] if validations else {}),
        "stability": stability,
        "incidents": [{"time_unix": row.get("time_unix"), "event": row.get("event"),
                       "detail": _incident_detail(row)} for row in reversed(incidents[-40:])],
        "operations": {
            "checkpoint_file": (checkpoint or {}).get("file"),
            "checkpoint_bytes": (checkpoint or {}).get("size_bytes"),
            "checkpoint_updates": checkpoint_state.get("updates"),
            "checkpoint_tokens": checkpoint_state.get("tokens_seen"),
            "free_bytes": usage.free,
            "heartbeat_age_seconds": heart_age,
            "heartbeat_status": (heartbeat or {}).get("status"),
            "remaining_gpu_hours": budget.get("remaining_gpu_hours"),
            "elapsed_gpu_hours": budget.get("elapsed_gpu_hours"),
            "training_seconds": budget.get("training_seconds"),
            "supervision_line": supervision_line,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    run = None
    supervision = None

    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, content_type):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            body = json.dumps(summarize(self.run, self.supervision), allow_nan=False)
            self._send(200, body, "application/json")
            return
        self._send(404, "not found", "text/plain; charset=utf-8")


def serve(run, supervision=None, host="127.0.0.1", port=8765):
    _Handler.run = str(Path(run))
    _Handler.supervision = str(supervision) if supervision else None
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"dashboard http://{host}:{port}/  run={run}", flush=True)
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch one Terminus output directory.")
    parser.add_argument("--run", required=True, help="Trainer output directory")
    parser.add_argument("--supervision", help="Supervisor directory, if it is separate from the run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.run, args.supervision, args.host, args.port)


if __name__ == "__main__":
    main()
