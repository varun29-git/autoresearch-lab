"""Fixed campaign contract and quality arithmetic for architecture trials.

This module is deliberately outside the agent-editable architecture surface.
It has no cloud side effects and can be tested without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "research/policy.json"
INDIC = ("hi", "mr", "kn", "te", "ta", "gu", "bn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    from terminus.integrity import write_json_atomic

    write_json_atomic(path, value)


def policy() -> dict:
    value = read_json(POLICY_PATH)
    if value.get("version") != 1:
        raise ValueError("unsupported architecture policy version")
    if value["domains"] != ["en"]:
        raise ValueError("stage-one architecture screening uses an English-only proxy")
    if not math.isclose(sum(value["group_weights"].values()), 1.0, abs_tol=1e-12):
        raise ValueError("quality group weights must sum to one")
    if set(value["group_weights"]) != {"en"}:
        raise ValueError("English proxy score must have only an en weight")
    if not (0 < value["screen_tokens"] < value["full_tokens"]):
        raise ValueError("screen and full token horizons are invalid")
    global_update_tokens = (value["microbatch"] * value["max_seq_len"] *
                            value["gradient_accumulation_steps"])
    if value["full_tokens"] % global_update_tokens:
        raise ValueError("full horizon must end at a complete optimizer update for exact QAT continuation")
    if value["qat_probe_tokens"] <= 0 or value["qat_probe_lr_multiplier"] <= 0:
        raise ValueError("QAT probe tokens and LR multiplier must be positive")
    if not (0 <= value["qat_probe_warmup_fraction"] < 1
            and 0 <= value["qat_probe_decay_fraction"] < 1
            and value["qat_probe_warmup_fraction"] + value["qat_probe_decay_fraction"] < 1):
        raise ValueError("QAT probe warmup and decay fractions are invalid")
    if not (0 < value["min_parameters"] <= value["max_parameters"]):
        raise ValueError("parameter envelope is invalid")
    return value


def resolve_from(config_path: Path, name: str) -> Path:
    value = Path(name).expanduser()
    return (value if value.is_absolute() else config_path.parent / value).resolve()


def campaign(config_path: Path) -> dict:
    config_path = Path(config_path).resolve()
    value = read_json(config_path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", str(value.get("tag", ""))):
        raise ValueError("campaign tag must be a short filesystem-safe name")
    if value.get("gpu_count") != 1:
        raise ValueError("architecture campaign currently supports exactly one GPU")
    for key in ("manifest", "data_root", "results_root"):
        if not value.get(key):
            raise ValueError(f"campaign needs {key}")
        value[key] = str(resolve_from(config_path, value[key]))
    if value.get("task_manifest"):
        value["task_manifest"] = str(resolve_from(config_path, value["task_manifest"]))
    else:
        value["task_manifest"] = None
    if not Path(value["manifest"]).is_file():
        raise FileNotFoundError(f"campaign manifest is not staged: {value['manifest']}")
    if value["task_manifest"] and not Path(value["task_manifest"]).is_file():
        raise FileNotFoundError(f"campaign task_manifest is not staged: {value['task_manifest']}")
    if not Path(value["data_root"]).is_dir():
        raise FileNotFoundError(f"campaign data_root is not staged: {value['data_root']}")
    if value.get("baseline_trial") != "baseline":
        raise ValueError("baseline trial id must be 'baseline'")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def assert_clean_git() -> str:
    if git("status", "--porcelain"):
        raise ValueError("commit the architecture candidate before running a trial")
    return git("rev-parse", "HEAD")


def assert_edit_scope(base_commit: str, rules: dict) -> list[str]:
    changed = git("diff", "--name-only", f"{base_commit}..HEAD").splitlines()
    allowed = set(rules["editable_paths"])
    prefixes = tuple(rules["editable_prefixes"])
    forbidden = [name for name in changed if name not in allowed and not name.startswith(prefixes)]
    if forbidden:
        raise ValueError(f"campaign changed protected files: {forbidden}")
    return changed


def load_candidate(rules: dict) -> dict:
    model = read_json(ROOT / rules["candidate_file"])
    return validate_candidate(model, rules)


def validate_candidate(model: dict, rules: dict) -> dict:
    if not isinstance(model, dict):
        raise ValueError("candidate model must be a JSON object")
    for field in ("qk_norm_mode", "use_subln"):
        if field not in model:
            raise ValueError(f"candidate must set {field} explicitly; model defaults include old unit protections")
    for field, expected in rules["fixed_model_fields"].items():
        if model.get(field) != expected:
            raise ValueError(f"candidate changed fixed model field {field}")
    if model.get("qk_norm_mode") in rules["excluded_qk_norm_modes"]:
        raise ValueError("gain-free unit Q/K normalization was already tested; "
                         "do not repeat unit_rms in this campaign")
    return model


def count_candidate(model: dict, rules: dict) -> int:
    import torch
    from terminus.model import BitNetLlama, BitNetLlamaConfig

    with torch.device("meta"):
        count = BitNetLlama(BitNetLlamaConfig(**model)).count_parameters()["total"]
    if not rules["min_parameters"] <= count <= rules["max_parameters"]:
        raise ValueError(f"candidate has {count:,} parameters outside the "
                         f"{rules['min_parameters']:,}–{rules['max_parameters']:,} envelope")
    return count


def validate_task_manifest(path: Path, required_domains: list[str]) -> dict:
    manifest = read_json(path)
    if manifest.get("version") != 1 or not isinstance(manifest.get("tasks"), list):
        raise ValueError("task manifest must use version 1 with a tasks array")
    seen, hashes = set(), {}
    for entry in manifest["tasks"]:
        if entry.get("role") != "development" or entry.get("domain") not in required_domains:
            raise ValueError("task entries must be development tasks in campaign domains")
        if not entry.get("answer_only") or not entry.get("source") or not entry.get("split"):
            raise ValueError("each task needs answer_only=true, source and split")
        identity = (entry["domain"], entry.get("task"))
        if identity in seen or not identity[1]:
            raise ValueError("duplicate or unnamed task in manifest")
        seen.add(identity)
        task_path = resolve_from(path, entry["path"])
        actual = sha256(task_path)
        if actual != entry.get("sha256"):
            raise ValueError(f"task checksum mismatch: {task_path}")
        count = 0
        with task_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if not item.get("prompt") or not item.get("reference") or "rationale" in item:
                    raise ValueError(f"task rows must have prompt and answer-only reference: {task_path}")
                count += 1
        if count == 0:
            raise ValueError(f"empty task file: {task_path}")
        hashes[str(task_path)] = {"sha256": actual, "rows": count,
                                  "domain": entry["domain"], "task": entry["task"]}
    missing = set(required_domains) - {domain for domain, _ in seen}
    if missing:
        raise ValueError(f"development tasks missing domains: {sorted(missing)}")
    return hashes


def grouped_losses(losses: dict[str, float], rules: dict | None = None) -> dict[str, float]:
    if rules is not None and rules["domains"] == ["en"]:
        if set(losses) != {"en"}:
            raise ValueError("English proxy evaluation must contain en only")
        if not math.isfinite(float(losses["en"])) or float(losses["en"]) <= 0:
            raise ValueError("held-out loss must be finite and positive")
        return {"en": float(losses["en"])}
    if set(losses) != {"en", *INDIC, "math", "code"}:
        raise ValueError("held-out loss must cover each language/domain exactly once")
    if any(not math.isfinite(float(loss)) or float(loss) <= 0 for loss in losses.values()):
        raise ValueError("held-out losses must be finite and positive")
    return {"en": float(losses["en"]),
            "indic": sum(float(losses[name]) for name in INDIC) / len(INDIC),
            "code": float(losses["code"]), "math": float(losses["math"])}


def compare_losses(candidate_losses: dict[str, float], baseline_losses: dict[str, float],
                   rules: dict) -> dict:
    """Positive improvement is better; compare only matched-horizon evaluations."""
    candidate = grouped_losses(candidate_losses, rules)
    baseline = grouped_losses(baseline_losses, rules)
    relative = {group: (baseline[group] - candidate[group]) / baseline[group]
                for group in baseline}
    score = sum(rules["group_weights"][group] * relative[group] for group in relative)
    per_domain = {name: (float(baseline_losses[name]) - float(candidate_losses[name]))
                  / float(baseline_losses[name]) for name in baseline_losses}
    regressions = {name: value for name, value in per_domain.items()
                   if value < -rules["regression_guardrail"]}
    return {"relative_improvement": score, "group_improvement": relative,
            "per_domain_improvement": per_domain, "material_regressions": regressions,
            "decision_band": "uncertain" if abs(score) < rules["indifference_band"]
                             else "better" if score > 0 else "worse"}


def paired_batch_interval(candidate: dict, baseline: dict, rules: dict,
                          *, resamples: int = 2000, seed: int = 1337) -> dict:
    """Paired bootstrap over the evaluator's fixed spaced validation batches.

    Batches are correlated within a source, so this interval is a screening
    uncertainty estimate, not a substitute for new documents or new seeds.
    """
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if set(candidate) != set(rules["domains"]) or set(baseline) != set(rules["domains"]):
        raise ValueError("paired evaluation needs exactly the policy domains")
    pairs = {}
    for domain in rules["domains"]:
        a, b = candidate[domain]["batches"], baseline[domain]["batches"]
        if len(a) != len(b) or not a:
            raise ValueError(f"validation batches differ for {domain}")
        for left, right in zip(a, b):
            if (left["start"], left["examples"], left["tokens"]) != (
                    right["start"], right["examples"], right["tokens"]):
                raise ValueError(f"validation block selection differs for {domain}")
        pairs[domain] = (a, b)
    generator = random.Random(seed)
    scores = []
    for _ in range(resamples):
        sampled_candidate, sampled_baseline = {}, {}
        for domain, (a, b) in pairs.items():
            indices = [generator.randrange(len(a)) for _ in a]
            tokens = sum(a[index]["tokens"] for index in indices)
            sampled_candidate[domain] = sum(a[index]["loss_sum"] for index in indices) / tokens
            sampled_baseline[domain] = sum(b[index]["loss_sum"] for index in indices) / tokens
        scores.append(compare_losses(sampled_candidate, sampled_baseline, rules)["relative_improvement"])
    scores.sort()
    return {"low": scores[int(.025 * (resamples - 1))],
            "high": scores[int(.975 * (resamples - 1))],
            "resamples": resamples,
            "unit": "paired spaced validation batch; source/document dependence remains"}
