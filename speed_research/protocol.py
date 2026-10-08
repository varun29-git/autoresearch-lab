"""Fail-closed, CPU-only provenance and paired speed-screen analysis.

This checks evidence consistency. It cannot establish that an agent-supplied
measurement is truthful, or replace native numerical and quality reviewers.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

from .physics import positive

ROOT = Path(__file__).resolve().parent

# These select execution, not a license to change their numerical semantics.
# The native constructor, parity and canonical parameter/state manifest remain
# mandatory; config validation alone cannot prove an implementation equivalent.
EXECUTION_KEYS = {
    "attention_backend", "linear_backend", "loss_backend", "loss_chunk_size",
    "gradient_checkpointing", "fp8_quantizer", "fp8_cache_weights", "fused_pointwise",
    "fuse_projections", "fused_mlp", "fused_qkv", "fp8_wgrad_splits",
}


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject_constant, object_pairs_hook=unique_keys)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_anchor(repo=".", *, checkout=False):
    anchor = read_json(ROOT / "anchor.json")
    failures = []
    for path, expected in anchor["source_sha256"].items():
        content = subprocess.check_output(["git", "-C", str(repo), "show", f"{anchor['winner_commit']}:{path}"])
        if hashlib.sha256(content).hexdigest() != expected:
            failures.append(path)
        if checkout and (not (Path(repo) / path).is_file() or file_digest(Path(repo) / path) != expected):
            failures.append(f"working tree: {path}")
    if failures:
        raise ValueError("winner source mismatch: " + ", ".join(failures))
    return {"winner_commit": anchor["winner_commit"], "verified_files": len(anchor["source_sha256"]),
            "checkout_verified": checkout}


def require_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")


def validate_campaign(campaign):
    if campaign.get("schema") != 1 or campaign.get("status") != "ready":
        raise ValueError("campaign must be schema 1, status ready; the template is not launchable")
    if (type(campaign.get("research_gpus")) is not int or type(campaign.get("production_gpus")) is not int
            or campaign["research_gpus"] != 1 or campaign["production_gpus"] != 32):
        raise ValueError("this campaign requires 1 research GPU and 32 production GPUs")
    if campaign.get("production_gpu_hours") != 6000 or campaign.get("experiments_in_production_budget") is not False:
        raise ValueError("production has 6000 GPU-hours including QAT; research is separate")
    positive(campaign.get("research_gpu_hour_cap"), "research_gpu_hour_cap")
    positive(campaign.get("trial_gpu_hour_cap"), "trial_gpu_hour_cap")
    if campaign["trial_gpu_hour_cap"] > campaign["research_gpu_hour_cap"]:
        raise ValueError("trial cap exceeds research cap")
    for field in ("campaign_id", "research_hardware", "production_hardware", "production_topology",
                  "environment_digest", "quality_protocol"):
        if not isinstance(campaign.get(field), str) or not campaign[field].strip():
            raise ValueError(f"missing {field}")
    fraction = positive(campaign.get("qat_token_fraction"), "qat_token_fraction", zero=True)
    if not 0 < fraction < 1:
        raise ValueError("this campaign must budget both dense training and QAT")
    positive(campaign.get("global_tokens_per_update"), "global_tokens_per_update")
    if (not isinstance(campaign["global_tokens_per_update"], int)
            or campaign["global_tokens_per_update"] % 4096):
        raise ValueError("full-update profile must contain an integer number of 4096-token sequences")
    for field in ("data_manifest_sha256", "tokenizer_sha256", "optimizer_recipe_sha256",
                  "schedule_sha256", "dense_checkpoint_sha256", "qat_checkpoint_sha256",
                  "quality_protocol_sha256"):
        require_hash(campaign.get(field), field)
    if campaign.get("model") != read_json(ROOT / "target_model.json"):
        raise ValueError("model must match the complete frozen stack-004 target contract")
    policy = campaign.get("screen", {})
    for field in ("min_pairs", "min_updates", "min_block_seconds", "memory_limit_bytes",
                  "min_speedup", "max_baseline_drift"):
        positive(policy.get(field), field)
    if not isinstance(policy["min_pairs"], int) or policy["min_pairs"] < 6:
        raise ValueError("at least six independent paired blocks required")
    if not isinstance(policy["min_updates"], int) or policy["min_updates"] < 1:
        raise ValueError("min_updates must be positive integer")
    if policy["min_speedup"] <= 1 or policy["max_baseline_drift"] >= 1:
        raise ValueError("invalid speed or drift threshold")
    return campaign


def validate_candidate_config(config, *, phase):
    reference = read_json(ROOT / "target_model.json")
    if phase not in {"dense", "qat"} or config.get("weight_mode") != ("dense" if phase == "dense" else "ternary"):
        raise ValueError("candidate weight mode does not match phase")
    if set(config) != set(reference):
        raise ValueError("candidate must materialize every reference model option; no unknown options")
    changed = {key for key in reference if config[key] != reference[key]}
    forbidden = changed - EXECUTION_KEYS - {"weight_mode"}
    if forbidden:
        raise ValueError("candidate changed protected model semantics: " + ", ".join(sorted(forbidden)))
    return {"phase": phase, "execution_overrides": sorted(changed & EXECUTION_KEYS),
            "remaining_checks": "native config validation, canonical parameter/state mapping and numerical/quality parity"}


def make_lock(campaign):
    validate_campaign(campaign)
    # Protect the independent checker and policy inputs, separate from editable
    # training implementations. Store the resulting lock outside the checkout.
    files = {p.name: file_digest(p) for p in sorted(ROOT.iterdir())
             if p.is_file() and p.suffix in {".py", ".json"}}
    payload = {"campaign": campaign, "checker_files": files}
    return {**payload, "sha256": digest(payload)}


def validate_lock(lock):
    payload = {k: v for k, v in lock.items() if k != "sha256"}
    if digest(payload) != lock.get("sha256"):
        raise ValueError("lock digest mismatch")
    validate_campaign(lock["campaign"])
    current = {p.name: file_digest(p) for p in sorted(ROOT.iterdir())
               if p.is_file() and p.suffix in {".py", ".json"}}
    if current != lock.get("checker_files"):
        raise ValueError("protected checker changed: start a new campaign version")
    return lock["campaign"]


def analyze_screen(evidence, lock):
    """One-sided 95% paired block bootstrap; never declares production ready."""
    campaign = validate_lock(lock)
    policy = campaign["screen"]
    if evidence.get("schema") != 1 or evidence.get("lock_sha256") != lock["sha256"]:
        raise ValueError("evidence/lock mismatch")
    if evidence.get("stage") != "full_update" or evidence.get("phase") not in {"dense", "qat"}:
        raise ValueError("only full-update dense or QAT screens may be scored")
    if type(evidence.get("world_size")) is not int or evidence["world_size"] != 1 or evidence.get("native_sm90") is not True:
        raise ValueError("screen requires a native single-H100 measurement")
    if evidence.get("hardware") != campaign["research_hardware"]:
        raise ValueError("hardware does not match campaign")
    for key in ("baseline_commit", "candidate_commit"):
        value = evidence.get(key)
        if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {key}")
    if evidence["baseline_commit"] == evidence["candidate_commit"]:
        raise ValueError("baseline and candidate must be different commits")
    pairs = evidence.get("pairs", [])
    if len(pairs) < policy["min_pairs"]:
        raise ValueError("insufficient independent paired blocks")
    ratios, baselines, peaks, seen_ids, orders = [], [], [], set(), []
    billed_seconds = 0
    for pair in pairs:
        if pair.get("order") not in {"AB", "BA"}:
            raise ValueError("pair must record AB or BA order")
        orders.append(pair["order"])
        rates, counts = [], []
        for side in ("baseline", "candidate"):
            run = pair[side]
            run_id = run.get("run_id")
            if not isinstance(run_id, str) or not run_id or run_id in seen_ids:
                raise ValueError("run IDs must be nonempty and unique")
            seen_ids.add(run_id)
            for key in ("seconds", "valid_tokens", "updates", "peak_device_bytes", "billed_gpu_seconds"):
                positive(run.get(key), key)
            if run.get("finite") is not True or run.get("completed") is not True:
                return {"decision": "reject", "reason": "failed or nonfinite block", "production_ready": False}
            if (not isinstance(run["updates"], int) or not isinstance(run["valid_tokens"], int)
                    or run["updates"] < policy["min_updates"] or run["seconds"] < policy["min_block_seconds"]):
                raise ValueError("block is too short or counters are not integers")
            if run["valid_tokens"] != run["updates"] * campaign["global_tokens_per_update"]:
                raise ValueError("valid loss-bearing token count differs from frozen update contract")
            if run["billed_gpu_seconds"] < run["seconds"]:
                raise ValueError("billed time cannot exclude measured time")
            if not isinstance(run["peak_device_bytes"], int):
                raise ValueError("peak_device_bytes must be integer")
            billed_seconds += run["billed_gpu_seconds"]
            rates.append(run["valid_tokens"] / run["seconds"])
            counts.append((run["updates"], run["valid_tokens"]))
            peaks.append(run["peak_device_bytes"])
        if counts[0] != counts[1]:
            raise ValueError("paired arms must run equal update and token counts")
        baselines.append(rates[0])
        ratios.append(math.log(rates[1] / rates[0]))
    if abs(orders.count("AB") - orders.count("BA")) > 1:
        raise ValueError("AB/BA order must be balanced")
    if billed_seconds / 3600 > campaign["trial_gpu_hour_cap"]:
        return {"decision": "reject", "reason": "trial charge exceeds cap", "production_ready": False}
    if max(peaks) > policy["memory_limit_bytes"]:
        return {"decision": "reject", "reason": "production memory envelope exceeded", "production_ready": False}
    drift = max(baselines) / min(baselines) - 1
    if drift > policy["max_baseline_drift"]:
        return {"decision": "remeasure", "reason": "baseline drift", "baseline_drift": drift, "production_ready": False}
    rng = random.Random(0)
    boot = sorted(math.exp(statistics.mean(rng.choices(ratios, k=len(ratios)))) for _ in range(10000))
    lower = boot[499]
    estimate = math.exp(statistics.mean(ratios))
    decision = "screen_pass" if lower >= policy["min_speedup"] else "unresolved"
    return {"decision": decision, "geometric_mean_speedup": estimate,
            "speedup_one_sided_95_lower": lower, "pairs": len(pairs), "baseline_drift": drift,
            "production_ready": False,
            "remaining_gates": ["independent native numerical review", "matched quality noninferiority",
                                "other precision phase", "fresh confirmation", "soak and resume", "32-GPU acceptance"]}


def append_receipt(path, receipt):
    """Single-controller append-only hash chain, with OS lock and fsync.

    Hashes expose edits relative to a separately archived head; they do not
    prevent an attacker rewriting an entire local chain. No generated receipt
    belongs in Git. The supervisor must archive each new head independently.
    """
    import fcntl
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = "0" * 64
        seen = set()
        for line in handle:
            item = json.loads(line)
            expected = digest({"previous": previous, "receipt": item["receipt"]})
            if item.get("previous") != previous or item.get("sha256") != expected:
                raise ValueError("ledger chain is corrupt")
            previous = item["sha256"]
            seen.add(item["receipt"]["trial_id"])
        trial_id = receipt.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in seen:
            raise ValueError("trial_id must be nonempty and unique")
        positive(receipt.get("billed_gpu_hours"), "billed_gpu_hours", zero=True)
        if receipt.get("outcome") not in {"screen_pass", "unresolved", "reject", "crash", "timeout", "remeasure"}:
            raise ValueError("unknown trial outcome")
        item = {"previous": previous, "receipt": receipt}
        item["sha256"] = digest(item)
        handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return item
