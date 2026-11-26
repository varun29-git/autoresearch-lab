"""Frozen contract for the bounded single-H100 proxy qualification campaign.

Single source of truth for the proxy geometry, the five-trial LR matrix,
fixed optimizer settings, token-aligned WSD schedules, observation cadence
and the one-time width transfer. See RUNBOOK.md sections 1-2 and
../amd/pretraining_readiness/LR_BATCH_MONITORING.md for the rationale.
Nothing here prepares data, enforces a live budget or launches training.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from production.recipe import SELECTED_MODEL, SELECTED_TRAINING

from .plan import (FULL_WIDTH, GLOBAL_BATCH_TOKENS, PROXY_WIDTH, TRIALS as _PLAN_TRIALS,
                   schedule)

REPOSITORY = Path(__file__).resolve().parents[2]

# Selected dense training semantics shared by every campaign config.
SELECTED_TRAINING_LIKE = dict(SELECTED_TRAINING)

# --- proxy geometry: reviewed 24-layer recipe narrowed to width 768 --------
# Only these four fields differ from the selected production recipe.
PROXY_MODEL_CHANGES = {"dim": 768, "ffn_dim": 1920, "n_heads": 6, "n_kv_heads": 2}
PROXY_PARAMETER_COUNT = 194_329_390
FULL_PARAMETER_COUNT = 1_446_478_126


def proxy_model_dict() -> dict:
    return {**SELECTED_MODEL, **PROXY_MODEL_CHANGES}


def full_model_dict() -> dict:
    return dict(SELECTED_MODEL)


# --- five-trial LR matrix, deterministic order ------------------------------
# H is the historical-rate anchor under the new contract, not the retained
# production-.003 hypothesis; R0 carries that hypothesis.
TRIAL_ORDER = tuple(name for name, _, _ in _PLAN_TRIALS)
TRIAL_LRS = {name: {"muon_lr": muon, "adamw_lr": adamw} for name, muon, adamw in _PLAN_TRIALS}

# Fixed in all five arms and the confirmation (RUNBOOK section 1).
FIXED_OPTIMIZER = {
    "kind": "muon",
    "muon_lr_rule": "spectral_unclamped",
    "muon_fan_in_base": None,
    "norm_lr": 0.0003,
    "qk_norm_lr": 0.0001,
    "weight_decay": 0.02,
    "embedding_weight_decay": 0.0,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "momentum": 0.95,
    # No role-specific LR overrides anywhere in the campaign.
    "muon_qkv_lr": None,
    "muon_o_lr": None,
    "muon_gate_up_lr": None,
    "muon_down_lr": None,
    "peri_norm_lr": None,
    "final_norm_lr": None,
    "value_mix_lr": None,
}


def trial_optimizer_dict(trial_id: str) -> dict:
    """Proxy optimizer for one matrix arm; muon/adamw LRs come from the matrix."""
    if trial_id not in TRIAL_LRS:
        raise ValueError(f"unknown qualification trial: {trial_id!r}")
    return {**FIXED_OPTIMIZER, **TRIAL_LRS[trial_id]}


def transferred_adamw_lr(trial_id: str) -> float:
    """The single alpha=1 width transfer, applied exactly once: x 768/2304."""
    if trial_id not in TRIAL_LRS:
        raise ValueError(f"unknown qualification trial: {trial_id!r}")
    return float(Decimal(str(TRIAL_LRS[trial_id]["adamw_lr"])) * PROXY_WIDTH / FULL_WIDTH)


def confirmation_optimizer_dict(trial_id: str) -> dict:
    """Full-size optimizer: transferred AdamW, unchanged Muon and small groups."""
    if trial_id not in TRIAL_LRS:
        raise ValueError(f"unknown qualification trial: {trial_id!r}")
    return {**FIXED_OPTIMIZER, "muon_lr": TRIAL_LRS[trial_id]["muon_lr"],
            "adamw_lr": transferred_adamw_lr(trial_id)}


# --- token-aligned complete WSD schedules (RUNBOOK section 2) ----------------
PROXY_SCHEDULE = schedule(2384, 238, 477)
CONFIRMATION_SCHEDULE = schedule(480, 48, 96)
PROXY_BOUNDARY_UPDATES = tuple(PROXY_SCHEDULE["boundary_updates"])
CONFIRMATION_BOUNDARY_UPDATES = tuple(CONFIRMATION_SCHEDULE["boundary_updates"])
OPTIONAL_BATCH_SCHEDULE = schedule(4768, 476, 954, GLOBAL_BATCH_TOKENS // 2)

# --- execution and observation cadence ---------------------------------------
WORLD_SIZE = 1
LOG_INTERVAL = 1                    # every complete update
CHECKPOINT_INTERVAL = 100           # plus explicit boundary checkpoints
VALIDATION_INTERVAL = 48            # quick fixed held-out slice (~100.66M tokens)
VALIDATION_BATCHES_PER_LANGUAGE = 8
# One frozen evaluation size for quick checks and boundaries alike, so every
# arm sees identical evaluator/examples at every cadence point.
VALIDATION_SEQUENCES_PER_LANGUAGE = 64
ROLE_DIAGNOSTIC_INTERVAL = 20       # logical-role deltas; also at boundaries
STABILITY_LOG_INTERVAL = 20         # legacy sampled probes remain a component
EMERGENCY_RESERVE_HOURS = 0.5       # checkpoint/stop time kept in the ledger
MAX_ALLOCATED_HOURS = 50.0

# Frozen selection contract, fixed before any trial result exists (MONITORING).
SELECTION_MARGINS = {
    "schema": 1,
    "status": "frozen_before_trials",
    "aggregate_superiority_nll": 0.005,
    "aggregate_noninferiority_nll": 0.002,
    "domain_noninferiority_nll": 0.01,
    "units": "token_mean_nll_on_frozen_held_out_identities",
    "notes": [
        "A tiny single-seed advantage inside the superiority margin is a tie.",
        "Document/bootstrap uncertainty addresses evaluation noise only, "
        "not training-seed variance or selection bias.",
        "The reserved final holdout is consulted once, at the declared step.",
    ],
}


def qualification_source_hashes() -> dict:
    """Trainer plus qualification-package source identity.

    terminus.integrity.source_hashes() covers terminus/*.py and
    production/amd/*.py only; the qualification package and the selected
    recipe are executable campaign dependencies and are pinned here too.
    """
    from terminus.integrity import file_sha256, source_hashes

    result = source_hashes()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        result[f"production/kimi_h100_5b/{path.name}"] = file_sha256(path)
    result["production/recipe.py"] = file_sha256(REPOSITORY / "production" / "recipe.py")
    return result
