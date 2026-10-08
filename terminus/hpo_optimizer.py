"""Named, role-aware optimizer groups for a future versioned HPO campaign.

This wraps the archived optimizer builder without changing its training path or
its update rule. Use only for fresh HPO runs: the group layout intentionally
differs from architecture-campaign optimizer checkpoints.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping

from .optim import OptimizerConfig, build_optimizer


_MATRIX_ROLES = {
    "qkv_proj": "matrix.attn_qkv",
    "q_proj": "matrix.attn_q",
    "k_proj": "matrix.attn_k",
    "v_proj": "matrix.attn_v",
    "o_proj": "matrix.attn_o",
    "gate_up_proj": "matrix.ffn_gate_up",
    "gate_proj": "matrix.ffn_gate",
    "up_proj": "matrix.ffn_up",
    "down_proj": "matrix.ffn_down",
}
_ATTN_MATRIX_MODULES = frozenset({"qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj"})
_FFN_MATRIX_MODULES = frozenset(_MATRIX_ROLES) - _ATTN_MATRIX_MODULES
_ROLES = frozenset({
    *_MATRIX_ROLES.values(), "matrix.embed_rot_in", "matrix.embed_rot_out",
    "embedding_readout", "block_norm", "peri_norm", "final_norm",
    "qk_gain", "value_residual", "sub_norm",
})


def parameter_role(name: str, *, use_muon: bool) -> str:
    """Return a stable physical-parameter role, failing on unknown architecture."""
    parts = name.split(".")
    if use_muon:
        if name in {"embed_rot_in.weight", "embed_rot_out.weight"}:
            return "matrix." + parts[0]
        if (len(parts) == 5 and parts[0] == "layers" and parts[1].isdigit()
                and parts[2] in {"attn", "ffn"} and parts[4] == "weight"):
            module = parts[3]
            if ((parts[2] == "attn" and module in _ATTN_MATRIX_MODULES)
                    or (parts[2] == "ffn" and module in _FFN_MATRIX_MODULES)):
                return _MATRIX_ROLES[module]
        raise ValueError(f"unclassified Muon parameter: {name}")

    if name == "tok_embeddings.weight":
        return "embedding_readout"
    if name == "norm.weight":
        return "final_norm"
    if len(parts) >= 4 and parts[0] == "layers" and parts[1].isdigit():
        if len(parts) == 4 and parts[3] == "weight":
            if parts[2] in {"attn_norm", "ffn_norm"}:
                return "block_norm"
            if parts[2] in {"attn_out_norm", "ffn_out_norm"}:
                return "peri_norm"
        if len(parts) == 4 and parts[2:] == ["attn", "value_lambda"]:
            return "value_residual"
        if len(parts) == 5 and parts[4] == "weight":
            if parts[2] == "attn" and parts[3] in {"q_norm", "k_norm"}:
                return "qk_gain"
            if (parts[2], parts[3]) in {("attn", "attn_sub_norm"), ("ffn", "ffn_sub_norm")}:
                return "sub_norm"
    raise ValueError(f"unclassified AdamW parameter: {name}")


def build_hpo_optimizer(model, config: OptimizerConfig, *, role_lrs: Mapping[str, float] | None = None):
    """Split and label groups while preserving the archived optimizer's defaults.

    `role_lrs` contains Muon *base* LRs for matrix roles and direct AdamW LRs
    for auxiliary roles. It changes LR only; the role's weight-decay setting
    remains fixed and the resulting per-step shrinkage must be logged.
    """
    if config.kind != "muon":
        raise ValueError("HPO role groups currently require the pinned Muon optimizer")
    proposed = dict(role_lrs or {})
    for role, lr in proposed.items():
        if role not in _ROLES:
            raise ValueError(f"unknown HPO optimizer role: {role}")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"{role} LR must be finite and positive")

    optimizer = build_optimizer(model, config)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    new_groups = []
    present_roles = set()
    for original in optimizer.param_groups:
        buckets = OrderedDict()
        for parameter in original["params"]:
            name = names[id(parameter)]
            role = parameter_role(name, use_muon=original["use_muon"])
            bucket = buckets.setdefault(role, {"params": [], "param_names": []})
            bucket["params"].append(parameter)
            bucket["param_names"].append(name)
        for role, bucket in buckets.items():
            group = dict(original)
            group.update(bucket)
            group["role"] = role
            group["name"] = (f"{role}:{bucket['param_names'][0]}" if original["use_muon"] else role)
            if role in proposed:
                group["base_lr"] = (proposed[role] * original["base_lr"] / config.muon_lr
                                    if original["use_muon"] else proposed[role])
                group["lr"] = group["base_lr"]
            present_roles.add(role)
            new_groups.append(group)

    absent = proposed.keys() - present_roles
    if absent:
        raise ValueError(f"HPO optimizer role absent from model: {sorted(absent)}")
    original_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    new_ids = [id(parameter) for group in new_groups for parameter in group["params"]]
    if len(new_ids) != len(set(new_ids)) or set(new_ids) != set(original_ids):
        raise ValueError("HPO optimizer groups must cover each trainable parameter exactly once")
    optimizer.param_groups = new_groups
    return optimizer
