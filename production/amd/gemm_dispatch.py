"""Optional per-shape backend table for the four block-FP8 projections.

An empty table keeps the configured backend. A filled entry is accepted only
when it names one of the three block-FP8 implementations and records that the
numerical check already matched. HipKittens has no hipBLASLt solution index,
so this table cannot store one.
"""
from __future__ import annotations

LEGAL_BACKENDS = ("rocm_fp8_te", "rocm_fp8", "rocm_fp8_triton", "rocm_fp8_bpreshuffle")
PROJECTIONS = {
    (3840, 2304): "qkv",
    (2304, 2304): "o",
    (11520, 2304): "gate_up",
    (2304, 5760): "down",
}
ROLES = ("forward", "dgrad", "wgrad")

_TABLE = None


def validate_table(table):
    if not isinstance(table, dict):
        raise ValueError("GEMM table must be an object")
    cleaned = {}
    for key, entry in table.items():
        projection, separator, role = str(key).partition(".")
        if separator != "." or projection not in PROJECTIONS.values() or role not in ROLES:
            raise ValueError(f"GEMM table key must be projection.role, not {key!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"GEMM table entry {key} must be an object")
        backend = entry.get("backend")
        if backend not in LEGAL_BACKENDS:
            raise ValueError(f"GEMM table entry {key} has no block-FP8 backend")
        if entry.get("numerics") != "matched":
            raise ValueError(f"GEMM table entry {key} requires numerics=matched")
        if backend == "rocm_fp8_bpreshuffle" and role == "wgrad":
            raise ValueError(f"GEMM table entry {key} selects a forward/dgrad-only kernel")
        if set(entry) - {"backend", "numerics", "splits"}:
            raise ValueError(f"GEMM table entry {key} has unknown fields")
        splits = entry.get("splits")
        if splits is not None and (role != "wgrad" or backend == "rocm_fp8_te"
                                   or type(splits) is not int or splits not in (1, 2, 4, 8, 16)):
            raise ValueError(f"GEMM table entry {key} has invalid split-K count")
        if role == "wgrad" and backend != "rocm_fp8_te" and splits is None:
            raise ValueError(f"GEMM table entry {key} needs a split-K count")
        cleaned[f"{projection}.{role}"] = {"backend": backend, "numerics": "matched"}
        if splits is not None:
            cleaned[f"{projection}.{role}"]["splits"] = splits
    return cleaned


def install_table(table):
    global _TABLE
    _TABLE = validate_table(table)
    return _TABLE


def reset_table():
    global _TABLE
    _TABLE = None


def choose_backend(default, role, weight_shape):
    """Return the configured backend, or a matched table entry for this weight."""
    if default not in LEGAL_BACKENDS and _TABLE is not None:
        raise ValueError("per-shape dispatch starts from a block-FP8 backend")
    if _TABLE is None:
        return default
    if role not in ROLES:
        raise ValueError("GEMM role must be forward, dgrad, or wgrad")
    if weight_shape is None:
        raise ValueError("a GEMM table requires the weight shape")
    projection = PROJECTIONS.get((int(weight_shape[0]), int(weight_shape[1])))
    if projection is None:
        return default
    entry = _TABLE.get(f"{projection}.{role}")
    return default if entry is None else entry["backend"]


def choose_splits(default, role, weight_shape):
    """Use an explicit split-K count when a table selects a local wgrad."""
    if _TABLE is None or role != "wgrad" or weight_shape is None:
        return default
    projection = PROJECTIONS.get((int(weight_shape[0]), int(weight_shape[1])))
    entry = _TABLE.get(f"{projection}.{role}")
    return default if entry is None else entry.get("splits", default)
