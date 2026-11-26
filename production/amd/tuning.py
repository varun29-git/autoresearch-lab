"""Validated, external, shape-specific gfx942 GEMM launch choices."""
from functools import lru_cache
import json
import os

from production.artifacts import external_path


def validate_tile(tile):
    required = {"bm", "bn", "num_warps", "num_stages", "waves_per_eu"}
    if not isinstance(tile, dict) or set(tile) != required or any(type(v) is not int for v in tile.values()):
        raise ValueError("GEMM tile needs explicit integer launch fields")
    if (tile["bm"] not in (16, 32, 64) or tile["bn"] not in (32, 64, 128)
            or tile["num_warps"] not in (4, 8) or tile["num_stages"] not in (1, 2, 3)
            or tile["waves_per_eu"] not in (0, 1, 2, 3, 4)):
        raise ValueError("unsupported gfx942 GEMM tile")
    return tile


@lru_cache(maxsize=1)
def table():
    path = os.environ.get("CI1_AMD_GEMM_TUNING")
    if not path:
        return {}
    payload = json.loads(external_path(path).read_text())
    if payload.get("schema") not in (1, 2) or payload.get("arch") != "gfx942" or not isinstance(payload.get("tiles"), dict):
        raise ValueError("invalid AMD GEMM tuning table")
    if payload["schema"] == 2:
        validate_measured_table(payload)
    return {k: validate_tile(v) for k, v in payload["tiles"].items()}


@lru_cache(maxsize=1)
def measured_splits():
    path = os.environ.get("CI1_AMD_GEMM_TUNING")
    if not path:
        return {}
    payload = json.loads(external_path(path).read_text())
    return ({k:v["splits"] for k,v in payload["measurements"].items()}
            if payload.get("schema") == 2 else {})


LEGAL_SPLITS = (1, 2, 4, 8, 16)


def resolve_splits(requested, recorded, *, group, k):
    """Use a measured per-shape split when a schema-2 table is loaded.

    With no table, this is the historical ``min(requested, K/128)`` rule.
    A loaded table replaces that request for the matching shape. Forward and
    input-gradient rows are rejected unless they were measured at one split.
    """
    if type(requested) is not int or requested not in LEGAL_SPLITS:
        raise ValueError("AMD wgrad split count must be 1, 2, 4, 8 or 16")
    if group not in (1, 128) or type(k) is not int or k <= 0 or k % 128:
        raise ValueError("split-K requires group 1 or 128 and a positive K multiple of 128")
    if recorded is None:
        return min(requested, k // 128)
    if type(recorded) is not int or recorded not in LEGAL_SPLITS:
        raise ValueError("measured split-K count must be 1, 2, 4, 8 or 16")
    if group != 1 and recorded != 1:
        raise ValueError("measured forward or dgrad split-K must stay 1")
    if recorded > k // 128:
        raise ValueError("measured split-K does not fit this reduction length")
    return recorded


def tile_for(m, n, k, group, *, splits=1):
    selected = table()
    expected = measured_splits().get(f"{m},{n},{k},{group}")
    if expected is not None and expected != splits:
        raise ValueError("measured tile used a different split-K count; remeasure this configuration")
    return selected.get(f"{m},{n},{k},{group}",
                       {"bm": 32, "bn": 64, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0})


def native_identity():
    import torch
    import triton
    from pathlib import Path
    from terminus.integrity import file_sha256
    from .runtime import require_gfx942
    gpu = require_gfx942(torch.device("cuda", torch.cuda.current_device()))
    root = Path(__file__).parent
    return {"torch": str(torch.__version__), "hip": str(torch.version.hip), "triton": str(triton.__version__),
            "arch": gpu.gcnArchName, "gpu": gpu.name, "image": os.environ.get("CI1_ROCM_IMAGE"),
            "sources": {name: file_sha256(root / name) for name in ("fp8_kernels.py", "tuning.py", "autotune.py")}}


def validate_measured_table(payload, *, identity=None):
    from terminus.integrity import fingerprint
    if payload.get("identity") != (native_identity() if identity is None else identity):
        raise ValueError("measured tuning table belongs to another runtime/source")
    if payload.get("sha256") != fingerprint({k:v for k,v in payload.items() if k != "sha256"}):
        raise ValueError("measured tuning table checksum changed")
    measurements = payload.get("measurements", {})
    for key, tile in payload["tiles"].items():
        validate_tile(tile)
        winner = measurements.get(key, {})
        if winner.get("tile") != tile or winner.get("correctness_passed") is not True:
            raise ValueError("tuning tile has no matching native correctness measurement")
        splits = winner.get("splits")
        if type(splits) is not int or splits not in LEGAL_SPLITS:
            raise ValueError("measured split-K count must be 1, 2, 4, 8 or 16")
        group = int(str(key).rsplit(",", 1)[-1])
        if group != 1 and splits != 1:
            raise ValueError("measured forward or dgrad split-K must stay 1")
    return payload
