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
    return payload
