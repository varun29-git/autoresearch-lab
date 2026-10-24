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
    if payload.get("schema") != 1 or payload.get("arch") != "gfx942" or not isinstance(payload.get("tiles"), dict):
        raise ValueError("invalid AMD GEMM tuning table")
    return {k: validate_tile(v) for k, v in payload["tiles"].items()}


def tile_for(m, n, k, group):
    return table().get(f"{m},{n},{k},{group}",
                       {"bm": 32, "bn": 64, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0})
