"""Review a same-source/toolchain native rebuild without hiding binary drift.

Only FA3/DeepGEMM shared-object hashes may differ. All Python/header identities,
native versions, GPU identity, compiler identities and driver must match. The
receipt becomes usable for a checkpoint fork only after native preflights and
the full-size checkpoint recovery gate pass. This is a reviewed runtime
transition, not a claim that the rebuilt binaries are byte-identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from terminus.integrity import fingerprint, write_json_atomic


def binary_changes(parent: dict, child: dict) -> list[str]:
    old, new = dict(parent), dict(child)
    old_native, new_native = old.pop("native_sources"), new.pop("native_sources")
    if old != new or set(old_native) != set(new_native):
        raise ValueError("native rebuild changed runtime/version/device identity")
    changes = []
    for module in old_native:
        before, after = old_native[module], new_native[module]
        if before == after:
            continue
        if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(after):
            raise ValueError("native rebuild changed source file inventory")
        for name in before:
            if before[name] == after[name]:
                continue
            if module not in {"deep_gemm", "flash_attn_3", "flash_attn_3_cuda"} or not name.endswith(".so"):
                raise ValueError(f"native rebuild changed source: {module}/{name}")
            changes.append(f"{module}/{name}")
    return sorted(changes)


def review(parent_identity: dict, child_identity: dict) -> dict:
    parent, child = parent_identity["identity"], child_identity["identity"]
    for name in ("compilers", "driver_versions"):
        if parent[name] != child[name]:
            raise ValueError(f"native rebuild changed {name}")
    for name in ("torch", "triton", "deep_gemm", "flash_attn_3", "muon-optimizer", "numpy", "sentencepiece"):
        if parent["installed_packages"].get(name) != child["installed_packages"].get(name):
            raise ValueError(f"native rebuild changed package: {name}")
    changes = binary_changes(parent["runtime"], child["runtime"])
    result = {"schema": 1, "policy": "same_source_same_toolchain_binary_rebuild",
              "parent_runtime_sha256": fingerprint(parent["runtime"]),
              "child_runtime_sha256": fingerprint(child["runtime"]),
              "changed_binaries": changes, "compilers": child["compilers"],
              "driver_versions": child["driver_versions"]}
    return {**result, "receipt_sha256": fingerprint(result)}


def verify(receipt: dict, parent: dict, child: dict, *, gate_root: Path) -> dict:
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (receipt.get("schema") != 1 or receipt.get("policy") != "same_source_same_toolchain_binary_rebuild"
            or fingerprint(body) != receipt.get("receipt_sha256")
            or receipt.get("parent_runtime_sha256") != fingerprint(parent)
            or receipt.get("child_runtime_sha256") != fingerprint(child)
            or receipt.get("changed_binaries") != binary_changes(parent, child)):
        raise ValueError("native rebuild receipt does not bind both runtime identities")
    # The trainer independently verifies the native preflight report content.
    # Here require both native shapes and the actual >5GB remote recovery gate.
    for name in ("preflight_proxy.json", "preflight_full.json"):
        gate = json.loads((gate_root / name).read_text())
        if gate.get("status") != "passed" or gate.get("identity", {}).get("runtime") != child:
            raise ValueError(f"native rebuild missing successful gate: {name}")
    fit = json.loads((gate_root / "full_size_fit.json").read_text())
    if fit.get("status") != "passed_native_fit_only" or fit.get("valid_updates", 0) < 5 or not fit.get("full_checkpoint_restore"):
        raise ValueError("native rebuild missing full-size recovery gate")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-identity", required=True)
    parser.add_argument("--child-identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = review(json.loads(Path(args.parent_identity).read_text()),
                     json.loads(Path(args.child_identity).read_text()))
    write_json_atomic(Path(args.output), receipt)
    print(json.dumps({"event": "native_rebuild_reviewed", **receipt}), flush=True)


if __name__ == "__main__":
    main()
