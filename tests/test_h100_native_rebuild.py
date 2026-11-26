import copy
import json

import pytest

from production.h100_24h.native_rebuild import binary_changes, review, verify


def identities():
    runtime = {"torch": "2.8.0", "gpu": "H100", "native_sources": {
        "deep_gemm": {"_C.so": "old", "kernel.hpp": "same"},
        "flash_attn_3": {"_C.abi3.so": "old"}, "muon": {"muon.py": "same"}}}
    parent = {"identity": {"runtime": runtime, "compilers": {"nvcc": "same"},
                           "driver_versions": ["same"], "installed_packages": {"torch": "2.8.0"}}}
    child = copy.deepcopy(parent)
    child["identity"]["runtime"]["native_sources"]["deep_gemm"]["_C.so"] = "new"
    return parent, child


def test_binary_rebuild_requires_bound_receipt_and_native_recovery_gates(tmp_path):
    parent, child = identities()
    receipt = review(parent, child)
    for name in ("preflight_proxy.json", "preflight_full.json"):
        (tmp_path / name).write_text(json.dumps({"status": "passed", "identity": {
            "runtime": child["identity"]["runtime"]}}))
    fit = tmp_path / "full_size_fit.json"
    fit.write_text(json.dumps({"status": "passed_native_fit_only", "valid_updates": 5,
                              "full_checkpoint_restore": {"sha256": "verified"}}))
    assert verify(receipt, parent["identity"]["runtime"], child["identity"]["runtime"],
                  gate_root=tmp_path) == receipt
    corrupt = {**receipt, "parent_runtime_sha256": "forged"}
    with pytest.raises(ValueError, match="bind both"):
        verify(corrupt, parent["identity"]["runtime"], child["identity"]["runtime"], gate_root=tmp_path)
    fit.write_text(json.dumps({"status": "failed", "valid_updates": 4}))
    with pytest.raises(ValueError, match="recovery gate"):
        verify(receipt, parent["identity"]["runtime"], child["identity"]["runtime"], gate_root=tmp_path)


@pytest.mark.parametrize("change", ["header", "muon", "compiler", "package", "gpu"])
def test_rebuild_rejects_semantic_or_toolchain_drift(change):
    parent, child = identities()
    identity = child["identity"]
    if change == "header":
        identity["runtime"]["native_sources"]["deep_gemm"]["kernel.hpp"] = "bad"
    elif change == "muon":
        identity["runtime"]["native_sources"]["muon"]["muon.py"] = "bad"
    elif change == "compiler":
        identity["compilers"]["nvcc"] = "bad"
    elif change == "package":
        identity["installed_packages"]["torch"] = "bad"
    else:
        identity["runtime"]["gpu"] = "different"
    with pytest.raises(ValueError):
        review(parent, child)
