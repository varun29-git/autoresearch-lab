"""Capture/verify runtime identity, including the actual DeepGEMM JIT compiler."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import torch

from production.artifacts import external_path
from terminus.integrity import file_sha256, fingerprint, source_hashes, write_json_atomic
from terminus.train import runtime_contract


def compiler_identity(path):
    path = Path(path).resolve()
    result = subprocess.run([str(path), "--version"], check=True, capture_output=True, text=True, timeout=30)
    return {"sha256": file_sha256(path), "version": result.stdout.strip()}


def capture_runtime(*, device="cuda"):
    runtime = runtime_contract(torch.device(device))
    compilers, driver = {}, None
    if device == "cuda":
        if runtime["capability"] != [9, 0] or "H100" not in runtime["gpu"]:
            raise ValueError("production runtime requires an H100 SM90 GPU")
        for module in ("deep_gemm", "triton", "muon", "flash_attn_interface"):
            if not runtime["native_sources"].get(module):
                raise ValueError(f"missing native runtime dependency: {module}")
        if not (runtime["native_sources"].get("flash_attn_3_cuda") or runtime["native_sources"].get("flash_attn_3")):
            raise ValueError("missing native FA3 binary")
        # Require explicit compiler paths instead of guessing from torch CUDA.
        jit = os.environ.get("DG_JIT_NVCC_COMPILER")
        fa_home = os.environ.get("FA3_CUDA_HOME")
        if not jit or not fa_home:
            raise ValueError("set DG_JIT_NVCC_COMPILER and FA3_CUDA_HOME explicitly")
        compilers = {"deepgemm_jit": compiler_identity(jit), "fa3_build": compiler_identity(Path(fa_home) / "bin/nvcc")}
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=30)
        driver = sorted(set(result.stdout.splitlines()))
    packages = dict(sorted((d.metadata["Name"].lower(), d.version) for d in importlib.metadata.distributions()
                           if d.metadata.get("Name")))
    identity = {"runtime": runtime, "compilers": compilers, "driver_versions": driver,
                "installed_packages": packages, "source": source_hashes()}
    return {"schema": 1, "scope": "runtime identity only; native preflight and training quality remain separate",
            "identity": identity, "identity_sha256": fingerprint(identity)}


def verify_runtime(report, current):
    if report.get("schema") != 1 or report.get("identity_sha256") != fingerprint(report.get("identity")):
        raise ValueError("invalid runtime receipt")
    if current["identity_sha256"] != report["identity_sha256"]:
        changed = [key for key in current["identity"] if current["identity"][key] != report["identity"].get(key)]
        raise ValueError("runtime changed: " + ", ".join(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--verify")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = capture_runtime(device=args.device)
    if args.verify:
        verify_runtime(json.loads(Path(args.verify).read_text()), result)
    else:
        path = external_path(args.output, fresh=True)
        write_json_atomic(path, result)
    print(json.dumps({"identity_sha256": result["identity_sha256"], "status": "identity_verified"}))


if __name__ == "__main__":
    main()
