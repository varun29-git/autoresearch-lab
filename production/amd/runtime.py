"""ROCm identity, inventory and verification; never starts paid hardware."""
import argparse
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from production.artifacts import external_path
from terminus.integrity import file_sha256, fingerprint, source_hashes, write_json_atomic


@lru_cache(maxsize=16)
def require_gfx942(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("this backend requires a ROCm PyTorch build and a gfx942 GPU")
    properties = torch.cuda.get_device_properties(device)
    if getattr(properties, "gcnArchName", "").split(":")[0] != "gfx942":
        raise RuntimeError("AMD CI-1 currently targets gfx942 (MI325X/MI300X), not OCP gfx950")
    return properties


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


@lru_cache(maxsize=16)
def hip_contract(device):
    properties = require_gfx942(device)
    tuning = {}
    # These upstream switches do not have an AITER_/NVTE_ prefix. Omitting
    # them lets a new process change CK to Triton with an unchanged receipt.
    dispatch_switches = {"ENABLE_CK", "FLASH_ATTENTION_TRITON_AMD_ENABLE",
                         "ROCBLAS_USE_HIPBLASLT", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"}
    for key, value in sorted(os.environ.items()):
        if key in dispatch_switches or key.startswith(("AITER_", "NVTE_", "PYTORCH_TUNABLEOP_", "TORCH_BLAS_", "HIPBLASLT_")):
            # Cache locations may differ by node/rank; bind actual tuning content.
            if key.endswith(("_FILE", "_FILENAME")) and Path(value).is_file():
                tuning[key] = {"sha256": file_sha256(value)}
            elif not key.endswith(("_DIR", "_PATH")):
                tuning[key] = value
    if os.environ.get("CI1_AMD_GEMM_TUNING"):
        tuning["CI1_AMD_GEMM_TUNING"] = file_sha256(external_path(os.environ["CI1_AMD_GEMM_TUNING"]))
    return {"hip": str(torch.version.hip), "arch": properties.gcnArchName,
            "rccl": list(torch.cuda.nccl.version()) if isinstance(torch.cuda.nccl.version(), tuple) else torch.cuda.nccl.version(),
            "image": os.environ.get("CI1_ROCM_IMAGE"), "tuning": tuning,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp8_format": "float8_e4m3fnuz", "fp8_max": torch.finfo(torch.float8_e4m3fnuz).max}


def capture_runtime(device="cuda", *, include_inventory=True):
    from terminus.train import runtime_contract

    properties = require_gfx942(torch.device(device))
    identity = {"runtime": runtime_contract(torch.device(device)), "source": source_hashes(),
                "python": platform.python_version(),
                "packages": dict(sorted((d.metadata["Name"].lower(), d.version)
                                         for d in importlib.metadata.distributions() if d.metadata.get("Name"))),
                "hipcc": _command(["hipcc", "--version"]),
                "driver": Path("/sys/module/amdgpu/version").read_text().strip()
                          if Path("/sys/module/amdgpu/version").exists() else platform.release()}
    inventory = {"amd_smi": _command(["amd-smi", "static", "--json"]),
                 "topology": _command(["amd-smi", "topology", "--json"]),
                 "gpu_name": properties.name, "gpu_memory_bytes": properties.total_memory,
                 "visible_devices": torch.cuda.device_count()} if include_inventory else None
    return {"schema": 1, "kind": "ci1_amd_runtime", "identity": identity,
            "identity_sha256": fingerprint(identity), "inventory": inventory,
            "scope": "Identity and inventory only; no throughput, numerical or quality certification"}


def verify(report, current):
    if (report.get("schema") != 1 or report.get("kind") != "ci1_amd_runtime"
            or report.get("identity_sha256") != fingerprint(report.get("identity"))
            or report.get("identity") != current["identity"]):
        raise ValueError("AMD runtime receipt is stale or incompatible")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--output")
    g.add_argument("--verify")
    args = p.parse_args()
    report = capture_runtime()
    if args.verify:
        verify(json.loads(external_path(args.verify).read_text()), report)
    else:
        write_json_atomic(external_path(args.output, fresh=True), report)
    print(json.dumps({"status": "identity_verified", "identity_sha256": report["identity_sha256"]}))


if __name__ == "__main__":
    main()
