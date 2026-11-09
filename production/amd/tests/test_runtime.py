"""Native dispatch choices must invalidate receipts across fresh processes."""
from types import SimpleNamespace

import pytest
import torch

from production.amd.runtime import hip_contract
from terminus.integrity import fingerprint


@pytest.mark.parametrize("switch,value", [
    ("ENABLE_CK", "0"),
    ("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE"),
    ("ROCBLAS_USE_HIPBLASLT", "1"),
    ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"),
])
def test_dispatch_environment_invalidates_native_identity(monkeypatch, switch, value):
    monkeypatch.setattr("production.amd.runtime.require_gfx942",
                        lambda _device: SimpleNamespace(gcnArchName="gfx942"))
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 20, 0))
    monkeypatch.delenv(switch, raising=False)
    hip_contract.cache_clear()
    try:
        before = hip_contract(torch.device("cuda", 0))
        monkeypatch.setenv(switch, value)
        hip_contract.cache_clear()  # Model another process, not live retuning.
        after = hip_contract(torch.device("cuda", 0))
        assert fingerprint(before) != fingerprint(after)
        assert after["tuning"][switch] == value
    finally:
        hip_contract.cache_clear()
