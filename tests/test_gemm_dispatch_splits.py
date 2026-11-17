import pytest

from production.amd.gemm_dispatch import (
    choose_backend, choose_splits, install_table, reset_table,
)


def test_mixed_te_and_local_wgrad_requires_explicit_splits():
    try:
        with pytest.raises(ValueError, match="split-K count"):
            install_table({"down.wgrad": {"backend": "rocm_fp8", "numerics": "matched"}})
        install_table({"down.wgrad": {"backend": "rocm_fp8", "numerics": "matched", "splits": 4}})
        assert choose_backend("rocm_fp8_te", "wgrad", (2304, 5760)) == "rocm_fp8"
        assert choose_splits(0, "wgrad", (2304, 5760)) == 4
        assert choose_splits(0, "forward", (2304, 5760)) == 0
        assert choose_splits(0, "wgrad", (3840, 2304)) == 0
        with pytest.raises(ValueError, match="invalid split-K count"):
            install_table({"down.wgrad": {"backend": "rocm_fp8", "numerics": "matched", "splits": 0}})
    finally:
        reset_table()
