import pytest
import torch

from production.amd import reference, te_gemm


@pytest.mark.parametrize('m,n,group', [(17,128,128), (256,128,1), (128,128,1), (17,256,128)])
def test_tile_alignment_preserves_quantized_product(monkeypatch, m, n, group):
    torch.manual_seed(29)
    lhs = reference.quantize(torch.randn(m, 256))
    rhs = reference.quantize(torch.randn(n, 256), group_rows=group)
    dtype = torch.float32 if group == 1 else torch.bfloat16
    expected = reference.gemm(lhs, rhs, rhs_group_size=group, out_dtype=dtype)
    calls = []
    def native(a, b, **kwargs):
        calls.append((a[0].shape[0], b[0].shape[0]))
        return reference.gemm(a, b, **kwargs)
    monkeypatch.setattr(te_gemm, '_gemm', native)
    actual = te_gemm.gemm(lhs, rhs, rhs_group_size=group, out_dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    assert actual.is_contiguous()
    assert calls == ([(n,m)] if group == 1 and n % 256 and m % 256 == 0
                     else [(m,((n+255)//256)*256)])
