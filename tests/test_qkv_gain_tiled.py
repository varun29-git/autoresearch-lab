"""Native tiled gain reduction versus the independent rotary/gain equations."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native Triton GPU required")
@pytest.mark.parametrize("q,kv,dim", [(18, 6, 128), (4, 1, 64), (6, 2, 256)])
@pytest.mark.parametrize("batched,broadcast", [(False, False), (True, True)])
def test_tiled_gain_strides_tail_and_rounding(q, kv, dim, batched, broadcast):
    from terminus.qkv_kernels import _tiled_gain_backward

    torch.manual_seed(1337)
    batch, length = 2, 67
    packed = torch.randn(batch, length, (q + 2 * kv) * dim,
                         device="cuda", dtype=torch.bfloat16)
    cos, sin = [torch.randn(batch if batched else 1, 1, length, dim // 2,
                            device="cuda") for _ in range(2)]
    inverse = torch.rand(batch * length, q + kv, device="cuda")
    gradients, expected = [], []
    for heads, start in ((q, 0), (kv, q)):
        gradient = torch.randn(batch, 1 if broadcast else heads, length, dim,
                               device="cuda", dtype=torch.bfloat16).expand(batch, heads, length, dim)
        gradients.append(gradient)
        da, db = gradient.chunk(2, -1)
        co, si = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        na, nb = da * co + db * si, db * co - da * si
        values = packed.view(batch, length, q + 2 * kv, dim)[:, :, start:start + heads].transpose(1, 2)
        scale = inverse.view(batch, length, q + kv)[:, :, start:start + heads].transpose(1, 2)
        normalized = values.float() * scale[..., None]
        expected.append((torch.cat((na, nb), -1).float() * normalized).sum((0, 1, 2)))
    actual = _tiled_gain_backward(*gradients, packed, cos, sin, inverse, q, kv)
    for result, reference in zip(actual, expected):
        error = (result - reference).square().mean().sqrt()
        assert error <= 2e-6 * reference.square().mean().sqrt() + 1e-5
