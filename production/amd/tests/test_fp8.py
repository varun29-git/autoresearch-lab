import pytest
import torch

from production.amd import backend, reference
from terminus.model import BitLinear

torch.set_num_threads(1)


@pytest.fixture
def emulated(monkeypatch):
    def weights(w, transposed=False, weight_mode="dense"):
        p = (reference.quantize_ternary(w) if weight_mode == "ternary"
             else reference.quantize(w, group_rows=128))
        return p, (p[0].T.contiguous(), p[1].T.contiguous()) if transposed else None

    monkeypatch.setattr(backend, "quantize_directions", reference.directional)
    monkeypatch.setattr(backend, "quantize_weight", weights)
    monkeypatch.setattr(backend, "gemm", reference.gemm)


def test_fnuz_scale_format_zero_blocks_and_nonfinite():
    x = torch.zeros(128, 256)
    x[5, 3] = 2.4
    codes, scales = reference.quantize(x, group_rows=128)
    assert codes.dtype == torch.float8_e4m3fnuz
    assert scales[0, 0] == pytest.approx(.01)
    assert codes[5, 3].float() == 240
    assert scales[0, 1] == 1
    for v in (float("nan"), float("inf"), -float("inf")):
        x[0, 0] = v
        assert torch.isnan(reference.quantize(x, group_rows=128)[0][0, 0].float())


def test_ternary_codes_are_fnuz_and_scales_are_mean_absolute():
    w = torch.ones(128, 256)
    w[:, :128] *= -1
    w[0, 0] = -.1
    w[:, 128:] = 0
    codes, scales = reference.quantize_ternary(w)
    assert codes.dtype == torch.float8_e4m3fnuz
    assert set(codes.float().unique().tolist()) == {-1., 0.}
    assert codes[0, 0].float() == 0
    assert scales[0, 0] == pytest.approx((128 * 128 - 1 + .1) / (128 * 128))
    assert scales[0, 1] == pytest.approx(1e-6)
    ties = torch.full((128, 128), 2.)
    ties[0, 0], ties[0, 1] = 1., 3.
    tie_codes, tie_scales = reference.quantize_ternary(ties)
    assert tie_scales[0, 0] == 2 and tie_codes[0, 0].float() == 0
    for special in (float("nan"), float("inf"), -float("inf")):
        w[0, 0] = special
        q, _ = reference.quantize_ternary(w)
        assert torch.isnan(q[0, 0].float())


@pytest.mark.parametrize("weight_mode", ["dense", "ternary"])
@pytest.mark.parametrize("rows", [1, 17, 129])
@pytest.mark.parametrize("gradient_scale", [1., 2. ** -24])
def test_full_backward_uses_independently_quantized_transposes(emulated, rows, gradient_scale, weight_mode):
    torch.manual_seed(123)
    x = torch.randn(rows, 256, dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(384, 256) * .03).requires_grad_()
    g = torch.randn(rows, 384, dtype=torch.bfloat16) * gradient_scale
    y = backend.linear(x, w, weight_mode=weight_mode)
    dx, dw = torch.autograd.grad(y, (x, w), g)
    xr, xt = reference.directional(x, transposed=True)
    wr = reference.quantize_ternary(w) if weight_mode == "ternary" else reference.quantize(w, group_rows=128)
    gr, gt = reference.directional(g, transposed=True, eps=0.)
    torch.testing.assert_close(y, reference.gemm(xr, wr), rtol=0, atol=0)
    torch.testing.assert_close(dx, reference.gemm(gr, (wr[0].T.contiguous(), wr[1].T.contiguous())), rtol=0, atol=0)
    torch.testing.assert_close(dw, reference.gemm(gt, xt, rhs_group_size=1, out_dtype=torch.float32), rtol=0, atol=0)
    assert dw.dtype == torch.float32
    if weight_mode == "ternary":
        # Identity STE updates latent weights even where the forward code is 0.
        assert torch.count_nonzero(dw[wr[0].float() == 0])
    assert not xt[0].float()[:, rows:].count_nonzero()
    # dW uses a quantization of X.T, not X's already-rounded row codes.
    assert reference.dequantize(xt).shape == (256, ((rows + 127) // 128) * 128)


def test_numerical_gate_rejects_zeroed_small_gradients():
    from production.amd.preflight import _errors

    expected = torch.full((16, 128), 1e-8)
    with pytest.raises(AssertionError, match="numerical mismatch"):
        _errors([torch.zeros_like(expected)], [expected], atol=0.)
    assert _errors([expected.clone()], [expected], atol=0.)[0]["relative"] == 0
    with pytest.raises(AssertionError, match="shapes"):
        _errors([expected[:1]], [expected], atol=0.)


@pytest.mark.parametrize("weight_mode", ["dense", "ternary"])
def test_accumulation_and_cache_refresh(emulated, weight_mode):
    layer = BitLinear(256, 256, weight_mode=weight_mode, linear_backend="rocm_fp8",
                      weight_scale_layout="blockwise", quantize_activations=True,
                      fp8_backward="full", fp8_quantizer="triton_split", fp8_cache_weights=True)
    x = torch.randn(17, 256, dtype=torch.bfloat16, requires_grad=True)
    first = layer(x)
    first.backward(torch.ones_like(first))
    g1 = layer.weight.grad.clone()
    second = layer(x)
    second.backward(torch.ones_like(second))
    torch.testing.assert_close(layer.weight.grad, 2 * g1)
    with torch.no_grad():
        layer.weight.mul_(2.)
    changed = layer(x)
    layer.clear_fp8_weight_cache()
    torch.testing.assert_close(changed, layer(x), rtol=0, atol=0)
    assert not torch.equal(first, changed)
    with torch.inference_mode():
        assert torch.isfinite(layer(x)).all()


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="native gfx942 GPU required")
def test_native_quantizer_gemm_and_full_backward():
    from production.amd.preflight import fp8_gate
    from production.amd.recipe import model_for
    from dataclasses import replace

    for mode in ("dense", "ternary"):
        small = replace(model_for("fp8_triton", weight_mode=mode),
                        dim=256, n_heads=4, n_kv_heads=2, ffn_dim=640)
        result = fp8_gate(small, torch.device("cuda"))
        assert result["fp32_wgrad"] and result["weight_mode"] == mode
