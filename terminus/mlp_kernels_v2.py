"""A/B variant of the packed SwiGLU(+SubLN) kernel.

Candidate against ``mlp_kernels._packed_swiglu``.

Bottleneck hypothesis: the reference pads each row to
``next_power_of_2(width)`` lanes — width 5760 becomes an 8192-wide FP32 block
(~30% dead lanes) with several live FP32 tensors (gate, up, dy, products) at
num_warps=8, which spills registers on gfx942.

The v2 kernel strip-mines the row into CHUNK-wide strips. Normalized paths do
two passes: pass 1 accumulates the row reductions (sum of squares forward;
dw-row and the post-gain dot backward), pass 2 reloads the strips from L2 and
emits gradients/outputs. Register pressure drops to one strip per tensor.

Numerics: all per-element BF16 roundings and FP32 products are identical to
the reference. Row reductions keep FP32 accumulation but run in chunk order
instead of one wide ``tl.sum``, so expect drift bounded near FP32 epsilon
(<1e-5 relative); the probe measures it. The CUDA-only FP8 sidecar
(EMIT_FP8, e4m3) is not implemented here — the AMD FP8 producers live in
``production/amd/producer_kernels_v2.py``.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_math(g, u, dtype: tl.constexpr):
    sigmoid = 1. / (1. + tl.exp(-g))
    silu = (g * sigmoid).to(dtype).to(tl.float32)
    product = (silu * u).to(dtype).to(tl.float32)
    return sigmoid, silu, product


@triton.jit
def _packed_swiglu_v2(X, W, R, Y, DY, DX, DW, WIDTH: tl.constexpr, EPS: tl.constexpr,
                      NORM: tl.constexpr, BACKWARD: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    if BACKWARD:
        if NORM:
            inverse = tl.load(R + row)
            gain = tl.load(W)
            dw_row = 0.0
            dot = 0.0
            for start in tl.static_range(0, WIDTH, CHUNK):
                col = start + tl.arange(0, CHUNK)
                mask = col < WIDTH
                g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
                u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
                dy = tl.load(DY + row * WIDTH + col, mask, other=0).to(tl.float32)
                _, _, product = _swiglu_math(g, u, X.dtype.element_ty)
                normalized = product * inverse
                dw_row += tl.sum(dy * normalized, 0)
                dot += tl.sum((dy * gain) * normalized, 0)
            tl.store(DW + row, dw_row)
            for start in tl.static_range(0, WIDTH, CHUNK):
                col = start + tl.arange(0, CHUNK)
                mask = col < WIDTH
                g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
                u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
                dy = tl.load(DY + row * WIDTH + col, mask, other=0).to(tl.float32)
                sigmoid, silu, product = _swiglu_math(g, u, X.dtype.element_ty)
                normalized = product * inverse
                dproduct = ((dy * gain) - normalized * (dot / WIDTH)) * inverse
                dy = dproduct.to(X.dtype.element_ty).to(tl.float32)
                dsilu = (dy * u).to(X.dtype.element_ty).to(tl.float32)
                dg = dsilu * sigmoid * (1. + g * (1. - sigmoid))
                du = dy * silu
                tl.store(DX + row * (2 * WIDTH) + col, dg, mask)
                tl.store(DX + row * (2 * WIDTH) + WIDTH + col, du, mask)
        else:
            for start in tl.static_range(0, WIDTH, CHUNK):
                col = start + tl.arange(0, CHUNK)
                mask = col < WIDTH
                g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
                u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
                dy = tl.load(DY + row * WIDTH + col, mask, other=0).to(tl.float32)
                sigmoid, silu, _ = _swiglu_math(g, u, X.dtype.element_ty)
                dsilu = (dy * u).to(X.dtype.element_ty).to(tl.float32)
                dg = dsilu * sigmoid * (1. + g * (1. - sigmoid))
                du = dy * silu
                tl.store(DX + row * (2 * WIDTH) + col, dg, mask)
                tl.store(DX + row * (2 * WIDTH) + WIDTH + col, du, mask)
    else:
        if NORM:
            total = 0.0
            for start in tl.static_range(0, WIDTH, CHUNK):
                col = start + tl.arange(0, CHUNK)
                mask = col < WIDTH
                g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
                u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
                _, _, product = _swiglu_math(g, u, X.dtype.element_ty)
                total += tl.sum(product * product, 0)
            inverse = tl.rsqrt(total / WIDTH + EPS)
            tl.store(R + row, inverse)
            gain = tl.load(W)
            for start in tl.static_range(0, WIDTH, CHUNK):
                col = start + tl.arange(0, CHUNK)
                mask = col < WIDTH
                g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
                u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
                _, _, product = _swiglu_math(g, u, X.dtype.element_ty)
                product = (product * inverse) * gain
                tl.store(Y + row * WIDTH + col, product.to(Y.dtype.element_ty), mask)
        else:
            for start in tl.static_range(0, WIDTH, CHUNK):
                col = start + tl.arange(0, CHUNK)
                mask = col < WIDTH
                g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
                u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
                _, _, product = _swiglu_math(g, u, X.dtype.element_ty)
                tl.store(Y + row * WIDTH + col, product.to(Y.dtype.element_ty), mask)


def packed_forward_v2(packed, weight, eps, *, chunk=1024, num_warps=4):
    width = packed.shape[-1] // 2
    rows = packed.numel() // (2 * width)
    output = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
    inverse = torch.empty(rows, device=packed.device, dtype=torch.float32) if weight is not None else None
    _packed_swiglu_v2[(rows,)](packed, weight, inverse, output, None, None, None,
                               width, eps, weight is not None, False, chunk,
                               num_warps=num_warps, enable_fp_fusion=False)
    return output, inverse, None


def packed_backward_v2(gradient, packed, weight, inverse, *, chunk=1024, num_warps=4):
    width = packed.shape[-1] // 2
    rows = packed.numel() // (2 * width)
    dx = torch.empty_like(packed)
    partial = torch.empty(rows, device=packed.device, dtype=torch.float32) if weight is not None else None
    _packed_swiglu_v2[(rows,)](packed, weight, inverse, None, gradient.contiguous(), dx, partial,
                               width, 0., weight is not None, True, chunk,
                               num_warps=num_warps, enable_fp_fusion=False)
    return dx, partial.sum().reshape_as(weight) if weight is not None else None
