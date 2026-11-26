"""A/B variants of the fused RMSNorm and flat SwiGLU pointwise kernels.

Candidates against ``pointwise_kernels._norm_forward``, ``_norm_backward`` and
``_swiglu``.

Bottleneck hypotheses:
- ``_norm_forward`` pads width 2304 to a 4096-wide block (44% dead lanes) and
  runs one CTA per row with default num_warps.
- ``_norm_backward`` serializes rows over 512 CTAs with the same 4096-wide
  padding inside the loop.
- ``_swiglu`` indexes a flat range with an integer div/mod per element and a
  1024-element block.

The v2 norm kernels load the row as a main power-of-2 block plus one masked
power-of-2 tail block (2304 = 2048 + 256: zero dead lanes in the main block),
holding both in registers so no reload is needed. The v2 swiglu uses a 2D
(row, column-block) grid: no div/mod and no flat-index masking.

Numerics: per-element BF16 roundings and FP32 products are identical. Row
reductions keep FP32 accumulation but the summation order differs from one
wide ``tl.sum`` (and dw partial counts change with PROGRAMS), so expect drift
bounded near FP32 epsilon (<1e-5 relative); the probe measures it. The flat
swiglu variant is elementwise and therefore bit-exact. The CUDA-only FP8
sidecar (EMIT_FP8, e4m3) is not implemented here.
"""
import torch
import triton
import triton.language as tl


def _split_width(width):
    """Largest pow2 main block plus a pow2 tail block covering width."""
    main = triton.next_power_of_2(width) // 2
    tail = triton.next_power_of_2(width - main) if width > main else 0
    return main, tail


@triton.jit
def _norm_forward_v2(X, W, Y, R, ROWS, WIDTH: tl.constexpr, GAIN: tl.constexpr, EPS: tl.constexpr,
                     BM: tl.constexpr, BT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    colm = tl.arange(0, BM)
    mm = colm < WIDTH
    x0 = tl.load(X + row * WIDTH + colm, mm, other=0).to(tl.float32)
    total = tl.sum(x0 * x0, 0)
    if BT > 0:
        colt = BM + tl.arange(0, BT)
        mt = colt < WIDTH
        x1 = tl.load(X + row * WIDTH + colt, mt, other=0).to(tl.float32)
        total += tl.sum(x1 * x1, 0)
    inverse = tl.rsqrt(total / WIDTH + EPS)
    n0 = x0 * inverse
    if GAIN:
        w0 = tl.load(W + (colm if GAIN == 2 else tl.full((BM,), 0, tl.int32)), mm, other=0).to(tl.float32)
        n0 *= w0
    tl.store(Y + row * WIDTH + colm, n0.to(Y.dtype.element_ty), mm)
    if BT > 0:
        n1 = x1 * inverse
        if GAIN:
            w1 = tl.load(W + (colt if GAIN == 2 else tl.full((BT,), 0, tl.int32)), mt, other=0).to(tl.float32)
            n1 *= w1
        tl.store(Y + row * WIDTH + colt, n1.to(Y.dtype.element_ty), mt)
    tl.store(R + row, inverse)


@triton.jit
def _norm_backward_v2(DY, X, W, R, DX, DW, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                      GAIN: tl.constexpr, PROGRAMS: tl.constexpr,
                      BM: tl.constexpr, BT: tl.constexpr):
    program = tl.program_id(0)
    colm = tl.arange(0, BM)
    mm = colm < WIDTH
    colt = BM + tl.arange(0, BT)
    mt = colt < WIDTH
    if GAIN:
        w0 = tl.load(W + (colm if GAIN == 2 else tl.full((BM,), 0, tl.int32)), mm, other=0).to(tl.float32)
        if BT > 0:
            w1 = tl.load(W + (colt if GAIN == 2 else tl.full((BT,), 0, tl.int32)), mt, other=0).to(tl.float32)
    dw0 = tl.zeros((BM,), tl.float32)
    dw1 = tl.zeros((BT,), tl.float32)
    for row in range(program, ROWS, PROGRAMS):
        x0 = tl.load(X + row * WIDTH + colm, mm, other=0).to(tl.float32)
        dy0 = tl.load(DY + row * WIDTH + colm, mm, other=0).to(tl.float32)
        if BT > 0:
            x1 = tl.load(X + row * WIDTH + colt, mt, other=0).to(tl.float32)
            dy1 = tl.load(DY + row * WIDTH + colt, mt, other=0).to(tl.float32)
        inverse = tl.load(R + row)
        n0 = x0 * inverse
        if BT > 0:
            n1 = x1 * inverse
        if GAIN:
            # rms_norm backward: dw gathers pre-gain dy; the dx dot uses post-gain dy.
            dw0 += dy0 * n0
            dy0 *= w0
            if BT > 0:
                dw1 += dy1 * n1
                dy1 *= w1
        dot = tl.sum(dy0 * n0, 0)
        if BT > 0:
            dot += tl.sum(dy1 * n1, 0)
        dx0 = (dy0 - n0 * (dot / WIDTH)) * inverse
        tl.store(DX + row * WIDTH + colm, dx0, mm)
        if BT > 0:
            dx1 = (dy1 - n1 * (dot / WIDTH)) * inverse
            tl.store(DX + row * WIDTH + colt, dx1, mt)
    if GAIN == 2:
        tl.store(DW + program * WIDTH + colm, dw0, mm)
        if BT > 0:
            tl.store(DW + program * WIDTH + colt, dw1, mt)
    elif GAIN == 1:
        tl.store(DW + program, tl.sum(dw0, 0) + tl.sum(dw1, 0))


def norm_forward_v2(values, weight, eps, *, num_warps=4):
    flat = values.contiguous().view(-1, values.shape[-1])
    rows, width = flat.shape
    output = torch.empty_like(flat)
    inverse = torch.empty(rows, device=values.device, dtype=torch.float32)
    gain = 0 if weight is None else (1 if weight.ndim == 0 else 2)
    bm, bt = _split_width(width)
    _norm_forward_v2[(rows,)](flat, weight, output, inverse, rows, width, gain, eps, bm, bt,
                              num_warps=num_warps, enable_fp_fusion=False)
    return output, inverse, flat, None


def norm_backward_v2(gradient, values, weight, inverse, *, programs=2048, num_warps=4):
    rows, width = values.shape
    gain = 0 if weight is None else (1 if weight.ndim == 0 else 2)
    programs = min(rows, programs)
    dx = torch.empty_like(values)
    partial = torch.empty((programs, width if gain == 2 else 1), device=values.device, dtype=torch.float32) if gain else None
    bm, bt = _split_width(width)
    _norm_backward_v2[(programs,)](gradient.contiguous(), values, weight, inverse, dx, partial,
                                   rows, width, gain, programs, bm, bt,
                                   num_warps=num_warps, enable_fp_fusion=False)
    dw = partial.sum(0).reshape(weight.shape).to(weight.dtype) if gain else None
    return dx, dw


@triton.jit
def _swiglu_v2(G, U, Y, DY, DG, DU, WIDTH: tl.constexpr,
               G_STRIDE: tl.constexpr, U_STRIDE: tl.constexpr,
               BACKWARD: tl.constexpr, BC: tl.constexpr):
    # 2D (row, column-block) grid: no per-element div/mod, coalesced rows.
    # Rows stay on axis 0: grid axes 1/2 cap at 65535 and ROWS is 65536.
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BC + tl.arange(0, BC)
    mask = col < WIDTH
    gate = tl.load(G + row * G_STRIDE + col, mask, other=0).to(tl.float32)
    up = tl.load(U + row * U_STRIDE + col, mask, other=0).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-gate))
    silu = gate * sigmoid
    # Match eager's intermediate BF16 SiLU rounding.
    rounded = silu.to(U.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        dy = tl.load(DY + row * WIDTH + col, mask, other=0).to(tl.float32)
        d_silu = (dy * up).to(U.dtype.element_ty).to(tl.float32)
        tl.store(DG + row * WIDTH + col, d_silu * sigmoid * (1. + gate * (1. - sigmoid)), mask)
        tl.store(DU + row * WIDTH + col, dy * rounded, mask)
    else:
        tl.store(Y + row * WIDTH + col, rounded * up, mask)


def swiglu_forward_v2(gate, up, *, bc=512, num_warps=4):
    output = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    rows, width = gate.reshape(-1, gate.shape[-1]).shape
    _swiglu_v2[(rows, triton.cdiv(width, bc))](gate, up, output, None, None, None,
        width, gate.stride(-2), up.stride(-2), False, bc,
        num_warps=num_warps, enable_fp_fusion=False)
    return output


def swiglu_backward_v2(gradient, gate, up, *, bc=512, num_warps=4):
    dg = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    du = torch.empty_like(dg)
    rows, width = gate.reshape(-1, gate.shape[-1]).shape
    _swiglu_v2[(rows, triton.cdiv(width, bc))](gate, up, None, gradient.contiguous(), dg, du,
        width, gate.stride(-2), up.stride(-2), True, bc,
        num_warps=num_warps, enable_fp_fusion=False)
    return dg, du
