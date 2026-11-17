"""A/B variants of the gfx942 BF16+rowwise-FNUZ producer kernels.

Candidates against ``producer_kernels._norm`` and ``producer_kernels._swiglu``.

Bottleneck hypothesis: both reference kernels pad the row to
``next_power_of_2(width)`` lanes (2304 -> 4096: 44% dead lanes; 5760 -> 8192:
30% dead) and hold several FP32 copies of the padded row plus a reshaped
(BLOCK//128, 128) view for the FP8 emit, all in registers at one CTA per row.

The v2 kernels strip-mine the row into CHUNK-wide strips (CHUNK % 128 == 0 so
each strip holds whole 128-wide scale groups). Two passes: pass 1 accumulates
the FP32 sum of squares; pass 2 reloads strips from L2, applies the norm with
the exact reference rounding chain and emits BF16 outputs plus FNUZ
codes/scales via the same ``_scale`` + ``div_rn`` + clamp sequence.

Numerics: per-element math and the FP8 emit are identical to the reference;
the 128-wide emit groups never straddle a chunk. Only the row sum-of-squares
runs in chunk order instead of one wide ``tl.sum``, so ``inverse`` (and
therefore every downstream value) can drift near FP32 epsilon (<1e-5
relative); codes can flip only on exact rounding boundaries. The probe reports
both the bit-exact flag and the max relative error.
"""
import torch
import triton
import triton.language as tl

from .fp8_kernels import FP8, _scale


@triton.jit
def _emit_strip(ROUNDED, C, S, row, col, GROUP0, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                CHUNK: tl.constexpr):
    # FNUZ emit of one CHUNK-wide strip of the BF16-rounded row: whole 128-wide
    # groups, same _scale/div_rn/clamp chain as the reference _emit.
    blocks = tl.reshape(ROUNDED.to(tl.float32), (CHUNK // 128, 128))
    scales = _scale(tl.max(tl.abs(blocks), 1), 1e-5, 240.)
    codes = tl.div_rn(blocks, scales[:, None])
    codes = tl.where((tl.abs(blocks) < float('inf')) & (tl.abs(codes) < float('inf')),
                     tl.minimum(tl.maximum(codes, -240.), 240.), float('nan'))
    tl.store(C + row * WIDTH + col, tl.reshape(codes, (CHUNK,)).to(C.dtype.element_ty), col < WIDTH)
    groups = GROUP0 + tl.arange(0, CHUNK // 128)
    tl.store(S + groups * ROWS + row, scales, groups < WIDTH // 128)


@triton.jit
def _norm_v2(X, W, Y, R, C, S, ROWS: tl.constexpr, WIDTH: tl.constexpr, GAIN: tl.constexpr,
             EPS: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    total = 0.0
    for start in tl.static_range(0, WIDTH, CHUNK):
        col = start + tl.arange(0, CHUNK)
        x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
        total += tl.sum(x * x, 0)
    inverse = tl.rsqrt(total / WIDTH + EPS)
    tl.store(R + row, inverse)
    for start in tl.static_range(0, WIDTH, CHUNK):
        col = start + tl.arange(0, CHUNK)
        mask = col < WIDTH
        x = tl.load(X + row * WIDTH + col, mask, other=0).to(tl.float32)
        y = x * inverse
        if GAIN:
            weight = tl.load(W + (col if GAIN == 2 else tl.full((CHUNK,), 0, tl.int32)), mask, other=0).to(tl.float32)
            y *= weight
        rounded = y.to(Y.dtype.element_ty)
        tl.store(Y + row * WIDTH + col, rounded, mask)
        _emit_strip(rounded, C, S, row, col, start // 128, ROWS, WIDTH, CHUNK)


@triton.jit
def _swiglu_v2(X, W, Y, R, C, S, ROWS: tl.constexpr, WIDTH: tl.constexpr, NORM: tl.constexpr,
               EPS: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    inverse = 1.0
    if NORM:
        total = 0.0
        for start in tl.static_range(0, WIDTH, CHUNK):
            col = start + tl.arange(0, CHUNK)
            mask = col < WIDTH
            g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
            u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
            sigmoid = 1. / (1. + tl.exp(-g))
            silu = (g * sigmoid).to(X.dtype.element_ty).to(tl.float32)
            product = (silu * u).to(X.dtype.element_ty).to(tl.float32)
            total += tl.sum(product * product, 0)
        inverse = tl.rsqrt(total / WIDTH + EPS)
        tl.store(R + row, inverse)
    gain = tl.load(W) if NORM else 1.0
    for start in tl.static_range(0, WIDTH, CHUNK):
        col = start + tl.arange(0, CHUNK)
        mask = col < WIDTH
        g = tl.load(X + row * (2 * WIDTH) + col, mask, other=0).to(tl.float32)
        u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, mask, other=0).to(tl.float32)
        sigmoid = 1. / (1. + tl.exp(-g))
        silu = (g * sigmoid).to(X.dtype.element_ty).to(tl.float32)
        product = (silu * u).to(X.dtype.element_ty).to(tl.float32)
        if NORM:
            product = (product * inverse) * gain
        rounded = product.to(Y.dtype.element_ty)
        tl.store(Y + row * WIDTH + col, rounded, mask)
        _emit_strip(rounded, C, S, row, col, start // 128, ROWS, WIDTH, CHUNK)


def _outputs(rows, width, device):
    if width % 128 or rows <= 0:
        raise ValueError('FP8 producers require nonempty aligned matrices')
    y = torch.empty((rows, width), device=device, dtype=torch.bfloat16)
    codes = torch.empty((rows, width), device=device, dtype=FP8)
    scales = torch.empty((width // 128, rows), device=device, dtype=torch.float32).T
    return y, codes, scales


def norm_forward_v2(values, weight, eps, *, chunk=2048, num_warps=4):
    flat = values.contiguous().view(-1, values.shape[-1])
    rows, width = flat.shape
    if chunk % 128:
        raise ValueError('chunk must hold whole 128-wide scale groups')
    y, c, s = _outputs(rows, width, flat.device)
    inverse = torch.empty(rows, device=flat.device, dtype=torch.float32)
    gain = 0 if weight is None else (1 if weight.ndim == 0 else 2)
    _norm_v2[(rows,)](flat, weight, y, inverse, c, s, rows, width, gain, eps, chunk,
                      num_warps=num_warps, enable_fp_fusion=False)
    return y, inverse, flat, (c, s)


def packed_forward_v2(packed, weight, eps, *, chunk=2048, num_warps=4):
    width = packed.shape[-1] // 2
    rows = packed.numel() // (2 * width)
    if chunk % 128:
        raise ValueError('chunk must hold whole 128-wide scale groups')
    y, c, s = _outputs(rows, width, packed.device)
    inverse = torch.empty(rows, device=packed.device, dtype=torch.float32) if weight is not None else None
    _swiglu_v2[(rows,)](packed, weight, y, inverse, c, s, rows, width, weight is not None, eps, chunk,
                        num_warps=num_warps, enable_fp_fusion=False)
    return y.reshape(*packed.shape[:-1], width), inverse, (c, s)
