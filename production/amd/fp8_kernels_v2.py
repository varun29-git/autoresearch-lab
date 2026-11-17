"""A/B variants of the gfx942 FNUZ directional quantizers.

These kernels are candidates against ``fp8_kernels._both`` / ``fp8_kernels._columns``.
They reuse ``_scale`` and ``_fnuz`` from the reference module so the rounding
chain is literally the same code: a max reduction (order-independent), one
``div_rn`` per element and the FNUZ clamp. Every variant here is expected to be
BIT-EXACT against the reference kernels for any floating input dtype; the
probe (``production/amd/kernel_v2_probe.py``) asserts that per shape.

Bottleneck hypotheses for the reference kernels:
- ``_both`` keeps the whole 128x128 tile live as FP32 (64 KB, ~64 regs/thread
  at num_warps=8) across both quotients, which spills on gfx942.
- ``_columns`` tiles the column reduction at 128x32 with num_warps=4, which
  underutilizes the wide-load path and launches 4x more CTAs than needed.
"""
import math

import torch
import triton
import triton.language as tl

from .fp8_kernels import FP8, FP8_MAX, _fnuz, _scale


@triton.jit
def _both_v2_strip(X, CR, SR, CC, SC, M: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
                   XR: tl.constexpr, XK: tl.constexpr, SR0: tl.constexpr, SR1: tl.constexpr,
                   EPS: tl.constexpr, LIMIT: tl.constexpr, CHUNK: tl.constexpr):
    """Two-pass strip-mined 128x128 tile: pass 1 walks CHUNK-wide strips.

    Column scale groups (128 rows per column) complete inside every strip, so
    column codes/scales are stored from pass 1 while the row amax accumulates.
    Pass 2 reloads the same strips (L2-resident: one tile is 32 KB of BF16) and
    emits the rowwise codes. Peak register use is one FP32 strip instead of the
    full FP32 tile. Bit-exact with ``_both``: max is order-independent and the
    same 128-row / 128-column value sets define each scale.
    """
    rb = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1).to(tl.int64)
    rows = rb * 128 + tl.arange(0, 128)
    row_max = tl.zeros((128,), tl.float32)
    for i in tl.static_range(128 // CHUNK):
        cols = cb * 128 + i * CHUNK + tl.arange(0, CHUNK)
        x = tl.load(X + rows[:, None] * XR + cols[None, :] * XK,
                    (rows[:, None] < M) & (cols[None, :] < K), other=0).to(tl.float32)
        amplitude = tl.abs(x)
        row_max = tl.maximum(row_max, tl.max(amplitude, 1))
        col_scale = _scale(tl.max(amplitude, 0), EPS, LIMIT)
        col_q = _fnuz(x, col_scale[None, :], LIMIT).to(CC.dtype.element_ty)
        # Same strided store as the reference: write all padded lanes as zeros.
        tl.store(CC + cols[None, :] * PAD + rows[:, None], col_q,
                 (rows[:, None] < PAD) & (cols[None, :] < K))
        tl.store(SC + rb * K + cols, col_scale, cols < K)
    row_scale = _scale(row_max, EPS, LIMIT)
    for i in tl.static_range(128 // CHUNK):
        cols = cb * 128 + i * CHUNK + tl.arange(0, CHUNK)
        x = tl.load(X + rows[:, None] * XR + cols[None, :] * XK,
                    (rows[:, None] < M) & (cols[None, :] < K), other=0).to(tl.float32)
        row_codes = _fnuz(x, row_scale[:, None], LIMIT).to(CR.dtype.element_ty)
        tl.store(CR + rows[:, None] * K + cols[None, :], row_codes,
                 (rows[:, None] < M) & (cols[None, :] < K))
    tl.store(SR + rows * SR0 + cb * SR1, row_scale, rows < M)


@triton.jit
def _both_v2_hold(X, CR, SR, CC, SC, M: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
                  XR: tl.constexpr, XK: tl.constexpr, SR0: tl.constexpr, SR1: tl.constexpr,
                  EPS: tl.constexpr, LIMIT: tl.constexpr):
    """Single pass; hold the tile in its LOADED dtype, convert on use.

    BF16 -> FP32 conversion is exact, so for BF16 inputs the scale math and
    quotients see bit-identical values to the reference while the live tile is
    32 KB instead of 64 KB. The column-code store goes through ``tl.trans``
    (shared-memory transpose) so both code stores are coalesced; addresses and
    values are unchanged, including zeroed padded lanes (masked load gives
    x=0 there and _fnuz(0, s)=0).
    """
    rb = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1).to(tl.int64)
    rows = rb * 128 + tl.arange(0, 128)
    cols = cb * 128 + tl.arange(0, 128)
    valid = (rows[:, None] < M) & (cols[None, :] < K)
    x = tl.load(X + rows[:, None] * XR + cols[None, :] * XK, valid, other=0)
    xf = x.to(tl.float32)
    amplitude = tl.abs(xf)
    row_scale = _scale(tl.max(amplitude, 1), EPS, LIMIT)
    col_scale = _scale(tl.max(amplitude, 0), EPS, LIMIT)
    row_codes = _fnuz(xf, row_scale[:, None], LIMIT).to(CR.dtype.element_ty)
    tl.store(CR + rows[:, None] * K + cols[None, :], row_codes, valid)
    tl.store(SR + rows * SR0 + cb * SR1, row_scale, rows < M)
    col_q = _fnuz(xf, col_scale[None, :], LIMIT).to(CC.dtype.element_ty)
    transposed = tl.trans(col_q)
    tl.store(CC + cols[:, None] * PAD + rows[None, :], transposed,
             (cols[:, None] < K) & (rows[None, :] < PAD))
    tl.store(SC + rb * K + cols, col_scale, cols < K)


@triton.jit
def _both_v2_persistent(X, CR, SR, CC, SC, M: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
                        XR: tl.constexpr, XK: tl.constexpr, SR0: tl.constexpr, SR1: tl.constexpr,
                        EPS: tl.constexpr, LIMIT: tl.constexpr, CHUNK: tl.constexpr,
                        TILES: tl.constexpr, PROGRAMS: tl.constexpr):
    """Persistent-CTA strip variant: a fixed grid loops over 128x128 tiles.

    Same per-tile math as ``_both_v2_strip`` (bit-exact); only the launch
    geometry changes, which removes per-tile CTA setup and lets the scheduler
    keep a tuned number of CTAs resident.
    """
    cb_total = K // 128
    for tile in range(tl.program_id(0), TILES, PROGRAMS):
        rb = (tile // cb_total).to(tl.int64)
        cb = (tile % cb_total).to(tl.int64)
        rows = rb * 128 + tl.arange(0, 128)
        row_max = tl.zeros((128,), tl.float32)
        for i in tl.static_range(128 // CHUNK):
            cols = cb * 128 + i * CHUNK + tl.arange(0, CHUNK)
            x = tl.load(X + rows[:, None] * XR + cols[None, :] * XK,
                        (rows[:, None] < M) & (cols[None, :] < K), other=0).to(tl.float32)
            amplitude = tl.abs(x)
            row_max = tl.maximum(row_max, tl.max(amplitude, 1))
            col_scale = _scale(tl.max(amplitude, 0), EPS, LIMIT)
            col_q = _fnuz(x, col_scale[None, :], LIMIT).to(CC.dtype.element_ty)
            tl.store(CC + cols[None, :] * PAD + rows[:, None], col_q,
                     (rows[:, None] < PAD) & (cols[None, :] < K))
            tl.store(SC + rb * K + cols, col_scale, cols < K)
        row_scale = _scale(row_max, EPS, LIMIT)
        for i in tl.static_range(128 // CHUNK):
            cols = cb * 128 + i * CHUNK + tl.arange(0, CHUNK)
            x = tl.load(X + rows[:, None] * XR + cols[None, :] * XK,
                        (rows[:, None] < M) & (cols[None, :] < K), other=0).to(tl.float32)
            row_codes = _fnuz(x, row_scale[:, None], LIMIT).to(CR.dtype.element_ty)
            tl.store(CR + rows[:, None] * K + cols[None, :], row_codes,
                     (rows[:, None] < M) & (cols[None, :] < K))
        tl.store(SR + rows * SR0 + cb * SR1, row_scale, rows < M)


@triton.jit
def _columns_v2(X, C, S, M: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
                XR: tl.constexpr, XK: tl.constexpr, EPS: tl.constexpr, LIMIT: tl.constexpr,
                BC: tl.constexpr, TRANS_STORE: tl.constexpr):
    """Columnwise quantize with a wider BC-column tile per 128-row group.

    The scale group is one column over 128 rows, complete in any aligned
    128-row block, so widening BC from 32 to 64/128 keeps the reduction sets
    identical (bit-exact) while cutting CTA count and widening loads. The tile
    is held in the loaded dtype and converted on use; ``TRANS_STORE`` selects
    an smem-transposed coalesced code store versus the reference strided store.
    Padded lanes are written as zeros either way.
    """
    block = tl.program_id(0).to(tl.int64)
    row = block * 128 + tl.arange(0, 128)
    col = tl.program_id(1).to(tl.int64) * BC + tl.arange(0, BC)
    x = tl.load(X + row[:, None] * XR + col[None, :] * XK,
                (row[:, None] < M) & (col[None, :] < K), other=0)
    xf = x.to(tl.float32)
    scale = _scale(tl.max(tl.abs(xf), 0), EPS, LIMIT)
    y = _fnuz(xf, scale[None, :], LIMIT).to(C.dtype.element_ty)
    if TRANS_STORE:
        transposed = tl.trans(y)
        tl.store(C + col[:, None] * PAD + row[None, :], transposed,
                 (col[:, None] < K) & (row[None, :] < PAD))
    else:
        tl.store(C + col[None, :] * PAD + row[:, None], y,
                 (row[:, None] < PAD) & (col[None, :] < K))
    tl.store(S + block * K + col, scale, col < K)


def _check(values, eps):
    from .runtime import require_gfx942

    require_gfx942(values.device)
    if (values.ndim != 2 or min(values.shape) <= 0 or values.shape[1] % 128
            or not math.isfinite(eps) or eps < 0):
        raise ValueError("invalid directional FP8 matrix")


def quantize_both_v2(values, *, eps=1e-5, variant="strip", chunk=32, num_warps=8, programs=0):
    """Variant launcher for the dual rowwise+columnwise FNUZ quantizer.

    variant="strip": two-pass strip-mined tile (chunk in {32, 64}).
    variant="hold": single pass, native-dtype tile, smem-transposed code store.
    variant="persistent": fixed-grid persistent CTAs running the strip body
        (``programs`` = resident CTA count).
    All variants are bit-exact with ``fp8_kernels.quantize_both``.
    """
    _check(values, eps)
    m, k = values.shape
    padded = triton.cdiv(m, 128) * 128
    row_codes = torch.empty((m, k), device=values.device, dtype=FP8)
    row_scales = torch.empty((k // 128, m), device=values.device, dtype=torch.float32).T
    col_codes = torch.empty((k, padded), device=values.device, dtype=FP8)
    col_scales = torch.empty((padded // 128, k), device=values.device, dtype=torch.float32)
    grid = (padded // 128, k // 128)
    if variant == "strip":
        if chunk <= 0 or 128 % chunk:
            raise ValueError("chunk must divide the 128-column tile")
        _both_v2_strip[grid](values, row_codes, row_scales, col_codes, col_scales, m, k, padded,
                             *values.stride(), *row_scales.stride(), eps, FP8_MAX, chunk,
                             num_warps=num_warps, enable_fp_fusion=False)
    elif variant == "hold":
        _both_v2_hold[grid](values, row_codes, row_scales, col_codes, col_scales, m, k, padded,
                            *values.stride(), *row_scales.stride(), eps, FP8_MAX,
                            num_warps=num_warps, enable_fp_fusion=False)
    elif variant == "persistent":
        if chunk <= 0 or 128 % chunk or programs <= 0:
            raise ValueError("persistent variant needs a valid chunk and program count")
        tiles = (padded // 128) * (k // 128)
        _both_v2_persistent[(programs,)](values, row_codes, row_scales, col_codes, col_scales,
                                         m, k, padded, *values.stride(), *row_scales.stride(),
                                         eps, FP8_MAX, chunk, tiles, programs,
                                         num_warps=num_warps, enable_fp_fusion=False)
    else:
        raise ValueError(f"unknown quantize_both variant {variant!r}")
    return (row_codes, row_scales), (col_codes, col_scales.T)


def quantize_columns_v2(values, *, eps=1e-5, bc=128, trans_store=True, num_warps=8):
    """Variant launcher for the columnwise-only quantizer; bit-exact with
    ``fp8_kernels.quantize_columns`` for any bc in {32, 64, 128}."""
    _check(values, eps)
    m, k = values.shape
    padded = triton.cdiv(m, 128) * 128
    codes = torch.empty((k, padded), device=values.device, dtype=FP8)
    scales = torch.empty((padded // 128, k), device=values.device, dtype=torch.float32)
    _columns_v2[(padded // 128, triton.cdiv(k, bc))](
        values, codes, scales, m, k, padded, *values.stride(), eps, FP8_MAX, bc, trans_store,
        num_warps=num_warps, enable_fp_fusion=False)
    return codes, scales.T
