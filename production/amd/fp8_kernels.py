"""gfx942 FNUZ quantization and full FP8 GEMMs, including FP32 1Dx1D wgrad.

Scales are applied after each 128-wide reduction tile. gfx942 has no native
scaled MFMA: moving these scales outside the K loop changes the computation.
No atomics or BF16 intermediate wgrad. Split-K has an explicit FP32 reduction.
"""
import math

import torch
import triton
import triton.language as tl
if torch.version.hip:
    from triton.language.extra.hip import libdevice
else:
    from triton.language.extra.cuda import libdevice

FP8 = torch.float8_e4m3fnuz
FP8_MAX = torch.finfo(FP8).max


@triton.jit
def _scale(amplitude, EPS: tl.constexpr, LIMIT: tl.constexpr):
    amplitude = tl.maximum(amplitude, EPS)
    return tl.where(amplitude == 0, 1., tl.maximum(amplitude * (1. / LIMIT), 1.1754943508222875e-38))


@triton.jit
def _quantize(X, C, S, M: tl.constexpr, K: tl.constexpr, XR: tl.constexpr, XK: tl.constexpr,
              SR: tl.constexpr, SK: tl.constexpr, GROUP: tl.constexpr, RB: tl.constexpr,
              EPS: tl.constexpr, LIMIT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * RB + tl.arange(0, RB)
    group = tl.program_id(1).to(tl.int64)
    col = group * 128 + tl.arange(0, 128)
    x = tl.load(X + row[:, None] * XR + col[None, :] * XK,
                (row[:, None] < M) & (col[None, :] < K), other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    if GROUP == 128:
        scale = _scale(tl.max(amax, 0), EPS, LIMIT)
        y = tl.div_rn(x, scale)
        tl.store(S + tl.program_id(0) * SR + group * SK, scale)
    else:
        scale = _scale(amax, EPS, LIMIT)
        y = tl.div_rn(x, scale[:, None])
        tl.store(S + row * SR + group * SK, scale, row < M)
    finite = (tl.abs(x) < float("inf")) & (tl.abs(y) < float("inf"))
    y = tl.where(finite, tl.minimum(tl.maximum(y, -LIMIT), LIMIT), float("nan"))
    tl.store(C + row[:, None] * K + col[None, :], y.to(C.dtype.element_ty),
             (row[:, None] < M) & (col[None, :] < K))


@triton.jit
def _quantize_ternary(W, C, S, K: tl.constexpr, WR: tl.constexpr, WK: tl.constexpr):
    # One 128x128 CTA reads the FP32 master once, computes the H100-compatible
    # mean-absolute scale and emits FNUZ codes directly. No intermediate
    # ternary or BF16 tensor is materialized in HBM.
    rb = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1).to(tl.int64)
    row = rb * 128 + tl.arange(0, 128)
    col = cb * 128 + tl.arange(0, 128)
    block = tl.load(W + row[:, None] * WR + col[None, :] * WK).to(tl.float32)
    scale = tl.maximum(tl.sum(tl.sum(tl.abs(block), 1), 0) * (1.0 / 16384.0), 1e-6)
    code = libdevice.nearbyint(tl.div_rn(block, scale))
    code = tl.minimum(tl.maximum(code, -1.0), 1.0)
    finite = (tl.abs(block) < float("inf")) & (tl.abs(scale) < float("inf"))
    code = tl.where(finite, code, float("nan")).to(C.dtype.element_ty)
    tl.store(C + row[:, None] * K + col[None, :], code)
    tl.store(S + rb * (K // 128) + cb, scale)


@triton.jit
def _columns(X, C, S, M: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
             XR: tl.constexpr, XK: tl.constexpr, EPS: tl.constexpr, LIMIT: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    row = block * 128 + tl.arange(0, 128)
    col = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32)
    x = tl.load(X + row[:, None] * XR + col[None, :] * XK,
                (row[:, None] < M) & (col[None, :] < K), other=0).to(tl.float32)
    scale = _scale(tl.max(tl.abs(x), 0), EPS, LIMIT)
    y = tl.div_rn(x, scale[None, :])
    y = tl.where((tl.abs(x) < float("inf")) & (tl.abs(y) < float("inf")),
                 tl.minimum(tl.maximum(y, -LIMIT), LIMIT), float("nan"))
    # Write ALL padded lanes. Uninitialized tail codes poison wgrad.
    tl.store(C + col[None, :] * PAD + row[:, None], y.to(C.dtype.element_ty),
             (row[:, None] < PAD) & (col[None, :] < K))
    tl.store(S + block * K + col, scale, col < K)


def quantize(values, *, group_rows=1, eps=0.0):
    from .runtime import require_gfx942

    require_gfx942(values.device)
    if (values.ndim != 2 or min(values.shape) <= 0 or values.shape[1] % 128
            or group_rows not in (1, 128) or (group_rows == 128 and values.shape[0] % 128)
            or not values.is_floating_point() or not math.isfinite(eps) or eps < 0):
        raise ValueError("invalid AMD FP8 quantization shape/group/epsilon")
    m, k = values.shape
    codes = torch.empty((m, k), device=values.device, dtype=FP8)
    if group_rows == 1:
        # This layout passes directly to AITER without a scale transpose copy.
        scales = torch.empty((k // 128, m), device=values.device, dtype=torch.float32).T
    else:
        scales = torch.empty((m // 128, k // 128), device=values.device, dtype=torch.float32)
    rb = 128 if group_rows == 128 else 16
    _quantize[(triton.cdiv(m, rb), k // 128)](
        values, codes, scales, m, k, *values.stride(), *scales.stride(), group_rows, rb, eps, FP8_MAX,
        num_warps=8 if group_rows == 128 else 4)
    return codes, scales


@torch.no_grad()
def quantize_ternary(weight):
    from .runtime import require_gfx942

    require_gfx942(weight.device)
    if (weight.ndim != 2 or min(weight.shape) <= 0 or weight.dtype != torch.float32
            or weight.shape[0] % 128 or weight.shape[1] % 128):
        raise ValueError("AMD ternary FP8 needs an aligned FP32 master matrix")
    rows, width = weight.shape
    codes = torch.empty((rows, width), device=weight.device, dtype=FP8)
    scales = torch.empty((rows // 128, width // 128), device=weight.device, dtype=torch.float32)
    _quantize_ternary[(rows // 128, width // 128)](
        weight, codes, scales, width, *weight.stride(),
        num_warps=8, enable_fp_fusion=False)
    return codes, scales


def quantize_directions(values, *, rowwise=True, transposed=False, eps=1e-5):
    from .runtime import require_gfx942

    require_gfx942(values.device)
    if (values.ndim != 2 or min(values.shape) <= 0 or values.shape[1] % 128
            or not (rowwise or transposed) or not math.isfinite(eps) or eps < 0):
        raise ValueError("invalid directional FP8 matrix")
    primary = quantize(values, eps=eps) if rowwise else None
    secondary = None
    if transposed:
        m, k = values.shape
        padded = triton.cdiv(m, 128) * 128
        codes = torch.empty((k, padded), device=values.device, dtype=FP8)
        scales = torch.empty((padded // 128, k), device=values.device, dtype=torch.float32)
        _columns[(padded // 128, triton.cdiv(k, 32))](
            values, codes, scales, m, k, padded, *values.stride(), eps, FP8_MAX, num_warps=4)
        secondary = codes, scales.T
    return primary, secondary


@triton.jit
def _gemm(A, B, AS, BS, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          AS0: tl.constexpr, AS1: tl.constexpr, BS0: tl.constexpr, BS1: tl.constexpr,
          GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          SPLITS: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr):
    # Group M tiles to improve reuse of the weight tile in the 256 MB LLC.
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group_id = pid // (8 * nn)
    first_m = group_id * 8
    group_m = tl.minimum(nm - first_m, 8)
    pm = first_m + (pid % (8 * nn)) % group_m
    pn = (pid % (8 * nn)) // group_m
    mi = pm.to(tl.int64) * BM + tl.arange(0, BM)
    ni = pn.to(tl.int64) * BN + tl.arange(0, BN)
    ki = tl.arange(0, 128)
    part = tl.program_id(1)
    acc = tl.full((BM, BN), 0., tl.float32)
    for block in range(part * BLOCKS_PER_SPLIT, tl.minimum((part + 1) * BLOCKS_PER_SPLIT, K // 128)):
        kk = block * 128 + ki
        a = tl.load(A + mi[:, None] * K + kk[None, :], mi[:, None] < M, other=0)
        b = tl.load(B + ni[None, :] * K + kk[:, None], ni[None, :] < N, other=0)
        sa = tl.load(AS + mi * AS0 + block * AS1, mi < M, other=0)
        sb = tl.load(BS + (ni // GROUP) * BS0 + block * BS1, ni < N, other=0)
        dot = tl.dot(a, b, out_dtype=tl.float32)
        acc += dot * (sa[:, None] * sb[None, :])
    tl.store(OUT + part.to(tl.int64) * M * N + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _sum_splits(PARTS, OUT, SIZE: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0., tl.float32)
    for part in range(SPLITS):
        total += tl.load(PARTS + part * SIZE + offsets, offsets < SIZE, other=0)
    tl.store(OUT + offsets, total, offsets < SIZE)


def check_operands(lhs, rhs, group, dtype):
    from .runtime import require_gfx942

    a, sa = lhs
    b, sb = rhs
    require_gfx942(a.device)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(*a.shape, *b.shape) <= 0
            or a.shape[1] % 128 or group not in (1, 128) or b.shape[0] % group
            or a.dtype != FP8 or b.dtype != FP8 or not a.is_contiguous() or not b.is_contiguous()
            or sa.shape != (a.shape[0], a.shape[1] // 128)
            or sb.shape != (b.shape[0] // group, b.shape[1] // 128)
            or sa.dtype != torch.float32 or sb.dtype != torch.float32
            or dtype not in (torch.bfloat16, torch.float32)
            or len({x.device for x in (a, b, sa, sb)}) != 1):
        raise ValueError("invalid FNUZ GEMM operands/scales/output dtype")


def gemm(lhs, rhs, *, rhs_group_size=128, out_dtype=torch.bfloat16, splits=1):
    check_operands(lhs, rhs, rhs_group_size, out_dtype)
    a, sa = lhs
    b, sb = rhs
    m, k = a.shape
    n = b.shape[0]
    from .tuning import measured_splits, resolve_splits, tile_for

    recorded = measured_splits().get(f"{m},{n},{k},{rhs_group_size}")
    splits = resolve_splits(splits, recorded, group=rhs_group_size, k=k)
    output = torch.empty((m, n), device=a.device, dtype=out_dtype)
    work = torch.empty((splits, m, n), device=a.device, dtype=torch.float32) if splits > 1 else output
    tile = tile_for(m, n, k, rhs_group_size, splits=splits)
    bm, bn = tile["bm"], tile["bn"]
    _gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
        a, b, sa, sb, work, m, n, k, *sa.stride(), *sb.stride(), rhs_group_size,
        bm, bn, splits, triton.cdiv(k // 128, splits), num_warps=tile["num_warps"],
        num_stages=tile["num_stages"], waves_per_eu=tile["waves_per_eu"], matrix_instr_nonkdim=16)
    if splits > 1:
        _sum_splits[(triton.cdiv(m * n, 1024),)](work, output, m * n, splits, 1024)
    return output
