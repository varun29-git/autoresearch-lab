"""A/B Triton gather kernels replacing the hot aten embedding/index paths.

Candidates against ``F.embedding`` (``indexFuncLargeIndex``) at
``terminus/model.py:611`` and the packed-layout rotary row gathers
``cos[0, 0][positions]`` / ``sin[0, 0][positions]`` at ``model.py:616-617``.

Bottleneck hypothesis: the ROCm embedding kernel is latency-bound gathering
2304-wide rows through int64 indices with poor vectorization, and the two
rotary gathers launch separate small index kernels per microstep.

``_embed_gather`` reads int64 token ids and writes BF16 rows (casting from an
FP32 or BF16 table) through wide vectorized column blocks; variants cover
one-row-per-CTA column tiles and multi-row 2D tiles. ``_rope_gather`` fuses
the cos and sin row gathers into one launch of multi-row tiles.

Numerics: pure row copies plus a round-to-nearest-even FP32->BF16 cast, the
same conversion torch's ``.to(torch.bfloat16)`` performs — bit-exact against
the aten reference for every variant.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _embed_gather(IDS, W, Y, N, D: tl.constexpr, BC: tl.constexpr):
    # One table row per CTA row-block; BC-wide vectorized column tile.
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BC + tl.arange(0, BC)
    token = tl.load(IDS + row)
    mask = col < D
    values = tl.load(W + token.to(tl.int64) * D + col, mask, other=0.)
    tl.store(Y + row * D + col, values.to(Y.dtype.element_ty), mask)


@triton.jit
def _embed_gather_rows(IDS, W, Y, N, D: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    # BR gathered rows per CTA against a BC-wide column tile.
    rows = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    col = tl.program_id(1).to(tl.int64) * BC + tl.arange(0, BC)
    tokens = tl.load(IDS + rows, rows < N, other=0)
    mask = (rows[:, None] < N) & (col[None, :] < D)
    values = tl.load(W + tokens.to(tl.int64)[:, None] * D + col[None, :], mask, other=0.)
    tl.store(Y + rows[:, None] * D + col[None, :], values.to(Y.dtype.element_ty), mask)


@triton.jit
def _rope_gather(POS, Cos, Sin, CO, SO, N, HALF: tl.constexpr, BR: tl.constexpr):
    # Fused cos+sin row gather for packed document layouts.
    rows = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    c = tl.arange(0, HALF)
    positions = tl.load(POS + rows, rows < N, other=0)
    mask = rows[:, None] < N
    co = tl.load(Cos + positions.to(tl.int64)[:, None] * HALF + c[None, :], mask, other=0.)
    si = tl.load(Sin + positions.to(tl.int64)[:, None] * HALF + c[None, :], mask, other=0.)
    tl.store(CO + rows[:, None] * HALF + c[None, :], co, mask)
    tl.store(SO + rows[:, None] * HALF + c[None, :], si, mask)


def embed_gather_v2(ids, weight, out_dtype=torch.bfloat16, *, rows_per_cta=1, bc=512, num_warps=4):
    """weight[ids] cast to out_dtype; ids any shape of int64, weight [V, D]."""
    flat = ids.reshape(-1)
    n = flat.numel()
    width = weight.shape[1]
    out = torch.empty((n, width), device=weight.device, dtype=out_dtype)
    if rows_per_cta == 1:
        _embed_gather[(n, triton.cdiv(width, bc))](flat, weight, out, n, width, bc,
                                                   num_warps=num_warps, enable_fp_fusion=False)
    else:
        _embed_gather_rows[(triton.cdiv(n, rows_per_cta), triton.cdiv(width, bc))](
            flat, weight, out, n, width, rows_per_cta, bc, num_warps=num_warps, enable_fp_fusion=False)
    return out.view(*ids.shape, width)


def rope_gather_v2(positions, cos, sin, *, rows_per_cta=16, num_warps=2):
    """cos[0, 0][positions] and sin[0, 0][positions] in one launch.

    cos/sin: [1, 1, L, HALF] contiguous; positions: int64 broadcastable flat.
    Returns two [*, HALF] tensors matching the unsqueezed-then-contiguous
    layout model.py builds.
    """
    half = cos.shape[-1]
    flat = positions.reshape(-1)
    n = flat.numel()
    cos_flat = cos.reshape(-1, half)
    sin_flat = sin.reshape(-1, half)
    co = torch.empty((n, half), device=cos.device, dtype=cos.dtype)
    so = torch.empty((n, half), device=sin.device, dtype=sin.dtype)
    _rope_gather[(triton.cdiv(n, rows_per_cta),)](flat, cos_flat, sin_flat, co, so, n, half,
                                                  rows_per_cta, num_warps=num_warps,
                                                  enable_fp_fusion=False)
    return co.view(*positions.shape, half), so.view(*positions.shape, half)
