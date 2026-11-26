"""A/B variants of the packed QKV RoPE and learned-gain backward kernels.

Candidates against ``qkv_kernels._packed_rope`` and ``_packed_gain_tiled``.

Bottleneck hypotheses for the reference kernels:
- ``_packed_rope`` runs one CTA per (row, 4-head group): a 4x64 half-head tile
  is only 256 values, so the launch is latency-bound on small masked loads and
  per-CTA cos/sin fetches at 65536 rows x 6 groups.
- ``_packed_gain_tiled`` reduces 64 row/head pairs per program; at production
  batch the Q pass alone is ~18k programs of strided 64-wide loads.

The v2 rope kernel flattens (row, head) pairs into one 1D grid so each program
handles PAIRS pairs (default 8) against the full half-head width. The v2 gain
kernel is the reference reduction with tunable TILE/num_warps.

Numerics: every per-element BF16 rounding and FP32 product is identical to the
reference. The per-head sum-of-squares / dot reductions keep FP32 accumulation
but may run in a different order (different block shape), so expect drift
bounded near FP32 epsilon (<1e-5 relative); the probe measures it.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _packed_rope_v2(X, Cos, Sin, R, Y, DQ, DK, DV, DX, GQ, GK,
                    Q: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                    QB: tl.constexpr, QH: tl.constexpr, QL: tl.constexpr,
                    KB: tl.constexpr, KH: tl.constexpr, KL: tl.constexpr,
                    VB: tl.constexpr, VH: tl.constexpr, VL: tl.constexpr,
                    ROWS: tl.constexpr, EPS: tl.constexpr, BACKWARD: tl.constexpr,
                    PAIRS: tl.constexpr, NORMALIZE: tl.constexpr, GAIN: tl.constexpr,
                    BATCHED_ROPE: tl.constexpr):
    # Flat (row, head) pairs: forward covers Q+KV heads, backward Q+2*KV.
    NH: tl.constexpr = Q + 2 * KV if BACKWARD else Q + KV
    pair = tl.program_id(0).to(tl.int64) * PAIRS + tl.arange(0, PAIRS)
    row = pair // NH
    h = pair % NH
    valid = row < ROWS
    c = tl.arange(0, D // 2)
    pos, batch = row % L, row // L
    offset = row[:, None] * ((Q + 2 * KV) * D) + h[:, None] * D + c[None, :]
    qk = valid[:, None] & (h[:, None] < Q + KV)
    a = tl.load(X + offset, qk, other=0.).to(tl.float32)
    b = tl.load(X + offset + D // 2, qk, other=0.).to(tl.float32)
    rope_row = row if BATCHED_ROPE else pos
    co = tl.load(Cos + rope_row[:, None] * (D // 2) + c[None, :], valid[:, None], other=0.).to(X.dtype.element_ty).to(tl.float32)
    si = tl.load(Sin + rope_row[:, None] * (D // 2) + c[None, :], valid[:, None], other=0.).to(X.dtype.element_ty).to(tl.float32)
    if GAIN:
        is_q = h[:, None] < Q
        ga = tl.where(is_q, tl.load(GQ + c)[None, :], tl.load(GK + c)[None, :]).to(tl.float32)
        gb = tl.where(is_q, tl.load(GQ + D // 2 + c)[None, :], tl.load(GK + D // 2 + c)[None, :]).to(tl.float32)
    if BACKWARD:
        qo = batch[:, None] * QB + h[:, None] * QH + pos[:, None] * QL + c[None, :]
        ko = batch[:, None] * KB + (h[:, None] - Q) * KH + pos[:, None] * KL + c[None, :]
        vo = batch[:, None] * VB + (h[:, None] - Q - KV) * VH + pos[:, None] * VL + c[None, :]
        is_qh = valid[:, None] & (h[:, None] < Q)
        is_kh = valid[:, None] & (h[:, None] >= Q) & (h[:, None] < Q + KV)
        da = tl.load(DQ + qo, is_qh, other=0.).to(tl.float32)
        db = tl.load(DQ + qo + D // 2, is_qh, other=0.).to(tl.float32)
        da += tl.load(DK + ko, is_kh, other=0.).to(tl.float32)
        db += tl.load(DK + ko + D // 2, is_kh, other=0.).to(tl.float32)
        na = ((da * co).to(X.dtype.element_ty).to(tl.float32)
              + (db * si).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        nb = ((db * co).to(X.dtype.element_ty).to(tl.float32)
              - (da * si).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        if NORMALIZE:
            inverse = tl.load(R + row * (Q + KV) + h, valid & (h < Q + KV), other=0.)
            ax, bx = a * inverse[:, None], b * inverse[:, None]
            if GAIN:
                # rms_norm backward: dw gathers pre-gain dy; dx uses post-gain dy.
                na = na * ga
                nb = nb * gb
            dot = (tl.sum(na * ax, 1) + tl.sum(nb * bx, 1)) / D
            dx_a = (na - ax * dot[:, None]) * inverse[:, None]
            dx_b = (nb - bx * dot[:, None]) * inverse[:, None]
        else:
            dx_a, dx_b = na, nb
        is_vh = valid[:, None] & (h[:, None] >= Q + KV) & (h[:, None] < Q + 2 * KV)
        va = tl.load(DV + vo, is_vh, other=0.).to(tl.float32)
        vb = tl.load(DV + vo + D // 2, is_vh, other=0.).to(tl.float32)
        tl.store(DX + offset, tl.where(h[:, None] < Q + KV, dx_a, va), valid[:, None])
        tl.store(DX + offset + D // 2, tl.where(h[:, None] < Q + KV, dx_b, vb), valid[:, None])
    else:
        if NORMALIZE:
            inverse = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1)) / D + EPS)
            if GAIN:
                a = (a * inverse[:, None] * ga).to(X.dtype.element_ty).to(tl.float32)
                b = (b * inverse[:, None] * gb).to(X.dtype.element_ty).to(tl.float32)
            else:
                a = (a * inverse[:, None]).to(X.dtype.element_ty).to(tl.float32)
                b = (b * inverse[:, None]).to(X.dtype.element_ty).to(tl.float32)
        ya = (a * co).to(X.dtype.element_ty).to(tl.float32) - (b * si).to(X.dtype.element_ty).to(tl.float32)
        yb = (b * co).to(X.dtype.element_ty).to(tl.float32) + (a * si).to(X.dtype.element_ty).to(tl.float32)
        out = row[:, None] * ((Q + KV) * D) + h[:, None] * D + c[None, :]
        tl.store(Y + out, ya, valid[:, None])
        tl.store(Y + out + D // 2, yb, valid[:, None])
        if NORMALIZE:
            tl.store(R + row * (Q + KV) + h, inverse, valid)


@triton.jit
def _packed_gain_tiled_v2(X, Cos, Sin, R, G, DW,
                          Q: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                          GB: tl.constexpr, GH: tl.constexpr, GL: tl.constexpr,
                          ROWS: tl.constexpr, HEADS: tl.constexpr, START: tl.constexpr,
                          BATCHED_ROPE: tl.constexpr, TILE: tl.constexpr):
    """Same reduction as ``_packed_gain_tiled`` with TILE/num_warps left to the
    launcher. Recomputes the rotary inverse of the incoming gradients and
    accumulates dy (pre-gain) against the ungained normalized activations.
    Only the order of the deterministic FP32 sum changes versus the reference.
    """
    index = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    row, head = index // HEADS, index % HEADS
    position = row % L
    col = tl.arange(0, D // 2)
    valid = row[:, None] < ROWS
    offset = row[:, None] * ((Q + 2 * KV) * D) + (head[:, None] + START) * D + col[None, :]
    a = tl.load(X + offset, valid, other=0.).to(tl.float32)
    b = tl.load(X + offset + D // 2, valid, other=0.).to(tl.float32)
    inverse = tl.load(R + row * (Q + KV) + head + START, row < ROWS, other=0.)
    rope_row = row if BATCHED_ROPE else position
    rope_offset = rope_row[:, None] * (D // 2) + col[None, :]
    co = tl.load(Cos + rope_offset, valid, other=0.).to(X.dtype.element_ty).to(tl.float32)
    si = tl.load(Sin + rope_offset, valid, other=0.).to(X.dtype.element_ty).to(tl.float32)
    go = (row[:, None] // L) * GB + head[:, None] * GH + position[:, None] * GL + col[None, :]
    da = tl.load(G + go, valid, other=0.).to(tl.float32)
    db = tl.load(G + go + D // 2, valid, other=0.).to(tl.float32)
    na = ((da * co).to(X.dtype.element_ty).to(tl.float32)
          + (db * si).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    nb = ((db * co).to(X.dtype.element_ty).to(tl.float32)
          - (da * si).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    dw_a = tl.sum(na * (a * inverse[:, None]), 0)
    dw_b = tl.sum(nb * (b * inverse[:, None]), 0)
    tl.store(DW + tl.program_id(0) * D + col, dw_a)
    tl.store(DW + tl.program_id(0) * D + D // 2 + col, dw_b)


def packed_qkv_forward_v2(packed, cos, sin, q, kv, eps, normalize_qk=True, q_gain=None, k_gain=None,
                          *, pairs=8, num_warps=4):
    batch, length, width = packed.shape
    dim = width // (q + 2 * kv)
    gain = q_gain is not None or k_gain is not None
    out = torch.empty((batch, length, q + kv, dim), device=packed.device, dtype=packed.dtype)
    inverse = torch.empty((batch * length, q + kv) if normalize_qk else (0,), device=packed.device, dtype=torch.float32)
    rows = batch * length
    _packed_rope_v2[(triton.cdiv(rows * (q + kv), pairs),)](
        packed, cos, sin, inverse, out, None, None, None, None, q_gain, k_gain,
        q, kv, dim, length, 0, 0, 0, 0, 0, 0, 0, 0, 0, rows, eps, False, pairs,
        normalize_qk, gain, cos.shape[0] != 1, num_warps=num_warps, enable_fp_fusion=False)
    value = packed.view(batch, length, q + 2 * kv, dim)[:, :, q + kv:]
    return out[:, :, :q].transpose(1, 2), out[:, :, q:].transpose(1, 2), value.transpose(1, 2), inverse


def packed_qkv_backward_rope_v2(dq, dk, dv, packed, cos, sin, inverse, q, kv, normalize_qk=True,
                                q_gain=None, k_gain=None, *, pairs=8, num_warps=4):
    """Only the fused rope/norm backward (the gain reduction is separate)."""
    batch, length, width = packed.shape
    dim = width // (q + 2 * kv)
    gain = q_gain is not None
    dq, dk, dv = (g if g.stride(-1) == 1 else g.contiguous() for g in (dq, dk, dv))
    output = torch.empty_like(packed)
    rows = batch * length
    _packed_rope_v2[(triton.cdiv(rows * (q + 2 * kv), pairs),)](
        packed, cos, sin, inverse, None, dq, dk, dv, output, q_gain, k_gain,
        q, kv, dim, length, *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3], rows, 0., True,
        pairs, normalize_qk, gain, cos.shape[0] != 1, num_warps=num_warps, enable_fp_fusion=False)
    return output


def tiled_gain_backward_v2(dq, dk, packed, cos, sin, inverse, q, kv, *, tile=64, num_warps=4):
    """Reference two-launch structure with tunable tile width / warps."""
    batch, length, width = packed.shape
    dim = width // (q + 2 * kv)
    gradients = []
    for heads, start, gradient in ((q, 0, dq), (kv, q, dk)):
        programs = triton.cdiv(batch * length * heads, tile)
        partial = torch.empty((programs, dim), device=packed.device, dtype=torch.float32)
        _packed_gain_tiled_v2[(programs,)](
            packed, cos, sin, inverse, gradient, partial, q, kv, dim, length,
            *gradient.stride()[:3], batch * length, heads, start, cos.shape[0] != 1, tile,
            num_warps=num_warps, enable_fp_fusion=False)
        gradients.append(partial.sum(0))
    return gradients
