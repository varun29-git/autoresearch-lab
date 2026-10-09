"""Normalize/rotate multiple Q/K heads per CTA; write packed QKV gradients.

Learned channel gains (learned_rms) share one vector across the query heads and
one across the key heads. Rounding boundaries match the unfused reference:
rms_norm's single BF16 store after the FP32 gain multiply, then per-multiply
BF16 rounding in the rotary, including the backward rotary inverse.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _packed_rope(X, Cos, Sin, R, Y, DQ, DK, DV, DX, GQ, GK,
                 Q: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 QB: tl.constexpr, QH: tl.constexpr, QL: tl.constexpr,
                 KB: tl.constexpr, KH: tl.constexpr, KL: tl.constexpr,
                 VB: tl.constexpr, VH: tl.constexpr, VL: tl.constexpr,
                 EPS: tl.constexpr, BACKWARD: tl.constexpr, HEADS: tl.constexpr,
                 NORMALIZE: tl.constexpr, GAIN: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    c = tl.arange(0, D // 2)
    pos, batch = row % L, row // L
    offset = row * ((Q + 2 * KV) * D) + h[:, None] * D + c[None, :]
    a = tl.load(X + offset, h[:, None] < Q + KV, other=0.).to(tl.float32)
    b = tl.load(X + offset + D // 2, h[:, None] < Q + KV, other=0.).to(tl.float32)
    co = tl.load(Cos + pos * (D // 2) + c).to(X.dtype.element_ty).to(tl.float32)
    si = tl.load(Sin + pos * (D // 2) + c).to(X.dtype.element_ty).to(tl.float32)
    if GAIN:
        is_q = h[:, None] < Q
        ga = tl.where(is_q, tl.load(GQ + c), tl.load(GK + c)).to(tl.float32)
        gb = tl.where(is_q, tl.load(GQ + D // 2 + c), tl.load(GK + D // 2 + c)).to(tl.float32)
    if BACKWARD:
        qo = batch * QB + h[:, None] * QH + pos * QL + c[None, :]
        ko = batch * KB + (h[:, None] - Q) * KH + pos * KL + c[None, :]
        vo = batch * VB + (h[:, None] - Q - KV) * VH + pos * VL + c[None, :]
        da = tl.load(DQ + qo, h[:, None] < Q, other=0.).to(tl.float32)
        db = tl.load(DQ + qo + D // 2, h[:, None] < Q, other=0.).to(tl.float32)
        da += tl.load(DK + ko, (h[:, None] >= Q) & (h[:, None] < Q + KV), other=0.).to(tl.float32)
        db += tl.load(DK + ko + D // 2, (h[:, None] >= Q) & (h[:, None] < Q + KV), other=0.).to(tl.float32)
        na = ((da * co[None, :]).to(X.dtype.element_ty).to(tl.float32)
              + (db * si[None, :]).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        nb = ((db * co[None, :]).to(X.dtype.element_ty).to(tl.float32)
              - (da * si[None, :]).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        if NORMALIZE:
            inverse = tl.load(R + row * (Q + KV) + h, h < Q + KV, other=0.)
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
        va = tl.load(DV + vo, (h[:, None] >= Q + KV) & (h[:, None] < Q + 2 * KV), other=0.).to(tl.float32)
        vb = tl.load(DV + vo + D // 2, (h[:, None] >= Q + KV) & (h[:, None] < Q + 2 * KV), other=0.).to(tl.float32)
        tl.store(DX + offset, tl.where(h[:, None] < Q + KV, dx_a, va), h[:, None] < Q + 2 * KV)
        tl.store(DX + offset + D // 2, tl.where(h[:, None] < Q + KV, dx_b, vb), h[:, None] < Q + 2 * KV)
    else:
        if NORMALIZE:
            inverse = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1)) / D + EPS)
            if GAIN:
                a = (a * inverse[:, None] * ga).to(X.dtype.element_ty).to(tl.float32)
                b = (b * inverse[:, None] * gb).to(X.dtype.element_ty).to(tl.float32)
            else:
                a = (a * inverse[:, None]).to(X.dtype.element_ty).to(tl.float32)
                b = (b * inverse[:, None]).to(X.dtype.element_ty).to(tl.float32)
        ya = (a * co[None, :]).to(X.dtype.element_ty).to(tl.float32) - (b * si[None, :]).to(X.dtype.element_ty).to(tl.float32)
        yb = (b * co[None, :]).to(X.dtype.element_ty).to(tl.float32) + (a * si[None, :]).to(X.dtype.element_ty).to(tl.float32)
        out = row * ((Q + KV) * D) + h[:, None] * D + c[None, :]
        tl.store(Y + out, ya, h[:, None] < Q + KV)
        tl.store(Y + out + D // 2, yb, h[:, None] < Q + KV)
        if NORMALIZE:
            tl.store(R + row * (Q + KV) + h, inverse, h < Q + KV)


@triton.jit
def _packed_gain_backward(X, Cos, Sin, R, DQ, DK, GQ, GK, DW,
                          Q: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                          QB: tl.constexpr, QH: tl.constexpr, QL: tl.constexpr,
                          KB: tl.constexpr, KH: tl.constexpr, KL: tl.constexpr,
                          ROWS: tl.constexpr, PROGRAMS: tl.constexpr):
    """Deterministic bounded-program reduction of the shared Q/K gain gradients.

    Recomputes the rotary inverse of the incoming gradients and accumulates
    dy (pre-gain) against the ungained normalized activations, matching
    rms_norm backward's dw = sum_rows(dy * x_hat). DQ/DK keep the stride layout
    autograd delivered, exactly like the main packed backward kernel."""
    program = tl.program_id(0)
    c = tl.arange(0, D // 2)
    dw_qa = tl.zeros((D // 2,), tl.float32)
    dw_qb = tl.zeros((D // 2,), tl.float32)
    dw_ka = tl.zeros((D // 2,), tl.float32)
    dw_kb = tl.zeros((D // 2,), tl.float32)
    total = ROWS * (Q + KV)
    for index in range(program, total, PROGRAMS):
        row = index // (Q + KV)
        h = index % (Q + KV)
        pos = row % L
        offset = row * ((Q + 2 * KV) * D) + h * D + c
        a = tl.load(X + offset).to(tl.float32)
        b = tl.load(X + offset + D // 2).to(tl.float32)
        inverse = tl.load(R + row * (Q + KV) + h)
        co = tl.load(Cos + pos * (D // 2) + c).to(X.dtype.element_ty).to(tl.float32)
        si = tl.load(Sin + pos * (D // 2) + c).to(X.dtype.element_ty).to(tl.float32)
        if h < Q:
            go = (row // L) * QB + h * QH + pos * QL + c
            da = tl.load(DQ + go).to(tl.float32)
            db = tl.load(DQ + go + D // 2).to(tl.float32)
        else:
            go = (row // L) * KB + (h - Q) * KH + pos * KL + c
            da = tl.load(DK + go).to(tl.float32)
            db = tl.load(DK + go + D // 2).to(tl.float32)
        na = ((da * co).to(X.dtype.element_ty).to(tl.float32)
              + (db * si).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        nb = ((db * co).to(X.dtype.element_ty).to(tl.float32)
              - (da * si).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        ax, bx = a * inverse, b * inverse
        if h < Q:
            dw_qa += na * ax
            dw_qb += nb * bx
        else:
            dw_ka += na * ax
            dw_kb += nb * bx
    half = tl.arange(0, D // 2)
    tl.store(DW + program * (2 * D) + half, dw_qa)
    tl.store(DW + program * (2 * D) + D // 2 + half, dw_qb)
    tl.store(DW + program * (2 * D) + D + half, dw_ka)
    tl.store(DW + program * (2 * D) + D + D // 2 + half, dw_kb)


def packed_qkv_forward(packed, cos, sin, q, kv, eps, normalize_qk=True, q_gain=None, k_gain=None):
    batch, length, width = packed.shape
    dim = width // (q + 2 * kv)
    gain = q_gain is not None or k_gain is not None
    if gain and not (normalize_qk and q_gain is not None and k_gain is not None
                     and q_gain.shape == (dim,) and k_gain.shape == (dim,)):
        raise ValueError("learned Q/K gains require normalization and one gain vector per head width")
    out = torch.empty((batch, length, q + kv, dim), device=packed.device, dtype=packed.dtype)
    inverse = torch.empty((batch * length, q + kv) if normalize_qk else (0,), device=packed.device, dtype=torch.float32)
    _packed_rope[(batch * length, triton.cdiv(q + kv, 4))](packed, cos, sin, inverse, out, None, None, None, None,
        q_gain, k_gain, q, kv, dim, length, 0, 0, 0, 0, 0, 0, 0, 0, 0, eps, False, 4, normalize_qk, gain,
        num_warps=4, enable_fp_fusion=False)
    value = packed.view(batch, length, q + 2 * kv, dim)[:, :, q + kv:]
    return out[:, :, :q].transpose(1, 2), out[:, :, q:].transpose(1, 2), value.transpose(1, 2), inverse


def packed_qkv_backward(dq, dk, dv, packed, cos, sin, inverse, q, kv, normalize_qk=True, q_gain=None, k_gain=None):
    batch, length, width = packed.shape
    dim = width // (q + 2 * kv)
    gain = q_gain is not None
    if gain != (k_gain is not None):
        raise ValueError("Q and K gains must arrive together")
    dq, dk, dv = (g if g.stride(-1) == 1 else g.contiguous() for g in (dq, dk, dv))
    output = torch.empty_like(packed)
    _packed_rope[(batch * length, triton.cdiv(q + 2 * kv, 4))](packed, cos, sin, inverse, None, dq, dk, dv, output,
        q_gain, k_gain, q, kv, dim, length, *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3], 0., True, 4,
        normalize_qk, gain, num_warps=4, enable_fp_fusion=False)
    if not gain:
        return output, None, None
    # Gain gradients: bounded deterministic reduction over row/head pairs,
    # indexing DQ/DK with the same stride contract as the main kernel.
    programs = min(batch * length * (q + kv), 512)
    partial = torch.empty((programs, 2 * dim), device=packed.device, dtype=torch.float32)
    _packed_gain_backward[(programs,)](packed, cos, sin, inverse, dq, dk, q_gain, k_gain, partial,
        q, kv, dim, length, *dq.stride()[:3], *dk.stride()[:3], batch * length, programs,
        num_warps=4, enable_fp_fusion=False)
    reduced = partial.sum(0)
    return output, reduced[:dim].to(q_gain.dtype), reduced[dim:].to(k_gain.dtype)
