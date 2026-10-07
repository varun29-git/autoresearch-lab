"""CUDA kernels for the stack's two fused additions (Peri-LN boundary, value mix).

FP32 RMS statistics match terminus/pointwise_kernels.py; gain gradients use the
same per-CTA partial workspace reduction as norm_backward.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _out_norm_residual_forward(X, B, G, Y, R, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    x = tl.load(X + row * WIDTH + col, mask, other=0).to(tl.float32)
    b = tl.load(B + row * WIDTH + col, mask, other=0).to(tl.float32)
    g = tl.load(G + col, mask, other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(b * b, 0) / WIDTH + EPS)
    tl.store(Y + row * WIDTH + col, x + b * inverse * g, mask)
    tl.store(R + row, inverse)


@triton.jit
def _out_norm_residual_backward(DY, B, G, R, DB, DG, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                                PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    g = tl.load(G + col, mask, other=0).to(tl.float32)
    dg = tl.full((BLOCK,), 0., tl.float32)
    for row in range(program, ROWS, PROGRAMS):
        dy = tl.load(DY + row * WIDTH + col, mask, other=0).to(tl.float32)
        b = tl.load(B + row * WIDTH + col, mask, other=0).to(tl.float32)
        inverse = tl.load(R + row)
        normalized = b * inverse
        dg += dy * normalized
        s = dy * g
        db = (s - normalized * (tl.sum(s * normalized, 0) / WIDTH)) * inverse
        tl.store(DB + row * WIDTH + col, db, mask)
    tl.store(DG + program * WIDTH + col, dg, mask)


def out_norm_residual_forward(residual, branch, gain, eps, out_dtype):
    flat_residual = residual.contiguous().view(-1, residual.shape[-1])
    flat_branch = branch.contiguous().view(-1, branch.shape[-1])
    rows, width = flat_branch.shape
    output = torch.empty(rows, width, device=branch.device, dtype=out_dtype)
    inverse = torch.empty(rows, device=branch.device, dtype=torch.float32)
    _out_norm_residual_forward[(rows,)](flat_residual, flat_branch, gain, output, inverse,
                                        width, eps, triton.next_power_of_2(width), enable_fp_fusion=False)
    return output, inverse, flat_branch


def out_norm_residual_backward(gradient, branch, gain, inverse, *, programs=512):
    rows, width = branch.shape
    programs = min(rows, programs)
    dbranch = torch.empty_like(branch)
    partial = torch.empty((programs, width), device=branch.device, dtype=torch.float32)
    _out_norm_residual_backward[(programs,)](gradient.contiguous().view(rows, width), branch, gain, inverse,
                                             dbranch, partial, rows, width, programs,
                                             triton.next_power_of_2(width), enable_fp_fusion=False)
    return dbranch, partial.sum(0).reshape(gain.shape).to(gain.dtype)


@triton.jit
def _boundary_forward(X, B, GO, GI, XN, H, RO, RI, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    x = tl.load(X + row * WIDTH + col, mask, other=0).to(tl.float32)
    b = tl.load(B + row * WIDTH + col, mask, other=0).to(tl.float32)
    go = tl.load(GO + col, mask, other=0).to(tl.float32)
    gi = tl.load(GI + col, mask, other=0).to(tl.float32)
    inverse_out = tl.rsqrt(tl.sum(b * b, 0) / WIDTH + EPS)
    xn = x + b * inverse_out * go
    tl.store(XN + row * WIDTH + col, xn, mask)
    tl.store(RO + row, inverse_out)
    inverse_in = tl.rsqrt(tl.sum(xn * xn, 0) / WIDTH + EPS)
    tl.store(H + row * WIDTH + col, xn * inverse_in * gi, mask)
    tl.store(RI + row, inverse_in)


@triton.jit
def _boundary_backward(DXN, DH, XN, B, GO, GI, RO, RI, DXR, DB, DGO, DGI,
                       ROWS: tl.constexpr, WIDTH: tl.constexpr, PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    go = tl.load(GO + col, mask, other=0).to(tl.float32)
    gi = tl.load(GI + col, mask, other=0).to(tl.float32)
    dgo = tl.full((BLOCK,), 0., tl.float32)
    dgi = tl.full((BLOCK,), 0., tl.float32)
    for row in range(program, ROWS, PROGRAMS):
        xn = tl.load(XN + row * WIDTH + col, mask, other=0).to(tl.float32)
        dh = tl.load(DH + row * WIDTH + col, mask, other=0).to(tl.float32)
        inverse_in = tl.load(RI + row)
        xn_hat = xn * inverse_in
        dgi += dh * xn_hat
        t = dh * gi
        dxn = tl.load(DXN + row * WIDTH + col, mask, other=0).to(tl.float32)
        dxn = dxn + (t - xn_hat * (tl.sum(t * xn_hat, 0) / WIDTH)) * inverse_in
        tl.store(DXR + row * WIDTH + col, dxn, mask)
        b = tl.load(B + row * WIDTH + col, mask, other=0).to(tl.float32)
        inverse_out = tl.load(RO + row)
        b_hat = b * inverse_out
        dgo += dxn * b_hat
        s = dxn * go
        db = (s - b_hat * (tl.sum(s * b_hat, 0) / WIDTH)) * inverse_out
        tl.store(DB + row * WIDTH + col, db, mask)
    tl.store(DGO + program * WIDTH + col, dgo, mask)
    tl.store(DGI + program * WIDTH + col, dgi, mask)


def boundary_forward(residual, branch, out_gain, in_gain, eps, out_dtype):
    flat_residual = residual.contiguous().view(-1, residual.shape[-1])
    flat_branch = branch.contiguous().view(-1, branch.shape[-1])
    rows, width = flat_branch.shape
    xn = torch.empty(rows, width, device=branch.device, dtype=out_dtype)
    h = torch.empty(rows, width, device=branch.device, dtype=out_dtype)
    inverse_out = torch.empty(rows, device=branch.device, dtype=torch.float32)
    inverse_in = torch.empty_like(inverse_out)
    _boundary_forward[(rows,)](flat_residual, flat_branch, out_gain, in_gain, xn, h,
                               inverse_out, inverse_in, width, eps,
                               triton.next_power_of_2(width), enable_fp_fusion=False)
    return xn, h, inverse_out, inverse_in, flat_branch


def boundary_backward(d_xn, d_h, xn, branch, out_gain, in_gain, inverse_out, inverse_in, *, programs=512):
    rows, width = branch.shape
    programs = min(rows, programs)
    d_residual = torch.empty_like(xn)
    d_branch = torch.empty_like(branch)
    partial_out = torch.empty((programs, width), device=branch.device, dtype=torch.float32)
    partial_in = torch.empty_like(partial_out)
    _boundary_backward[(programs,)](d_xn.contiguous().view(rows, width), d_h.contiguous().view(rows, width),
                                    xn, branch, out_gain, in_gain, inverse_out, inverse_in,
                                    d_residual, d_branch, partial_out, partial_in,
                                    rows, width, programs, triton.next_power_of_2(width),
                                    enable_fp_fusion=False)
    return (d_residual, d_branch, partial_out.sum(0).to(out_gain.dtype),
            partial_in.sum(0).to(in_gain.dtype))


@triton.jit
def _value_mix_forward(V1, VN, L, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    v1 = tl.load(V1 + i, mask, other=0).to(tl.float32)
    vn = tl.load(VN + i, mask, other=0).to(tl.float32)
    l0 = tl.load(L + 0).to(tl.float32)
    l1 = tl.load(L + 1).to(tl.float32)
    tl.store(OUT + i, l0 * v1 + l1 * vn, mask)


@triton.jit
def _value_mix_backward(DOUT, V1, VN, L, DV1, DVN, DL, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    dout = tl.load(DOUT + i, mask, other=0).to(tl.float32)
    v1 = tl.load(V1 + i, mask, other=0).to(tl.float32)
    vn = tl.load(VN + i, mask, other=0).to(tl.float32)
    l0 = tl.load(L + 0).to(tl.float32)
    l1 = tl.load(L + 1).to(tl.float32)
    tl.store(DV1 + i, l0 * dout, mask)
    tl.store(DVN + i, l1 * dout, mask)
    program = tl.program_id(0)
    tl.store(DL + program * 2 + 0, tl.sum(dout * v1, 0))
    tl.store(DL + program * 2 + 1, tl.sum(dout * vn, 0))


def value_mix_forward(values_first, values_current, lambdas):
    output = torch.empty_like(values_first)
    _value_mix_forward[(triton.cdiv(values_first.numel(), 4096),)](
        values_first, values_current, lambdas, output, values_first.numel(), 4096,
        enable_fp_fusion=False)
    return output


def value_mix_backward(gradient, values_first, values_current, lambdas):
    programs = triton.cdiv(gradient.numel(), 4096)
    d_first = torch.empty_like(values_first)
    d_current = torch.empty_like(values_current)
    partial = torch.empty((programs, 2), device=gradient.device, dtype=torch.float32)
    _value_mix_backward[(programs,)](gradient.contiguous(), values_first, values_current, lambdas,
                                     d_first, d_current, partial, gradient.numel(), 4096,
                                     enable_fp_fusion=False)
    return d_first, d_current, partial.sum(0).to(lambdas.dtype)
