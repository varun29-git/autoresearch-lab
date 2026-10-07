"""Fused operators for the stack's two additions and the Peri-LN block boundary.

The trial-004 stack (branch_norm="peri", value_residual="learned") currently
runs each addition as separate pointwise kernels: RMSNorm + residual add per
branch, and a multiply-add chain for the V' = l0*V1 + l1*Vn value mix. Each
operator below collapses one of those chains into a single kernel with a custom
backward that retains only what the gradient math needs:

- out_norm_residual: x' = x + OutNorm(branch) — one kernel, saves branch and
  the inverse RMS only (the residual gradient is the identity).
- block_boundary (idea 3): x' = x + OutNorm(branch), h = InNorm(x') in one
  pass. Saves x' (the residual stream, which persists anyway), branch, both
  inverse RMS vectors and the gains; h and the normed branch never persist.
- value_mix: V' = l0*V1 + l1*Vn in one pass, with dV1/dVn/dλ in one pass.

CPU paths are exact reference reimplementations of the eager model math; CUDA
paths use the kernels in stack_fusion_kernels.py. The kernels keep one BF16
rounding where eager rounds per operator, so GPU-vs-eager parity is checked
with BF16 tolerance in optimizations/gpu_check.py.
"""
from __future__ import annotations

import math

import torch


def _norm_stats(flat, eps):
    inverse = torch.rsqrt(flat.float().square().mean(dim=-1) + eps)
    return inverse


def _validate_boundary(residual, branch, gain, eps, name):
    if residual.shape != branch.shape or residual.device != branch.device:
        raise ValueError(f"{name} requires residual and branch with identical shape and device")
    if branch.ndim < 1 or not branch.is_floating_point() or not residual.is_floating_point():
        raise ValueError(f"{name} requires nonempty floating-point operands")
    if gain is not None and (gain.shape != (branch.shape[-1],) or gain.device != branch.device):
        raise ValueError(f"{name} requires a channel gain of shape (width,) on the input device")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"{name} requires positive finite epsilon")


class _OutNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, branch, gain, eps):
        out_dtype = torch.promote_types(residual.dtype, branch.dtype)
        if branch.is_cuda:
            from . import stack_fusion_kernels as kernels

            output, inverse, flat = kernels.out_norm_residual_forward(residual, branch, gain, eps, out_dtype)
        else:
            flat = branch.reshape(-1, branch.shape[-1])
            inverse = _norm_stats(flat, eps)
            normalized = (flat.float() * inverse[:, None] * gain.float()[None, :]).to(branch.dtype)
            output = residual.reshape(flat.shape) + normalized
        ctx.save_for_backward(flat, gain, inverse)
        ctx.branch_dtype = branch.dtype
        ctx.residual_dtype = residual.dtype
        return output.reshape(branch.shape)

    @staticmethod
    def backward(ctx, gradient):
        branch, gain, inverse = ctx.saved_tensors
        rows, width = branch.shape
        if branch.is_cuda:
            from . import stack_fusion_kernels as kernels

            d_branch, d_gain = kernels.out_norm_residual_backward(gradient, branch, gain, inverse)
        else:
            dy = gradient.reshape(rows, width).to(ctx.branch_dtype).float()
            normalized = branch.float() * inverse[:, None]
            d_gain = (dy * normalized).sum(0).to(gain.dtype)
            s = dy * gain.float()[None, :]
            d_branch = ((s - normalized * (s * normalized).mean(dim=-1, keepdim=True)) * inverse[:, None]).to(ctx.branch_dtype)
        return gradient.to(ctx.residual_dtype), d_branch.reshape(gradient.shape), d_gain, None


def out_norm_residual(residual, branch, gain, eps=1e-5):
    """x' = residual + gain * RMSNorm(branch), one kernel, minimal retention."""
    _validate_boundary(residual, branch, gain, eps, "out_norm_residual")
    return _OutNormResidual.apply(residual, branch, gain, eps)


class _BlockBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, branch, out_gain, in_gain, eps):
        out_dtype = torch.promote_types(residual.dtype, branch.dtype)
        if branch.is_cuda:
            from . import stack_fusion_kernels as kernels

            xn, h, inverse_out, inverse_in, flat = kernels.boundary_forward(
                residual, branch, out_gain, in_gain, eps, out_dtype)
        else:
            flat = branch.reshape(-1, branch.shape[-1])
            inverse_out = _norm_stats(flat, eps)
            normalized = (flat.float() * inverse_out[:, None] * out_gain.float()[None, :]).to(branch.dtype)
            xn = (residual.reshape(flat.shape) + normalized).to(out_dtype)
            inverse_in = _norm_stats(xn, eps)
            h = (xn.float() * inverse_in[:, None] * in_gain.float()[None, :]).to(out_dtype)
        ctx.save_for_backward(xn, flat, out_gain, in_gain, inverse_out, inverse_in)
        ctx.branch_dtype = branch.dtype
        ctx.residual_dtype = residual.dtype
        shape = branch.shape
        return xn.reshape(shape), h.reshape(shape)

    @staticmethod
    def backward(ctx, d_xn, d_h):
        xn, branch, out_gain, in_gain, inverse_out, inverse_in = ctx.saved_tensors
        rows, width = branch.shape
        if branch.is_cuda:
            from . import stack_fusion_kernels as kernels

            d_residual, d_branch, d_out_gain, d_in_gain = kernels.boundary_backward(
                d_xn, d_h, xn, branch, out_gain, in_gain, inverse_out, inverse_in)
        else:
            d_h = d_h.reshape(rows, width).to(xn.dtype).float()
            xn_hat = xn.float() * inverse_in[:, None]
            d_in_gain = (d_h * xn_hat).sum(0).to(in_gain.dtype)
            t = d_h * in_gain.float()[None, :]
            d_residual = d_xn.reshape(rows, width) + ((t - xn_hat * (t * xn_hat).mean(dim=-1, keepdim=True))
                                                      * inverse_in[:, None]).to(xn.dtype)
            dy = d_residual.to(ctx.branch_dtype).float()
            b_hat = branch.float() * inverse_out[:, None]
            d_out_gain = (dy * b_hat).sum(0).to(out_gain.dtype)
            s = dy * out_gain.float()[None, :]
            d_branch = ((s - b_hat * (s * b_hat).mean(dim=-1, keepdim=True)) * inverse_out[:, None]).to(ctx.branch_dtype)
        shape = d_xn.shape
        return (d_residual.reshape(shape).to(ctx.residual_dtype), d_branch.reshape(shape),
                d_out_gain, d_in_gain, None)


def block_boundary(residual, branch, out_gain, in_gain, eps=1e-5):
    """(x' = residual + OutNorm(branch), h = InNorm(x')) as one fused operator.

    Idea 3 of the optimization set: the Peri-LN output norm, the residual add
    and the following input norm share one kernel launch and one pass over the
    row. Backward retains x' (the residual stream, alive anyway), the branch
    output and the two inverse-RMS vectors; it never materializes the normed
    branch or a separate h buffer.
    """
    _validate_boundary(residual, branch, out_gain, eps, "block_boundary")
    _validate_boundary(residual, branch, in_gain, eps, "block_boundary")
    return _BlockBoundary.apply(residual, branch, out_gain, in_gain, eps)


class _ValueMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values_first, values_current, lambdas):
        if values_first.is_cuda:
            from . import stack_fusion_kernels as kernels

            output = kernels.value_mix_forward(values_first, values_current, lambdas)
        else:
            output = lambdas[0] * values_first + lambdas[1] * values_current
        ctx.save_for_backward(values_first, values_current, lambdas)
        return output

    @staticmethod
    def backward(ctx, gradient):
        values_first, values_current, lambdas = ctx.saved_tensors
        if values_first.is_cuda:
            from . import stack_fusion_kernels as kernels

            return kernels.value_mix_backward(gradient, values_first, values_current, lambdas)
        d_first = lambdas[0] * gradient
        d_current = lambdas[1] * gradient
        d_lambdas = torch.stack(((gradient * values_first).sum(), (gradient * values_current).sum()))
        return d_first, d_current, d_lambdas.to(lambdas.dtype)


def value_mix(values_first, values_current, lambdas):
    """V' = lambda[0]*V1 + lambda[1]*Vn in one kernel (stack value residual)."""
    if values_first.shape != values_current.shape or values_first.device != values_current.device:
        raise ValueError("value_mix requires same-shape values on one device")
    if values_first.dtype != values_current.dtype or not values_first.is_floating_point():
        raise ValueError("value_mix requires matching floating-point dtypes")
    if lambdas.shape != (2,) or lambdas.device != values_first.device:
        raise ValueError("value_mix requires a 2-element lambda vector on the input device")
    return _ValueMix.apply(values_first, values_current, lambdas)
