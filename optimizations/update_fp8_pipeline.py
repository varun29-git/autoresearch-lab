"""Idea 5: treat the optimizer update and FP8 cache production as one pipeline.

At the stack's two accumulation steps per rank, every Muon update is followed
immediately by a rebuild of the projection's FP8 weight operands (both GEMM
orientations) plus, under idea 1's owner scheme, one communication of those
operands. This module provides the two building blocks:

- produce_weight_operands: one call that computes a projection's block scales
  once and emits both FP8 layouts (rowwise and transposed) from that single
  scale pass, for both ternary QAT and dense FP8 weight modes. The bundle is
  the exact object idea 1's owner broadcasts after updating a matrix.
- batched_muon_update: groups equal-shaped matrices and runs the pinned
  Muon Newton-Schulz iteration as batched GEMMs (the pinned implementation
  already accepts stacked inputs) instead of one 5-step iteration per matrix.

Logical fused projections (packed QKV / gate-up) must be sliced into their
logical rows before batching, exactly as LogicalMuonWithAuxAdam slices them;
batching then groups the slices by shape.
"""
from __future__ import annotations

import torch

from terminus.quantization import dense_weight_scales, expand_scales, weight_scales


@torch.no_grad()
def produce_weight_operands(weight, *, weight_mode, transposed=True, block_size=128):
    """One scale pass -> both FP8 layouts for a projection's forward/backward GEMMs.

    Returns {"forward": (codes, scales), "backward": (codes_t, scales_t) | None}
    with FP8 E4M3 codes and FP32 blockwise scales, matching the operand format
    terminus.kernels.deep_gemm_fp8_gemm_nt consumes. Ternary mode uses the
    mean-abs block scales and +/-1 codes of terminus.quantization; dense mode
    uses absmax/448 scales.
    """
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("weight operands require a nonempty matrix")
    if weight_mode not in {"ternary", "dense"}:
        raise ValueError("weight_mode must be ternary or dense")
    if weight_mode == "ternary":
        scales = weight_scales(weight, block_size, 1e-6, layout="blockwise")
        codes = (weight.float() / expand_scales(scales, *weight.shape, block_size, "blockwise")).round().clamp(-1, 1)
    else:
        scales = dense_weight_scales(weight, block_size)
        scaled = weight.float() / expand_scales(scales, *weight.shape, block_size, "blockwise")
        codes = torch.where(torch.isfinite(scaled), scaled.clamp(-448.0, 448.0),
                            torch.full_like(scaled, float("nan")))
    codes = codes.to(torch.float8_e4m3fn).contiguous()
    bundle = {"forward": (codes, scales.contiguous())}
    bundle["backward"] = ((codes.T.contiguous(), scales.T.contiguous()) if transposed else None)
    return bundle


def operand_nbytes(bundle):
    """Communication volume of one operand bundle (codes as one byte, scales FP32)."""
    total = 0
    for direction in bundle.values():
        if direction is not None:
            total += sum(tensor.numel() * tensor.element_size() for tensor in direction)
    return total


def _muon_spectral_scale(rows, width):
    return max(1, rows / width) ** 0.5


@torch.no_grad()
def batched_muon_update(grads, momentums, *, beta=0.95, ns_steps=5, nesterov=True):
    """Muon orthogonalized updates for a list of same-shape-grouped matrices.

    grads: list of 2D FP32 gradient tensors. momentums: list of same-shape
    buffers, updated in place (the pinned Muon contract). Returns the list of
    updates in input order; the caller applies lr/weight decay. Equal-shaped
    matrices share one batched Newton-Schulz run; shapes with a single member
    take the ordinary per-matrix path.
    """
    from muon import zeropower_via_newtonschulz5

    if len(grads) != len(momentums) or not grads:
        raise ValueError("batched Muon requires one momentum buffer per gradient")
    for grad, momentum in zip(grads, momentums):
        if grad.ndim != 2 or grad.shape != momentum.shape:
            raise ValueError("batched Muon requires 2D gradients with matching momentum buffers")
    updates = [None] * len(grads)
    by_shape = {}
    for index, grad in enumerate(grads):
        by_shape.setdefault(tuple(grad.shape), []).append(index)
    for (rows, width), indices in by_shape.items():
        stacked_grad = torch.stack([grads[index] for index in indices])
        stacked_momentum = torch.stack([momentums[index] for index in indices])
        stacked_momentum.lerp_(stacked_grad, 1 - beta)
        for slot, index in enumerate(indices):
            momentums[index].copy_(stacked_momentum[slot])
        update = stacked_grad.lerp(stacked_momentum, beta) if nesterov else stacked_momentum
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        update = update * _muon_spectral_scale(rows, width)
        for slot, index in enumerate(indices):
            updates[index] = update[slot]
    return updates
