"""Idea 2: the output head and cross entropy as one tiled training operation.

terminus/loss.py loops over token chunks and materializes each chunk's logits
plus its softmax gradient, running separate head forward, input-gradient and
weight-gradient GEMMs. For the tied 65k x 2304 head that is ~151M parameters of
GEMM work per pass and large intermediates per chunk. This module instead tiles
over the *vocabulary*: forward computes exact full-vocabulary FP32 log-sum-exp
with an online running max/sum and never materializes logits; backward
recomputes each tile's logits and forms the input and weight gradients in FP32
accumulators. The head operands (BF16 cast or FP8 codes) are recomputed per
tile in backward from the saved FP32 weight, so no [rows, vocab] or cast
[vocab, dim] buffer persists.

Semantics match an unnormalized cross entropy sum (the training loop's
loss_sum): gradient of the returned scalar w.r.t. each hidden row is
softmax - onehot for valid targets and zero for -100. Recomputing logits in
backward is the same trade the checkpoint loss backend already makes; the win
is bounded memory and one autograd node instead of a chunk loop.

fp8_head=True quantizes each head tile to FP8 E4M3 with blockwise absmax scales
(dense_weight_scales recipe) before the GEMM, retaining FP32 softmax
accumulation. The reference path dequantizes for the matmul, which reproduces
FP8 GEMM numerics exactly on any device; on the H100 the same codes/scales feed
deep_gemm_fp8_gemm_nt directly (wired up in gpu_check.py). This changes
numerics and must earn its place through the dense-and-QAT quality comparison
before adoption.
"""
from __future__ import annotations

import torch

from terminus.quantization import dequantize_weight_fp8_e4m3_2d, quantize_weight_dense_fp8_e4m3_2d


def _head_tile(weight_tile, compute_dtype, fp8_head):
    if fp8_head:
        codes, scales = quantize_weight_dense_fp8_e4m3_2d(weight_tile)
        return dequantize_weight_fp8_e4m3_2d(codes, scales, out_dtype=compute_dtype)
    return weight_tile.to(compute_dtype)


class FusedHeadLoss(torch.autograd.Function):
    """Exact full-vocabulary linear cross entropy sum, tiled over the vocab."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, tile_size, fp8_head):
        ctx.hidden_shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
        ctx.weight_dtype = weight.dtype
        flat = hidden.reshape(-1, hidden.shape[-1])
        compute_dtype = flat.dtype if flat.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.float32
        flat = flat.to(compute_dtype)
        labels = targets.reshape(-1)
        rows, vocab = labels.numel(), weight.shape[0]
        valid = labels.ne(-100)
        running_max = torch.full((rows,), float("-inf"), device=hidden.device, dtype=torch.float32)
        running_sum = torch.zeros(rows, device=hidden.device, dtype=torch.float32)
        target_logits = torch.zeros_like(running_sum)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, vocab, tile_size):
                stop = min(start + tile_size, vocab)
                head = _head_tile(weight[start:stop], compute_dtype, fp8_head)
                logits = (flat @ head.T).float()
                tile_max = logits.max(dim=-1).values
                tile_sum = torch.exp(logits - tile_max[:, None]).sum(dim=-1)
                new_max = torch.maximum(running_max, tile_max)
                running_sum = running_sum * torch.exp(running_max - new_max) + tile_sum * torch.exp(tile_max - new_max)
                running_max = new_max
                in_tile = valid & labels.ge(start) & labels.lt(stop)
                if in_tile.any():
                    target_logits[in_tile] = logits[in_tile, labels[in_tile] - start]
        lse = running_max + torch.log(running_sum)
        loss = torch.where(valid, lse - target_logits, torch.zeros_like(lse)).sum()
        ctx.save_for_backward(flat, weight, labels, lse, valid)
        ctx.tile_size, ctx.fp8_head = tile_size, fp8_head
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        flat, weight, labels, lse, valid = ctx.saved_tensors
        rows, dim = flat.shape
        vocab = weight.shape[0]
        d_hidden = torch.zeros_like(flat, dtype=torch.float32) if ctx.needs_input_grad[0] else None
        d_weight = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        # cuBLAS accumulates BF16 operands directly into FP32 C (same recipe as
        # terminus/loss.py); elsewhere accumulate with FP32 operands.
        fast = flat.is_cuda and flat.dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type=weight.device.type, enabled=False):
            for start in range(0, vocab, ctx.tile_size):
                stop = min(start + ctx.tile_size, vocab)
                head = _head_tile(weight[start:stop], flat.dtype, ctx.fp8_head)
                logits = (flat @ head.T).float()
                dz = torch.exp(logits - lse[:, None])
                in_tile = valid & labels.ge(start) & labels.lt(stop)
                if in_tile.any():
                    dz[in_tile, labels[in_tile] - start] -= 1.0
                dz.mul_(valid[:, None])
                if d_hidden is not None:
                    if fast:
                        torch.addmm(d_hidden, dz.to(flat.dtype), head, beta=1, out_dtype=torch.float32, out=d_hidden)
                    else:
                        d_hidden.addmm_(dz, head.float())
                if d_weight is not None:
                    if fast:
                        torch.addmm(d_weight[start:stop], dz.T.to(flat.dtype), flat, beta=1,
                                    out_dtype=torch.float32, out=d_weight[start:stop])
                    else:
                        d_weight[start:stop].addmm_(dz.T, flat.float())
        return ((d_hidden * grad_output).to(ctx.hidden_dtype).reshape(ctx.hidden_shape) if d_hidden is not None else None,
                (d_weight * grad_output).to(ctx.weight_dtype) if d_weight is not None else None,
                None, None, None)


def fused_head_loss(hidden, weight, targets, *, tile_size=4096, fp8_head=False):
    """Cross entropy sum of linear(head) over the full vocabulary, tiled.

    hidden: [..., dim] floating; weight: [vocab, dim] (the tied embedding);
    targets: int64 matching hidden.shape[:-1], -100 ignored. Returns the loss
    SUM as a scalar; divide by the valid token count outside, as the training
    loop already does with loss_sum.
    """
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("fused head loss requires a positive integer tile size")
    if not isinstance(fp8_head, bool):
        raise ValueError("fp8_head must be a boolean")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("fused head loss requires matching hidden/head widths")
    if targets.shape != hidden.shape[:-1] or targets.dtype != torch.long or targets.numel() == 0:
        raise ValueError("fused head loss requires nonempty int64 targets matching hidden rows")
    if hidden.device != weight.device or hidden.device != targets.device:
        raise ValueError("fused head loss operands must share a device")
    if not torch.is_grad_enabled():
        hidden, weight = hidden.detach(), weight.detach()
    return FusedHeadLoss.apply(hidden, weight, targets, tile_size, fp8_head)
