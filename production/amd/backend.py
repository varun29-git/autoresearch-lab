"""Dense and ternary AMD FP8 autograd with explicit scaling and FP32 wgrad.

rocm_fp8 uses AITER for 1Dx2D GEMMs and the local Triton 1Dx1D wgrad.
rocm_fp8_triton uses local Triton for all three GEMMs. Both use FNUZ.
rocm_fp8_te uses ROCm Transformer Engine's HipKittens block GEMMs.
"""
import torch


def validate_recipe(config):
    if (config.weight_mode not in {"dense", "ternary"} or config.fp8_backward != "full"
            or config.fp8_quantizer != "triton_split" or config.weight_scale_layout != "blockwise"
            or config.alpha_group_size != 128 or not config.quantize_activations
            or config.quantization_strength != 1 or config.dim % 128 or config.ffn_dim % 128
            or (config.n_kv_heads * config.head_dim) % 128 or config.fp8_wgrad_recipe != "1d1d"):
        raise ValueError("ROCm FP8 requires full-backward 128-block scaling and immediate dense/ternary weights")
    if config.linear_backend == "rocm_fp8_te" and config.fp8_wgrad_splits:
        raise ValueError("TE owns its split-K scheduling; fp8_wgrad_splits must be zero")


def quantize_directions(values, *, rowwise=True, transposed=False, eps=1e-5):
    from .fp8_kernels import quantize_directions as native

    return native(values, rowwise=rowwise, transposed=transposed, eps=eps)


def quantize_weight(weight, *, transposed=False, weight_mode="dense"):
    from .fp8_kernels import quantize, quantize_ternary

    if weight_mode == "dense":
        primary = quantize(weight, group_rows=128)
    elif weight_mode == "ternary":
        primary = quantize_ternary(weight)
    else:
        raise ValueError("unknown AMD FP8 weight mode")
    # Codes and scales are transposed, not requantized: exact block identity.
    secondary = (primary[0].T.contiguous(), primary[1].T.contiguous()) if transposed else None
    return primary, secondary


def gemm(lhs, rhs, *, backend, rhs_group_size=128, out_dtype=torch.bfloat16, splits=1,
         role="forward", weight_shape=None):
    from .fp8_kernels import check_operands, gemm as triton_gemm
    from .gemm_dispatch import choose_backend, choose_splits

    splits = choose_splits(splits, role, weight_shape)
    backend = choose_backend(backend, role, weight_shape)
    check_operands(lhs, rhs, rhs_group_size, out_dtype)
    if backend == "rocm_fp8_te":
        from .te_gemm import gemm as te_gemm

        return te_gemm(lhs, rhs, rhs_group_size=rhs_group_size, out_dtype=out_dtype)
    if backend == "rocm_fp8_bpreshuffle":
        if rhs_group_size != 128:
            raise ValueError("bpreshuffle blockscale covers 2D-scaled forward/dgrad GEMMs")
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_tune
        from aiter.ops.shuffle import shuffle_weight

        # Same operands as the tuned probe: activation scales keep their
        # producer layout, weight scales are contiguous, and the weight codes
        # are shuffled for the CK bpreshuffle kernel (kernelId 0 won every
        # production shape at ~3e-6 RMS relative error against TE).
        out = torch.empty(lhs[0].shape[0], rhs[0].shape[0], device=lhs[0].device, dtype=out_dtype)
        gemm_a8w8_blockscale_bpreshuffle_tune(
            lhs[0], shuffle_weight(rhs[0], layout=(16, 16)), lhs[1], rhs[1].contiguous(), out, kernelId=0)
        return out
    if backend == "rocm_fp8" and rhs_group_size == 128:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale

        # Logical shape [M, K/128], with column-major physical storage.
        # AITER's unshuffled paths require contiguous row-major 1x128 scales.
        # Our producer keeps a column-major view for TE/local GEMMs, so give
        # AITER a contiguous copy of those same FP32 scale values.
        result = gemm_a8w8_blockscale(lhs[0], rhs[0], lhs[1].contiguous(),
                                    rhs[1], dtype=out_dtype)
        if result.shape != (lhs[0].shape[0], rhs[0].shape[0]) or result.dtype != out_dtype:
            raise RuntimeError("AITER blockscale returned an unexpected shape/dtype")
        return result
    if backend not in {"rocm_fp8", "rocm_fp8_triton"}:
        raise ValueError("unknown AMD FP8 backend")
    return triton_gemm(lhs, rhs, rhs_group_size=rhs_group_size, out_dtype=out_dtype, splits=splits)


def _rows(values, activation_rows):
    codes, scales = activation_rows
    m, k = values.numel() // values.shape[-1], values.shape[-1]
    if (codes.dtype != torch.float8_e4m3fnuz or codes.shape != (m, k) or not codes.is_contiguous()
            or scales.dtype != torch.float32 or scales.shape != (m, k // 128)
            or scales.stride() != (1, m)
            or codes.device != values.device or scales.device != values.device):
        raise ValueError("AMD producer sidecar has incompatible FNUZ codes/scales")
    return codes, scales


def linear(values, weight, *, weight_operands=None, backend="rocm_fp8", splits=1,
           weight_mode="dense", activation_rows=None):
    if values.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise ValueError("AMD FP8 needs BF16 activations and FP32 master weights")
    if weight_mode not in {"dense", "ternary"}:
        raise ValueError("unknown AMD FP8 weight mode")
    if not torch.is_grad_enabled() or not (values.requires_grad or weight.requires_grad):
        x = _rows(values, activation_rows) if activation_rows is not None else quantize_directions(values.reshape(-1, values.shape[-1]))[0]
        w, _ = weight_operands or quantize_weight(weight, weight_mode=weight_mode)
        return gemm(x, w, backend=backend, role="forward", weight_shape=weight.shape).reshape(*values.shape[:-1], weight.shape[0])
    return DenseFP8.apply(values, weight, weight_operands, backend, splits, weight_mode, activation_rows)


class DenseFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, weight_operands, backend, splits, weight_mode, activation_rows):
        flat = values.reshape(-1, values.shape[-1])
        need_x, need_w = ctx.needs_input_grad[:2]
        if activation_rows is None:
            x, xt = quantize_directions(flat, transposed=need_w)
        else:
            x = _rows(values, activation_rows)
            xt = quantize_directions(flat, rowwise=False, transposed=True)[1] if need_w else None
        w, wt = weight_operands or quantize_weight(weight, transposed=need_x,
                                                   weight_mode=weight_mode)
        output = gemm(x, w, backend=backend, role="forward", weight_shape=weight.shape)
        ctx.save_for_backward(*(xt if need_w else (None, None)), *(wt if need_x else (None, None)))
        ctx.input_shape, ctx.weight_shape = values.shape, weight.shape
        ctx.backend, ctx.splits = backend, splits
        return output.reshape(*values.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        xc, xs, wc, ws = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[:2]
        g, gt = quantize_directions(gradient.reshape(-1, ctx.weight_shape[0]),
                                   rowwise=need_x, transposed=need_w, eps=0.0)
        dx = gemm(g, (wc, ws), backend=ctx.backend, role="dgrad",
                  weight_shape=ctx.weight_shape).reshape(ctx.input_shape) if need_x else None
        dw = gemm(gt, (xc, xs), backend=ctx.backend, rhs_group_size=1,
                  out_dtype=torch.float32, splits=ctx.splits, role="wgrad",
                  weight_shape=ctx.weight_shape) if need_w else None
        return dx, dw, None, None, None, None, None
