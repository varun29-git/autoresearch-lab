"""Dense AMD FP8 autograd with explicit per-operation scaling and FP32 wgrad.

rocm_fp8 uses AITER for 1Dx2D GEMMs and the local Triton 1Dx1D wgrad.
rocm_fp8_triton uses local Triton for all three GEMMs. Both use FNUZ.
rocm_fp8_te uses ROCm Transformer Engine's HipKittens block GEMMs.
"""
import torch


def validate_recipe(config):
    if (config.weight_mode != "dense" or config.fp8_backward != "full"
            or config.fp8_quantizer != "triton_split" or config.weight_scale_layout != "blockwise"
            or config.alpha_group_size != 128 or not config.quantize_activations
            or config.quantization_strength != 1 or config.dim % 128 or config.ffn_dim % 128
            or (config.n_kv_heads * config.head_dim) % 128 or config.fp8_wgrad_recipe != "1d1d"
            or config.fused_fp8_producers):
        raise ValueError("ROCm FP8 requires dense full-backward 128-block scaling; QAT/producer fusion is not certified")
    if config.linear_backend == "rocm_fp8_te" and config.fp8_wgrad_splits:
        raise ValueError("TE owns its split-K scheduling; fp8_wgrad_splits must be zero")


def quantize_directions(values, *, rowwise=True, transposed=False, eps=1e-5):
    from .fp8_kernels import quantize_directions as native

    return native(values, rowwise=rowwise, transposed=transposed, eps=eps)


def quantize_weight(weight, *, transposed=False):
    from .fp8_kernels import quantize

    primary = quantize(weight, group_rows=128)
    # Codes and scales are transposed, not requantized: exact block identity.
    secondary = (primary[0].T.contiguous(), primary[1].T.contiguous()) if transposed else None
    return primary, secondary


def gemm(lhs, rhs, *, backend, rhs_group_size=128, out_dtype=torch.bfloat16, splits=1):
    from .fp8_kernels import check_operands, gemm as triton_gemm

    check_operands(lhs, rhs, rhs_group_size, out_dtype)
    if backend == "rocm_fp8_te":
        from .te_gemm import gemm as te_gemm

        return te_gemm(lhs, rhs, rhs_group_size=rhs_group_size, out_dtype=out_dtype)
    if backend == "rocm_fp8" and rhs_group_size == 128:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale

        # Logical shape [M, K/128], with column-major physical storage.
        result = gemm_a8w8_blockscale(lhs[0], rhs[0], lhs[1],
                                    rhs[1], dtype=out_dtype)
        if result.shape != (lhs[0].shape[0], rhs[0].shape[0]) or result.dtype != out_dtype:
            raise RuntimeError("AITER blockscale returned an unexpected shape/dtype")
        return result
    if backend not in {"rocm_fp8", "rocm_fp8_triton"}:
        raise ValueError("unknown AMD FP8 backend")
    return triton_gemm(lhs, rhs, rhs_group_size=rhs_group_size, out_dtype=out_dtype, splits=splits)


def linear(values, weight, *, weight_operands=None, backend="rocm_fp8", splits=1):
    if values.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise ValueError("AMD dense FP8 needs BF16 activations and FP32 master weights")
    if not torch.is_grad_enabled() or not (values.requires_grad or weight.requires_grad):
        x, _ = quantize_directions(values.reshape(-1, values.shape[-1]))
        w, _ = weight_operands or quantize_weight(weight)
        return gemm(x, w, backend=backend).reshape(*values.shape[:-1], weight.shape[0])
    return DenseFP8.apply(values, weight, weight_operands, backend, splits)


class DenseFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, weight_operands, backend, splits):
        flat = values.reshape(-1, values.shape[-1])
        need_x, need_w = ctx.needs_input_grad[:2]
        x, xt = quantize_directions(flat, transposed=need_w)
        w, wt = weight_operands or quantize_weight(weight, transposed=need_x)
        output = gemm(x, w, backend=backend)
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
        dx = gemm(g, (wc, ws), backend=ctx.backend).reshape(ctx.input_shape) if need_x else None
        dw = gemm(gt, (xc, xs), backend=ctx.backend, rhs_group_size=1,
                  out_dtype=torch.float32, splits=ctx.splits) if need_w else None
        return dx, dw, None, None, None
