"""Optional CUDA pointwise fusions; FP32 RMS statistics and gain gradients."""
import torch


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, eps):
        from .pointwise_kernels import norm_forward

        output, inverse, flat = norm_forward(values, weight, eps)
        ctx.save_for_backward(flat, weight, inverse)
        ctx.shape = values.shape
        return output.reshape(values.shape)

    @staticmethod
    def backward(ctx, gradient):
        from .pointwise_kernels import norm_backward

        values, weight, inverse = ctx.saved_tensors
        dx, dw = norm_backward(gradient, values, weight, inverse)
        return dx.reshape(ctx.shape), dw, None


def rms_norm(values, weight, eps):
    if not values.is_cuda:
        raise RuntimeError("fused RMSNorm requires CUDA")
    return _RMSNorm.apply(values, weight, eps)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        from .pointwise_kernels import swiglu_forward

        ctx.save_for_backward(gate, up)
        return swiglu_forward(gate, up)

    @staticmethod
    def backward(ctx, gradient):
        from .pointwise_kernels import swiglu_backward

        return swiglu_backward(gradient, *ctx.saved_tensors)


def swiglu(gate, up):
    if not gate.is_cuda or gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError("fused SwiGLU requires CUDA operands with contiguous columns")
    return _SwiGLU.apply(gate, up)


class _PackedSwiGLUNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, weight, eps):
        from .mlp_kernels import packed_forward

        output, inverse = packed_forward(packed, weight, eps)
        ctx.save_for_backward(packed, weight, inverse)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        from .mlp_kernels import packed_backward

        dx, dw = packed_backward(gradient, *ctx.saved_tensors)
        return dx, dw, None


def packed_swiglu_norm(packed, weight=None, eps=1e-5):
    """Packed gate/up -> SwiGLU -> optional scalar RMSNorm, retaining only packed.

    The packed gradient is written directly in the gate/up projection layout,
    eliminating SplitBackward's concatenation. BF16 boundaries match the
    separate operators, including the pre-norm product and norm input gradient.
    """
    import math

    if not packed.is_cuda or not packed.is_contiguous() or packed.ndim < 2 or packed.shape[-1] % 2:
        raise ValueError("packed SwiGLU needs contiguous CUDA [..., 2 * width] inputs")
    if packed.numel() == 0 or packed.dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("packed SwiGLU needs nonempty BF16/FP32 inputs")
    if weight is not None and (weight.ndim != 0 or weight.dtype != torch.float32 or weight.device != packed.device):
        raise ValueError("packed SwiGLU norm requires a scalar FP32 gain on the input device")
    if not math.isfinite(eps) or eps < 0 or (weight is not None and eps == 0):
        raise ValueError("normalized SwiGLU requires positive finite epsilon")
    return _PackedSwiGLUNorm.apply(packed, weight, eps)


class _QKRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, cos, sin, eps):
        from .pointwise_kernels import qk_rope_forward

        output, inverse = qk_rope_forward(values, cos, sin, eps)
        ctx.save_for_backward(values, cos, sin, inverse)
        return output

    @staticmethod
    def backward(ctx, gradient):
        from .pointwise_kernels import qk_rope_backward

        return qk_rope_backward(gradient, *ctx.saved_tensors), None, None, None


def qk_rope(values, cos, sin, eps):
    return _QKRoPE.apply(values, cos, sin, eps)


class _PackedQKVRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, cos, sin, query_heads, kv_heads, eps, normalize_qk, q_gain, k_gain):
        from .qkv_kernels import packed_qkv_forward

        query, key, value, inverse = packed_qkv_forward(packed, cos, sin, query_heads, kv_heads, eps, normalize_qk,
                                                        q_gain=q_gain, k_gain=k_gain)
        ctx.save_for_backward(packed, cos, sin, inverse, q_gain, k_gain)
        ctx.heads = query_heads, kv_heads
        ctx.normalize_qk = normalize_qk
        return query, key, value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dq, dk, dv):
        from .qkv_kernels import packed_qkv_backward

        packed, cos, sin, inverse, q_gain, k_gain = ctx.saved_tensors
        dx, dq_gain, dk_gain = packed_qkv_backward(dq, dk, dv, packed, cos, sin, inverse, *ctx.heads,
                                                   ctx.normalize_qk, q_gain=q_gain, k_gain=k_gain)
        return dx, None, None, None, None, None, None, dq_gain, dk_gain


def packed_qkv_rope(packed, cos, sin, query_heads, kv_heads, eps, *, normalize_qk=True, q_gain=None, k_gain=None):
    import math

    if not packed.is_cuda or packed.ndim != 3 or not packed.is_contiguous() or packed.numel() == 0:
        raise ValueError("packed QKV requires nonempty contiguous CUDA [batch, length, width]")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in (query_heads, kv_heads)):
        raise ValueError("QKV head counts must be positive integers")
    if query_heads % kv_heads or packed.shape[-1] % (query_heads + 2 * kv_heads):
        raise ValueError("invalid packed QKV head dimensions")
    if not isinstance(normalize_qk, bool):
        raise ValueError("normalize_qk must be boolean")
    if (q_gain is None) != (k_gain is None):
        raise ValueError("Q and K gains must arrive together")
    if q_gain is not None and not normalize_qk:
        raise ValueError("learned gains require Q/K normalization")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("QKV normalization requires positive finite epsilon")
    dim = packed.shape[-1] // (query_heads + 2 * kv_heads)
    if dim not in {64, 128, 256} or packed.dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("packed QKV requires BF16/FP32 and head width 64, 128 or 256")
    if any(t.device != packed.device or t.shape != (1, 1, packed.shape[1], dim // 2) or not t.is_contiguous() for t in (cos, sin)):
        raise ValueError("RoPE tables must be contiguous and match sequence/head dimensions")
    if q_gain is not None:
        if any(g.shape != (dim,) or not g.is_cuda or g.device != packed.device for g in (q_gain, k_gain)):
            raise ValueError("learned Q/K gains must be CUDA channel vectors matching the head width")
    return _PackedQKVRoPE.apply(packed, cos, sin, query_heads, kv_heads, eps, normalize_qk, q_gain, k_gain)


class _ResidualNorms(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, g_o, g_i, eps):
        from .pointwise_kernels import residual_norms_forward

        a, z, r1, r2 = residual_norms_forward(x, u, g_o, g_i, eps)
        ctx.save_for_backward(u, a, g_o, g_i, r1, r2)
        return a, z

    @staticmethod
    def backward(ctx, da, dz):
        from .pointwise_kernels import residual_norms_backward

        du, dx, dgo, dgi = residual_norms_backward(dz, da, *ctx.saved_tensors)
        return dx, du, dgo, dgi, None


def residual_norms(x, u, g_o, g_i, eps):
    """Fused Peri-LN boundary: branch out-norm, residual add, next in-norm.

    a = x + g_o ⊙ rms(u); z = g_i ⊙ rms(a), with the reference kernels' BF16
    rounding boundaries. CUDA-only fusion; callers keep the reference ops
    elsewhere.
    """
    import math

    if not (x.is_cuda and u.is_cuda) or x.shape != u.shape or x.stride(-1) != 1 or u.stride(-1) != 1:
        raise ValueError("residual norm fusion needs CUDA tensors with matching contiguous rows")
    if x.dtype != torch.bfloat16 or u.dtype != torch.bfloat16:
        raise ValueError("residual norm fusion expects the BF16 residual/branch contract")
    if g_o.shape != (x.shape[-1],) or g_i.shape != (x.shape[-1],):
        raise ValueError("channel gains must match the residual width")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("normalization requires positive finite epsilon")
    return _ResidualNorms.apply(x, u, g_o, g_i, eps)
