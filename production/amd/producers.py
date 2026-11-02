"""Explicit AMD FP8 producer candidate; backward preserves existing BF16 math."""
import torch
from .runtime import require_gfx942


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,weight,eps):
        from .producer_kernels import norm_forward
        out,inverse,flat,fp8=norm_forward(values,weight,eps)
        ctx.save_for_backward(flat,weight,inverse);ctx.shape=values.shape
        ctx.mark_non_differentiable(*fp8)
        return out.reshape(values.shape),*fp8
    @staticmethod
    def backward(ctx,gradient,_codes,_scales):
        from terminus.pointwise_kernels import norm_backward
        values,weight,inverse=ctx.saved_tensors
        dx,dw=norm_backward(gradient,values,weight,inverse)
        return dx.reshape(ctx.shape),dw,None


class _Packed(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,weight,eps):
        from .producer_kernels import packed_forward
        out,inverse,fp8=packed_forward(values,weight,eps)
        ctx.save_for_backward(values,weight,inverse);ctx.mark_non_differentiable(*fp8)
        return out,*fp8
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gradient,_codes,_scales):
        from terminus.mlp_kernels import packed_backward
        dx,dw=packed_backward(gradient,*ctx.saved_tensors)
        return dx,dw,None


def rms_norm_fp8(values,weight,eps):
    require_gfx942(values.device)
    if values.dtype!=torch.bfloat16 or values.ndim<2:raise ValueError('AMD FP8 norm needs BF16 activations')
    return _Norm.apply(values,weight,eps)


def packed_swiglu_norm_fp8(values,weight=None,eps=1e-5):
    require_gfx942(values.device)
    if values.dtype!=torch.bfloat16 or values.ndim<2 or values.shape[-1]%2 or not values.is_contiguous():
        raise ValueError('AMD packed FP8 producer needs contiguous BF16 packed activations')
    return _Packed.apply(values,weight,eps)
