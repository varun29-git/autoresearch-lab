"""Explicit ROCm attention dispatch; never silently changes the local mask."""
from functools import lru_cache

import torch


@lru_cache(maxsize=2)
def implementation(backend):
    if backend == "rocm_aiter":
        from aiter.ops.mha import flash_attn_func
    elif backend == "rocm_ck":
        from flash_attn import flash_attn_func
    else:
        raise ValueError(f"unknown ROCm attention: {backend}")
    return flash_attn_func


@lru_cache(maxsize=2)
def varlen_implementation(backend):
    # Pinned builds in build_runtime.sh expose this autograd interface.
    if backend == "rocm_aiter":
        from aiter.ops.mha import flash_attn_varlen_func
    elif backend == "rocm_ck":
        from flash_attn import flash_attn_varlen_func
    else:
        raise ValueError(f"unknown ROCm attention: {backend}")
    return flash_attn_varlen_func


def attention(query, key, value, *, backend, local_window=None,
              query_position_offset=0, attention_mask=None, dropout_p=0.0, document_layout=None):
    from .runtime import require_gfx942

    require_gfx942(query.device)
    if attention_mask is not None or dropout_p:
        raise ValueError("ROCm training attention requires packed causal streams and zero dropout")
    if query_position_offset != key.shape[2] - query.shape[2]:
        raise ValueError("ROCm cached attention requires bottom-right causal alignment")
    if any(x.dtype != torch.bfloat16 or x.device != query.device for x in (query, key, value)):
        raise ValueError("ROCm attention requires colocated BF16 Q/K/V")
    window = (-1, -1) if local_window is None else (local_window - 1, 0)
    operands = [x.transpose(1, 2) for x in (query, key, value)]
    operands = [x if x.stride(-1) == 1 else x.contiguous() for x in operands]
    kwargs = dict(causal=True, dropout_p=0.0, window_size=window, deterministic=False)
    if backend == "rocm_aiter":
        # AITER otherwise defaults to RTNA on gfx942. RTNE matches the BF16
        # rounding convention of the PyTorch/FA3 reference more closely.
        kwargs["how_v3_bf16_cvt"] = 0
    if document_layout is not None:
        document_layout.validate_for((query.shape[0], query.shape[2]), query.device)
        if query_position_offset or query.shape[2] != key.shape[2]:
            raise ValueError("packed document attention cannot use an inference cache")
        if not document_layout.spans:
            return query * 0 + (key.sum() + value.sum()) * 0
        indices = document_layout.token_indices
        full = indices.numel() == query.shape[0]*query.shape[2]
        packed = [x.reshape(-1, x.shape[2], x.shape[3]) for x in operands]
        if not full:
            packed = [x.index_select(0, indices) for x in packed]
        result = varlen_implementation(backend)(*packed,
            cu_seqlens_q=document_layout.cu_seqlens, cu_seqlens_k=document_layout.cu_seqlens,
            max_seqlen_q=document_layout.max_segment_length,
            max_seqlen_k=document_layout.max_segment_length, **kwargs)
        if not isinstance(result, torch.Tensor) or result.shape != packed[0].shape:
            raise RuntimeError("unexpected ROCm variable-length attention result; verify pinned build")
        if full:
            return result.reshape_as(operands[0]).transpose(1, 2)
        output = result.new_zeros((query.shape[0]*query.shape[2], query.shape[1], query.shape[3]))
        return output.index_copy(0, indices, result).reshape_as(operands[0]).transpose(1, 2)
    result = implementation(backend)(*operands, **kwargs)
    if not isinstance(result, torch.Tensor) or result.shape != operands[0].shape:
        raise RuntimeError("unexpected ROCm attention API result; verify the pinned build")
    return result.transpose(1, 2)
