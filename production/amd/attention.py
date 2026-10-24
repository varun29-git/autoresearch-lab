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


def attention(query, key, value, *, backend, local_window=None,
              query_position_offset=0, attention_mask=None, dropout_p=0.0):
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
    result = implementation(backend)(*operands, **kwargs)
    if not isinstance(result, torch.Tensor) or result.shape != operands[0].shape:
        raise RuntimeError("unexpected ROCm attention API result; verify the pinned build")
    return result.transpose(1, 2)
