"""Hardware migration candidates, with every arithmetic delta enumerated."""
from dataclasses import asdict

from production.recipe import SELECTED_MODEL, SELECTED_OPTIMIZER
from terminus.model import BitNetLlamaConfig
from terminus.optim import OptimizerConfig

CANDIDATES = ("fp8", "fp8_triton", "fp8_te", "bf16")


def model_for(candidate="fp8", *, attention="rocm_aiter", local_attention="",
              wgrad_splits=4, loss_chunk_size=4096):
    if candidate not in CANDIDATES or attention not in {"rocm_aiter", "rocm_ck"} or local_attention not in {"", "rocm_aiter", "rocm_ck"}:
        raise ValueError("unknown AMD linear/attention candidate")
    values = SELECTED_MODEL | {"attention_backend": attention, "local_attention_backend": local_attention,
                               "loss_chunk_size": loss_chunk_size}
    if candidate == "bf16":
        values.update(linear_backend="torch", quantize_activations=False,
                      fp8_backward="bf16", fp8_quantizer="torch", fp8_cache_weights=False)
    else:
        if candidate == "fp8_te":
            wgrad_splits = 0  # TE owns its internal GEMM scheduling.
        values.update(linear_backend="rocm_" + candidate, fp8_wgrad_splits=wgrad_splits)
    return BitNetLlamaConfig(**values)


def candidate_of(model):
    mapping = {"rocm_fp8": "fp8", "rocm_fp8_triton": "fp8_triton", "rocm_fp8_te": "fp8_te", "torch": "bf16"}
    if model.linear_backend not in mapping:
        raise ValueError("not an AMD migration candidate")
    return mapping[model.linear_backend]


def validate_production(model, training, optimizer):
    candidate = candidate_of(model)
    expected = model_for(candidate, attention=model.attention_backend,
                         local_attention=model.local_attention_backend,
                         wgrad_splits=model.fp8_wgrad_splits, loss_chunk_size=model.loss_chunk_size)
    if model != expected or optimizer != OptimizerConfig(**SELECTED_OPTIMIZER):
        raise ValueError("model/optimizer differs from the reviewed AMD migration candidate")
    if training.device != "cuda" or training.dtype != "bf16" or training.development_mode or training.compile or training.cuda_graphs:
        raise ValueError("AMD handoff requires eager HIP BF16 training with native preflight")
    return candidate


def changes(model):
    old, new = asdict(BitNetLlamaConfig(**SELECTED_MODEL)), asdict(model)
    return {k: {"h100": old[k], "amd": v} for k, v in new.items() if v != old[k]}
