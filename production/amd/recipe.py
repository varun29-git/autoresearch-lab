"""Hardware migration candidates, with every arithmetic delta enumerated."""
from dataclasses import asdict

from production.recipe import SELECTED_MODEL, SELECTED_OPTIMIZER
from terminus.model import BitNetLlamaConfig
from terminus.optim import OptimizerConfig

CANDIDATES = ("fp8", "fp8_triton", "fp8_te", "bf16")
# Reviewed promotion from the completed H100 width-transfer confirmation.
# Transfer is already applied: 0.006 * 768 / 2304 = 0.002. No further scaling.
# Keep the historical H100 baseline and the separately qualified QAT policy distinct.
SELECTED_DENSE_OPTIMIZER = SELECTED_OPTIMIZER | {"adamw_lr": 0.002}
SELECTED_QAT_OPTIMIZER = SELECTED_OPTIMIZER | {
    "muon_lr": 0.0085, "adamw_lr": 0.003,
    "norm_lr": 0.00015, "qk_norm_lr": 0.00005,
}


def model_for(candidate="fp8", *, attention="rocm_aiter", local_attention="",
              wgrad_splits=None, loss_chunk_size=4096, weight_mode="dense", fused_fp8_producers=False):
    if candidate not in CANDIDATES or attention not in {"rocm_aiter", "rocm_ck"} or local_attention not in {"", "rocm_aiter", "rocm_ck"}:
        raise ValueError("unknown AMD linear/attention candidate")
    if weight_mode not in {"dense", "ternary"} or (candidate == "bf16" and weight_mode == "ternary"):
        raise ValueError("AMD QAT requires an FP8 candidate and ternary weight mode")
    values = SELECTED_MODEL | {"attention_backend": attention, "local_attention_backend": local_attention,
                               "loss_chunk_size": loss_chunk_size, "weight_mode": weight_mode,
                               "fused_fp8_producers": fused_fp8_producers}
    if candidate == "bf16":
        if wgrad_splits not in (None, 0):
            raise ValueError("BF16 candidate does not use the FP8 split-K setting")
        values.update(linear_backend="torch", quantize_activations=False,
                      fp8_backward="bf16", fp8_quantizer="torch", fp8_cache_weights=False)
    else:
        if candidate == "fp8_te":
            if wgrad_splits not in (None, 0):
                raise ValueError("TE owns split-K scheduling; use no override or zero")
            wgrad_splits = 0  # TE owns its internal GEMM scheduling.
        elif wgrad_splits is None:
            wgrad_splits = 4
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
                         wgrad_splits=model.fp8_wgrad_splits, loss_chunk_size=model.loss_chunk_size,
                         weight_mode=model.weight_mode, fused_fp8_producers=model.fused_fp8_producers)
    selected_optimizer = SELECTED_QAT_OPTIMIZER if model.weight_mode == "ternary" else SELECTED_DENSE_OPTIMIZER
    if model != expected or optimizer != OptimizerConfig(**selected_optimizer):
        raise ValueError("model/optimizer differs from the reviewed AMD migration candidate")
    if training.device != "cuda" or training.dtype != "bf16" or training.development_mode or training.compile or training.cuda_graphs:
        raise ValueError("AMD handoff requires eager HIP BF16 training with native preflight")
    if (training.grad_clip != 1.0 or training.quantization_warmup_tokens
            or training.quantization_start_strength != 1.0):
        raise ValueError("AMD handoff preserves clipping and uses immediate weight quantization")
    if model.weight_mode == "ternary" and not training.phase_start_tokens:
        raise ValueError("AMD QAT requires a phase_start_tokens offset from a dense transition")
    if model.weight_mode == "dense" and training.phase_start_tokens:
        raise ValueError("AMD dense phase cannot use a QAT phase offset")
    return candidate


def changes(model):
    old, new = asdict(BitNetLlamaConfig(**SELECTED_MODEL)), asdict(model)
    return {k: {"h100": old[k], "amd": v} for k, v in new.items() if v != old[k]}
