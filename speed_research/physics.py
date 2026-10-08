"""Optimistic hardware bounds, not achievable-throughput predictions.

FMA counts as two FLOPs. All hardware rates are dense, per GPU, decimal SI.
No torch import, cloud API, or training side effect is needed by this module.
"""
from dataclasses import asdict, dataclass
import math


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return value


@dataclass(frozen=True)
class Shape:
    layers: int = 25
    width: int = 2304
    ffn: int = 5760
    heads: int = 18
    kv_heads: int = 6
    vocab: int = 65536
    sequence: int = 4096
    window: int = 1024
    global_every: int = 5

    def __post_init__(self):
        for key, value in asdict(self).items():
            positive(value, key)
            if not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("head dimensions must divide exactly")
        if self.layers % self.global_every or self.window > self.sequence:
            raise ValueError("invalid hybrid attention geometry")

    @property
    def head_dim(self):
        return self.width // self.heads

    @property
    def kv_width(self):
        return self.kv_heads * self.head_dim

    @property
    def block_matrix_parameters(self):
        return self.layers * (2 * self.width**2 + 2 * self.width * self.kv_width
                              + 3 * self.width * self.ffn)

    @property
    def parameters(self):
        # Four learned channel RMSNorms/block, Q/K gains shared across heads,
        # final RMSNorm, two free value-mixing scalars in every nonzero layer.
        return (self.block_matrix_parameters + self.vocab * self.width
                + self.layers * (4 * self.width + 2 * self.head_dim)
                + self.width + 2 * (self.layers - 1))


# NVIDIA's H100 product table lists starred SPARSE rates. Divide those by two.
HARDWARE = {
    "h100_sxm": {"fp8_tflops": 1979.0, "bf16_tflops": 989.5,
                 "hbm_tb_s": 3.35, "memory_gb": 80},
    "h100_nvl": {"fp8_tflops": 1670.5, "bf16_tflops": 835.5,
                 "hbm_tb_s": 3.9, "memory_gb": 94},
}


def causal_pairs(length, window):
    """Valid query/key pairs, including diagonal; window=None is global."""
    positive(length, "length")
    if not isinstance(length, int):
        raise ValueError("length must be integer")
    if window is None:
        return length * (length + 1) // 2
    positive(window, "window")
    if not isinstance(window, int):
        raise ValueError("window must be integer")
    window = min(window, length)
    return window * (window + 1) // 2 + (length - window) * window


def operation_inventory(shape=Shape()):
    global_layers = shape.layers // shape.global_every
    local_layers = shape.layers - global_layers
    pairs = (global_layers * causal_pairs(shape.sequence, None)
             + local_layers * causal_pairs(shape.sequence, shape.window))
    return {
        "parameters": shape.parameters,
        "block_fp8_flops_per_token": 6 * shape.block_matrix_parameters,
        "head_bf16_flops_per_token": 6 * shape.vocab * shape.width,
        # QK and PV forward, plus both operand gradients for each = 12*d/pair.
        "attention_bf16_flops_per_token": 12 * shape.width * pairs / shape.sequence,
        "attention_valid_pairs_per_sequence_all_layers": pairs,
        "global_layers": global_layers,
        "local_layers": local_layers,
        "logical_projection_shapes_out_in": {
            "q": [shape.width, shape.width], "k_v_each": [shape.kv_width, shape.width],
            "o": [shape.width, shape.width], "gate_up_each": [shape.ffn, shape.width],
            "down": [shape.width, shape.ffn], "tied_head": [shape.vocab, shape.width],
        },
    }


def compute_floor(shape=Shape(), hardware="h100_sxm"):
    """Tensor-core-only lower time bound for the existing mixed precision path.

    FP8 and BF16 share tensor-core resources: their ideal service times add.
    Excludes recomputation (including FA backward score recompute), tile waste,
    softmax/norm/casts, optimizer, HBM, launches and network. Consequently this
    is a loose optimistic bound, NEVER a production throughput estimate.
    """
    inventory = operation_inventory(shape)
    spec = HARDWARE[hardware]
    fp8 = inventory["block_fp8_flops_per_token"]
    bf16 = inventory["head_bf16_flops_per_token"] + inventory["attention_bf16_flops_per_token"]
    seconds = fp8 / (spec["fp8_tflops"] * 1e12) + bf16 / (spec["bf16_tflops"] * 1e12)
    return {**inventory, "hardware": hardware, "ideal_seconds_per_token": seconds,
            "optimistic_tokens_per_second_per_gpu": 1 / seconds,
            "ideal_gpu_hours_1_5t": 1.5e12 * seconds / 3600,
            "ideal_gpu_hours_2t": 2e12 * seconds / 3600}


def budget_targets(*, gpu_hours=6000, gpus=32, unavailable_fraction=0.0):
    positive(gpu_hours, "gpu_hours")
    positive(gpus, "gpus")
    if not isinstance(gpus, int):
        raise ValueError("gpus must be an integer")
    positive(unavailable_fraction, "unavailable_fraction", zero=True)
    if unavailable_fraction >= 1:
        raise ValueError("unavailable_fraction must be less than one")
    useful_seconds = gpu_hours * 3600 * (1 - unavailable_fraction)
    return {
        "production_wall_hours": gpu_hours / gpus,
        "unavailable_fraction": unavailable_fraction,
        "minimum_per_gpu_tps": 1.5e12 / useful_seconds,
        "target_per_gpu_tps": 2e12 / useful_seconds,
        "minimum_cluster_tps": gpus * 1.5e12 / useful_seconds,
        "target_cluster_tps": gpus * 2e12 / useful_seconds,
    }


def projection(*, dense_tps, qat_tps, qat_fraction, scaling_efficiency,
               unavailable_fraction, gpu_hours=6000):
    for key, value in (("dense_tps", dense_tps), ("qat_tps", qat_tps),
                       ("scaling_efficiency", scaling_efficiency), ("gpu_hours", gpu_hours)):
        positive(value, key)
    for key, value in (("qat_fraction", qat_fraction), ("unavailable_fraction", unavailable_fraction)):
        positive(value, key, zero=True)
    if not 0 <= qat_fraction <= 1 or not 0 <= unavailable_fraction < 1 or scaling_efficiency > 1:
        raise ValueError("invalid fraction or scaling efficiency")
    harmonic_tps = 1 / ((1 - qat_fraction) / dense_tps + qat_fraction / qat_tps)
    effective_tps = harmonic_tps * scaling_efficiency * (1 - unavailable_fraction)
    return {"single_gpu_phase_weighted_tps": harmonic_tps,
            "effective_per_gpu_tps": effective_tps,
            "projected_tokens": effective_tps * 3600 * gpu_hours,
            "gpu_hours_for_1_5t": 1.5e12 / effective_tps / 3600,
            "gpu_hours_for_2t": 2e12 / effective_tps / 3600,
            "status": "projection_only_not_a_32_gpu_measurement"}


def recoverable_speedup(fraction, local_speedup):
    positive(fraction, "fraction", zero=True)
    positive(local_speedup, "local_speedup")
    if fraction > 1:
        raise ValueError("fraction exceeds full update")
    return 1 / (1 - fraction + fraction / local_speedup)
