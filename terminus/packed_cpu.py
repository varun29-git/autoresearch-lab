"""Two-bit ternary storage and bounded-workspace PyTorch CPU inference.

Codes are packed along matrix rows, four weights per byte, least-significant
pair first: 0=-1, 1=0, 2=+1; 3 is invalid. Learned block scales remain FP32.
The portable runtime unpacks output-row tiles for ordinary CPU matrix products.
It does not claim an ISA-specific packed GEMV speedup.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import torch
from torch import nn
from torch.nn import functional as F

from .model import BitNetLlama, BitNetLlamaConfig
from .quantization import quantize_activation_fp8_e4m3, dequantize_activation_fp8_e4m3


def pack_codes(codes):
    if codes.ndim != 2 or not torch.all((codes == -1) | (codes == 0) | (codes == 1)):
        raise ValueError("ternary codes must be a matrix containing -1, 0, +1")
    values = F.pad((codes.to(torch.int16) + 1), (0, (-codes.shape[1]) % 4), value=1)
    values = values.reshape(codes.shape[0], -1, 4)
    return sum(values[..., i] << (2 * i) for i in range(4)).to(torch.uint8)


def unpack_codes(packed, width):
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device)
    values = ((packed.unsqueeze(-1).to(torch.int16) >> shifts) & 3).reshape(packed.shape[0], -1)
    return (values[:, :width] - 1).to(torch.int8)


def cpu_config(config):
    return replace(config, attention_backend="sdpa_reference", local_attention_backend="",
                   linear_backend="torch", fp8_backward="bf16", fp8_quantizer="torch",
                   fp8_cache_weights=False, fp8_wgrad_splits=0, fp8_wgrad_recipe="1d1d",
                   fused_pointwise=False, fused_mlp=False, fused_qkv=False,
                   fused_fp8_producers=False, gradient_checkpointing=False, loss_backend="streaming")


class PackedTernaryLinear(nn.Module):
    def __init__(self, record, *, quantize_activations, tile_rows=128):
        super().__init__()
        if type(tile_rows) is not int or tile_rows <= 0:
            raise ValueError("tile_rows must be positive")
        self.out_features, self.in_features = record["shape"]
        self.alpha_group_size = record["block_size"]
        self.weight_scale_layout = record["layout"]
        if self.weight_scale_layout not in {"blockwise", "rowwise"} or self.alpha_group_size <= 0:
            raise ValueError("invalid packed scale geometry")
        self.quantize_activations = quantize_activations
        self.tile_rows = tile_rows
        packed, scales = record["packed"], record["scales"]
        row_block = self.alpha_group_size if self.weight_scale_layout == "blockwise" else 1
        if (packed.dtype != torch.uint8 or tuple(packed.shape) != (self.out_features, (self.in_features + 3) // 4)
                or scales.dtype != torch.float32 or tuple(scales.shape) !=
                ((self.out_features + row_block - 1) // row_block,
                 (self.in_features + self.alpha_group_size - 1) // self.alpha_group_size)):
            raise ValueError("packed tensor shapes or dtypes are invalid")
        if not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("packed scales must be finite and positive")
        for start in range(0, self.out_features, tile_rows):
            if (unpack_codes(packed[start:start + tile_rows], self.in_features) == 2).any():
                raise ValueError("reserved two-bit code in packed weights")
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)

    def unpack_rows(self, start, end):
        codes = unpack_codes(self.packed[start:end], self.in_features).float()
        row_block = self.alpha_group_size if self.weight_scale_layout == "blockwise" else 1
        indices = torch.arange(start, end) // row_block
        scales = self.scales[indices].repeat_interleave(self.alpha_group_size, dim=1)[:, :self.in_features]
        return codes * scales

    @torch.inference_mode()
    def forward(self, values, activation_rows=None):
        if activation_rows is not None:
            raise ValueError("CPU inference does not accept native FP8 sidecars")
        if values.device.type != "cpu":
            raise ValueError("packed CPU inference requires CPU tensors")
        if self.quantize_activations:
            codes, scales = quantize_activation_fp8_e4m3(values, self.alpha_group_size)
            values = dequantize_activation_fp8_e4m3(codes, scales, self.alpha_group_size, values.dtype)
        chunks = []
        for start in range(0, self.out_features, self.tile_rows):
            weights = self.unpack_rows(start, min(start + self.tile_rows, self.out_features)).to(values.dtype)
            chunks.append(F.linear(values, weights))
        return torch.cat(chunks, dim=-1)


class PackedCPUModel(BitNetLlama):
    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.config.residual_dtype == "bf16"):
            return super().forward(*args, **kwargs)

    @torch.inference_mode()
    def forward_cached(self, *args, **kwargs):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.config.residual_dtype == "bf16"):
            return super().forward_cached(*args, **kwargs)


@torch.no_grad()
def pack_model_state(config, state):
    if config.weight_mode != "ternary" or config.quantization_strength != 1:
        raise ValueError("export requires fully ternary QAT, not a dense checkpoint")
    reference_config = cpu_config(config)
    with torch.device("meta"):
        model = BitNetLlama(reference_config)
    model.load_state_dict(state, strict=True, assign=True)
    matrices, other = {}, dict(state)
    for name, module in model.bitlinear_named_modules():
        if module.quantization_strength.item() != 1:
            raise ValueError("checkpoint has not reached full ternary strength")
        if not torch.isfinite(module.weight).all():
            raise ValueError("nonfinite export weights")
        scales = module.alpha_values()
        # Process one logical matrix at a time; FP32 latent weights live only
        # in the export process, never in the resulting inference bundle.
        matrices[name] = {"shape": list(module.weight.shape), "block_size": module.alpha_group_size,
                          "layout": module.weight_scale_layout,
                          "packed": pack_codes(module.ternary_weight()), "scales": scales.cpu()}
        del other[f"{name}.weight"]
        del other[f"{name}.quantization_strength"]
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in other.values()):
        raise ValueError("nonfinite unquantized export state")
    return {"schema": 1, "format": "terminus-ternary-2bit-lsb-v1", "model": asdict(config),
            "cpu_model": asdict(reference_config), "matrices": matrices, "unquantized": other}


def load_packed_state(bundle, *, tile_rows=128):
    if bundle.get("schema") != 1 or bundle.get("format") != "terminus-ternary-2bit-lsb-v1":
        raise ValueError("unknown packed model format")
    original = BitNetLlamaConfig(**bundle["model"])
    config = cpu_config(original)
    if original.weight_mode != "ternary" or original.quantization_strength != 1 or asdict(config) != bundle["cpu_model"]:
        raise ValueError("packed model configuration mismatch")
    with torch.device("meta"):
        model = PackedCPUModel(config)
    expected = dict(model.bitlinear_named_modules())
    if set(bundle["matrices"]) != set(expected):
        raise ValueError("packed matrix names differ from architecture")
    for name, module in expected.items():
        record = bundle["matrices"][name]
        if (record["shape"] != list(module.weight.shape) or record["block_size"] != module.alpha_group_size
                or record["layout"] != module.weight_scale_layout):
            raise ValueError("packed matrix geometry differs from architecture")
        parent, _, child = name.rpartition(".")
        model.get_submodule(parent).add_module(child, PackedTernaryLinear(
            record, quantize_activations=config.quantize_activations, tile_rows=tile_rows))
    result = model.load_state_dict(bundle["unquantized"], strict=False, assign=True)
    expected_missing = {f"{name}.{buffer}" for name in expected for buffer in ("packed", "scales")}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError("unquantized state differs from architecture")
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("incomplete packed model")
    model.requires_grad_(False)
    return model.eval()
