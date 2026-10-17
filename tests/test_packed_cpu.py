from dataclasses import replace

import pytest
import torch

from terminus.model import BitNetLlama, BitLinear
from terminus.packed_cpu import PackedTernaryLinear, cpu_config, pack_codes, unpack_codes, pack_model_state, load_packed_state
from terminus.quantization import quantize_activation_fp8_e4m3, dequantize_activation_fp8_e4m3
from tests.helpers import tiny_config


def test_two_bit_roundtrip_and_padding():
    codes = torch.tensor([[-1, 0, 1, -1, 1], [0, 1, 0, -1, 0]], dtype=torch.int8)
    packed = pack_codes(codes)
    assert packed.shape == (2, 2)
    assert packed[0, 0] == 36
    torch.testing.assert_close(unpack_codes(packed, 5), codes)
    with pytest.raises(ValueError):
        pack_codes(torch.tensor([[2]]))


@pytest.mark.parametrize("layout", ["rowwise", "blockwise"])
@pytest.mark.parametrize("residual_dtype", ["fp32", "bf16"])
def test_packed_24_layer_forward_and_cached_inference(layout, residual_dtype):
    torch.manual_seed(17)
    config = tiny_config(n_layers=24, global_attention_every=6, local_attention_window=8,
                         max_seq_len=32, branch_norm="peri", value_residual="learned",
                         qk_norm_mode="learned_rms", fuse_projections=True,
                         weight_scale_layout=layout, quantize_activations=True,
                         alpha_group_size=16, residual_dtype=residual_dtype)
    model = BitNetLlama(config).eval()
    bundle = pack_model_state(config, model.state_dict())
    packed = load_packed_state(bundle, tile_rows=128)
    # Independent uncompressed deployment reference. Training's identity STE
    # evaluates latent+(target-latent), introducing cancellation roundoff that
    # can cross a later FP8 rounding boundary. Native ternary operands and this
    # export use target=scales*codes directly.
    for _, layer in model.bitlinear_named_modules():
        weight = layer.alpha_scale() * layer.ternary_weight().float()
        layer.quantized_weight = lambda weight=weight: weight
        def quantized_activation(values, block=layer.alpha_group_size):
            codes, scales = quantize_activation_fp8_e4m3(values, block)
            return dequantize_activation_fp8_e4m3(codes, scales, block, values.dtype), scales
        layer.quantized_activation = quantized_activation
    assert sum(isinstance(m, PackedTernaryLinear) for m in packed.modules()) == 24 * 4
    assert not any("proj.weight" in name for name, _ in packed.named_parameters())
    ids = torch.randint(0, 32, (1, 19))
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=residual_dtype == "bf16"):
        expected = model(ids)["logits"]
    actual = packed(ids)["logits"]
    # FP32 STE evaluation adds/subtracts the latent tensor; packed inference
    # directly uses ternary*scale. BF16 rounds away that small difference.
    torch.testing.assert_close(actual, expected, atol=2e-5 if residual_dtype == "fp32" else .05,
                               rtol=2e-5 if residual_dtype == "fp32" else .03)
    cache, reference_cache, chunks, reference_chunks = None, None, [], []
    for start in range(0, ids.shape[1], 3):
        output, cache = packed.forward_cached(ids[:, start:start + 3], cache=cache)
        chunks.append(output["logits"])
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=residual_dtype == "bf16"):
            reference, reference_cache = model.forward_cached(ids[:, start:start + 3], cache=reference_cache)
        reference_chunks.append(reference["logits"])
    torch.testing.assert_close(torch.cat(chunks, 1), torch.cat(reference_chunks, 1),
                               atol=2e-5 if residual_dtype == "fp32" else .05,
                               rtol=2e-5 if residual_dtype == "fp32" else .03)
    assert [i + 1 for i, layer in enumerate(packed.layers) if layer.attn.is_global_layer] == [6, 12, 18, 24]


def test_cache_matches_full_without_activation_rounding_discontinuities():
    torch.manual_seed(11)
    config = tiny_config(n_layers=24, global_attention_every=6, local_attention_window=4,
                         max_seq_len=32, branch_norm="peri", value_residual="learned",
                         qk_norm_mode="learned_rms", fuse_projections=True)
    model = BitNetLlama(config).eval()
    packed = load_packed_state(pack_model_state(config, model.state_dict()))
    ids = torch.randint(0, 32, (1, 21))
    full = packed(ids)["logits"]
    cache, pieces = None, []
    for start in range(0, 21, 3):
        output, cache = packed.forward_cached(ids[:, start:start + 3], cache=cache)
        pieces.append(output["logits"])
    torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=2e-5, atol=2e-5)


def test_row_tiles_cross_scale_blocks_and_partial_packed_bytes():
    torch.manual_seed(31)
    reference = BitLinear(51, 37, alpha_group_size=16, weight_scale_layout="blockwise")
    target = reference.alpha_scale() * reference.ternary_weight().float()
    record = {"shape": [37, 51], "block_size": 16, "layout": "blockwise",
              "packed": pack_codes(reference.ternary_weight()), "scales": reference.alpha_values()}
    packed = PackedTernaryLinear(record, quantize_activations=False, tile_rows=7)
    torch.testing.assert_close(packed.unpack_rows(7, 21), target[7:21], atol=0, rtol=0)
    values = torch.randn(3, 51)
    torch.testing.assert_close(packed(values), torch.nn.functional.linear(values, target), atol=2e-6, rtol=2e-6)


def test_dense_and_incomplete_qat_export_rejected():
    dense = BitNetLlama(tiny_config(weight_mode="dense"))
    with pytest.raises(ValueError, match="QAT"):
        pack_model_state(dense.config, dense.state_dict())
    qat = BitNetLlama(tiny_config())
    qat.set_quantization_strength_(.5)
    with pytest.raises(ValueError, match="strength"):
        pack_model_state(qat.config, qat.state_dict())


def test_reserved_codes_and_incomplete_bundle_rejected():
    model = BitNetLlama(tiny_config())
    bundle = pack_model_state(model.config, model.state_dict())
    name = next(iter(bundle["matrices"]))
    bundle["matrices"][name]["packed"][0, 0] = 255
    with pytest.raises(ValueError, match="reserved"):
        load_packed_state(bundle)
