"""Pinned ROCm TE block GEMM adapter over our already-quantized FNUZ operands.

No TE Linear modules, alternate weight parameters, delayed scaling, power-of-two
scale rounding, or TE optimizer are introduced. Native validation is mandatory.
"""
import torch


def gemm(lhs, rhs, *, rhs_group_size=128, out_dtype=torch.bfloat16):
    # CDNA3 TE tiles N by 256. A 128-column tail selects a much slower
    # partial-N specialization for every tile, including the full tiles.
    # Preserve the quantized operands while avoiding that specialization.
    m, n = lhs[0].shape[0], rhs[0].shape[0]
    if n % 256 == 128:
        if rhs_group_size == 1 and m % 256 == 0:
            # 1Dx1D scaling is symmetric. Exchange operands and transpose the
            # FP32 result; no operand requantization or reduction split occurs.
            return _gemm(rhs, lhs, rhs_group_size=1, out_dtype=out_dtype).T.contiguous()
        codes, scales = rhs
        padded_codes = torch.zeros((n + 128, codes.shape[1]), device=codes.device,
                                   dtype=torch.uint8)
        padded_codes[:n].copy_(codes.view(torch.uint8))
        padded_scales = torch.ones(((n + 128) // rhs_group_size, scales.shape[1]),
                                   device=scales.device, dtype=scales.dtype)
        padded_scales[:scales.shape[0]].copy_(scales)
        padded_rhs = padded_codes.view(codes.dtype), padded_scales
        return _gemm(lhs, padded_rhs, rhs_group_size=rhs_group_size,
                     out_dtype=out_dtype)[:, :n].contiguous()
    return _gemm(lhs, rhs, rhs_group_size=rhs_group_size, out_dtype=out_dtype)


def _gemm(lhs, rhs, *, rhs_group_size=128, out_dtype=torch.bfloat16):
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
    from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
    import transformer_engine_torch as tex

    def wrap(operand, group):
        codes, scales = operand
        # ROCm TE v2.17: 1D scales stored as [K/128, M]; 2D as [N/128,K/128].
        physical_scales = scales.T.contiguous() if group == 1 else scales.contiguous()
        return Float8BlockwiseQTensorStorage(
            rowwise_data=codes.view(torch.uint8), rowwise_scale_inv=physical_scales,
            columnwise_data=None, columnwise_scale_inv=None,
            fp8_dtype=tex.DType.kFloat8E4M3, quantizer=None,
            is_2D_scaled=group == 128, fake_dtype=torch.bfloat16)

    result = general_gemm(wrap(rhs, rhs_group_size), wrap(lhs, 1), out_dtype=out_dtype,
                          layout="TN", use_split_accumulator=True)[0]
    if result.shape != (lhs[0].shape[0], rhs[0].shape[0]) or result.dtype != out_dtype:
        raise RuntimeError("ROCm TE returned an unexpected shape/dtype")
    return result
