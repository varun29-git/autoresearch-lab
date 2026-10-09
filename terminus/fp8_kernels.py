import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _safe_scale(amplitude, EPS: tl.constexpr):
    amplitude = tl.maximum(amplitude, EPS)
    # Match ATen's division by a scalar (FP32 reciprocal multiply). A one-ulp
    # scale change can flip many exact BF16/FP8 halfway rounding cases.
    return tl.where(amplitude > 0, tl.maximum(amplitude * (1.0 / 448.0), 1.1754943508222875e-38), 1.0)


@triton.jit
def _scaled_values(values, scales, inverse):
    quotient = values * inverse
    # One reciprocal per 128-value group, with an FMA residual correction.
    # This avoids a full IEEE division for every matrix element while keeping
    # FP8 halfway rounding consistent with the reference division.
    residual = tl.fma(-quotient, scales, values)
    return tl.fma(residual, inverse, quotient)


@triton.jit
def _quantize_directions_kernel(values, row_codes, row_scales, column_codes, column_scales,
                                rows: tl.constexpr, width: tl.constexpr,
                                source_row_stride: tl.constexpr, source_column_stride: tl.constexpr,
                                row_scale_stride: tl.constexpr, column_scale_stride: tl.constexpr,
                                padded_rows: tl.constexpr, WRITE_ROWS: tl.constexpr, WRITE_COLUMNS: tl.constexpr,
                                EPS: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_COLUMNS: tl.constexpr):
    # Activation/gradient matrices can exceed 2^31 elements when a
    # checkpointed run uses a large microbatch. Address calculations must not
    # wrap before Triton forms the source or transposed destination pointer.
    row_block = tl.program_id(0).to(tl.int64)
    column_block = tl.program_id(1).to(tl.int64)
    row_indices = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column_indices = column_block * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
    valid = (row_indices[:, None] < rows) & (column_indices[None, :] < width)
    block = tl.load(values + row_indices[:, None] * source_row_stride + column_indices[None, :] * source_column_stride,
                    mask=valid, other=0).to(tl.float32)
    if WRITE_ROWS:
        scale_rows = _safe_scale(tl.max(tl.abs(block), axis=1), EPS)
        inverse_rows = tl.div_rn(1., scale_rows)
        quantized_rows = _scaled_values(block, scale_rows[:, None], inverse_rows[:, None]).to(tl.float8e4nv)
        tl.store(row_codes + row_indices[:, None] * width + column_indices[None, :], quantized_rows, mask=valid)
        tl.store(row_scales + row_indices + column_block * row_scale_stride, scale_rows, mask=row_indices < rows)
    if WRITE_COLUMNS:
        scale_columns = _safe_scale(tl.max(tl.abs(block), axis=0), EPS)
        inverse_columns = tl.div_rn(1., scale_columns)
        quantized_columns = _scaled_values(block, scale_columns[None, :], inverse_columns[None, :]).to(tl.float8e4nv)
        tl.store(column_codes + column_indices[None, :] * padded_rows + row_indices[:, None], quantized_columns,
                 mask=(column_indices[None, :] < width) & (row_indices[:, None] < padded_rows))
        tl.store(column_scales + column_indices + row_block * column_scale_stride,
                 scale_columns, mask=column_indices < width)


@triton.jit
def _quantize_ternary_kernel(weight, codes, scales,
                             rows: tl.constexpr, width: tl.constexpr,
                             source_row_stride: tl.constexpr, source_column_stride: tl.constexpr):
    row_block, column_block = tl.program_id(0), tl.program_id(1)
    row_indices = row_block * 128 + tl.arange(0, 128)
    column_indices = column_block * 128 + tl.arange(0, 128)
    block = tl.load(weight + row_indices[:, None] * source_row_stride + column_indices[None, :] * source_column_stride)
    scale = tl.maximum(tl.sum(tl.sum(tl.abs(block), axis=1), axis=0) / 16384.0, 1e-6)
    ternary = libdevice.nearbyint(tl.div_rn(block, scale))
    ternary = tl.minimum(tl.maximum(ternary, -1.0), 1.0)
    finite = (tl.abs(block) < float("inf")) & (tl.abs(scale) < float("inf"))
    ternary = tl.where(finite, ternary, float("nan")).to(tl.float8e4nv)
    tl.store(codes + row_indices[:, None] * width + column_indices[None, :], ternary)
    tl.store(scales + row_block * (width // 128) + column_block, scale)


@triton.jit
def _quantize_dense_kernel(weight, codes, scales,
                           rows: tl.constexpr, width: tl.constexpr,
                           source_row_stride: tl.constexpr, source_column_stride: tl.constexpr):
    # Dense-weight FP8: per-128x128-block absmax/448 scales, no ternary rounding.
    row_block, column_block = tl.program_id(0), tl.program_id(1)
    row_indices = row_block * 128 + tl.arange(0, 128)
    column_indices = column_block * 128 + tl.arange(0, 128)
    block = tl.load(weight + row_indices[:, None] * source_row_stride + column_indices[None, :] * source_column_stride)
    scale = _safe_scale(tl.max(tl.max(tl.abs(block), axis=1), axis=0), 0.0)
    quotient = tl.div_rn(block, scale)
    finite = (tl.abs(block) < float("inf")) & (tl.abs(scale) < float("inf"))
    dense = tl.where(finite, quotient, float("nan")).to(tl.float8e4nv)
    tl.store(codes + row_indices[:, None] * width + column_indices[None, :], dense)
    tl.store(scales + row_block * (width // 128) + column_block, scale)


@triton.jit
def _quantize_dual_kernel(values, row_codes, row_scales, column_codes, column_scales,
                          rows: tl.constexpr, width: tl.constexpr,
                          source_row_stride: tl.constexpr, source_column_stride: tl.constexpr,
                          row_scale_stride: tl.constexpr, column_scale_stride: tl.constexpr,
                          padded_rows: tl.constexpr, EPS: tl.constexpr,
                          CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    """One read of each 128x128 tile emits both FP8 directions bit-exactly.

    A tile is exactly one rowwise scale group (128 cols per row) and one
    columnwise scale group (128 rows per column). Pass 1 walks tall chunks:
    column groups complete within each chunk (codes+scales stored there) while
    row amaxes accumulate. Pass 2 reloads the L2-resident tile once and emits
    rowwise codes+scales. Register pressure stays off the critical path; the
    second load hits L2 instead of HBM.
    """
    row_block = tl.program_id(0).to(tl.int64)
    column_block = tl.program_id(1).to(tl.int64)
    row_max = tl.zeros((128,), tl.float32)
    for i in tl.static_range(NUM_CHUNKS):
        rr = row_block * 128 + tl.arange(0, 128)
        cc = column_block * 128 + i * CHUNK + tl.arange(0, CHUNK)
        valid = (rr[:, None] < rows) & (cc[None, :] < width)
        block = tl.load(values + rr[:, None] * source_row_stride + cc[None, :] * source_column_stride,
                        mask=valid, other=0).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(tl.abs(block), axis=1))
        scale_columns = _safe_scale(tl.max(tl.abs(block), axis=0), EPS)
        inverse_columns = tl.div_rn(1., scale_columns)
        quantized_columns = _scaled_values(block, scale_columns[None, :], inverse_columns[None, :]).to(tl.float8e4nv)
        tl.store(column_codes + cc[None, :] * padded_rows + rr[:, None], quantized_columns,
                 mask=(cc[None, :] < width) & (rr[:, None] < padded_rows))
        tl.store(column_scales + cc + row_block * column_scale_stride, scale_columns, mask=cc < width)
    row_indices = row_block * 128 + tl.arange(0, 128)
    column_indices = column_block * 128 + tl.arange(0, 128)
    valid = (row_indices[:, None] < rows) & (column_indices[None, :] < width)
    block = tl.load(values + row_indices[:, None] * source_row_stride + column_indices[None, :] * source_column_stride,
                    mask=valid, other=0).to(tl.float32)
    scale_rows = _safe_scale(row_max, EPS)
    inverse_rows = tl.div_rn(1., scale_rows)
    quantized_rows = _scaled_values(block, scale_rows[:, None], inverse_rows[:, None]).to(tl.float8e4nv)
    tl.store(row_codes + row_indices[:, None] * width + column_indices[None, :], quantized_rows, mask=valid)
    tl.store(row_scales + row_indices + column_block * row_scale_stride, scale_rows, mask=row_indices < rows)


@triton.jit
def _quantize_dual_v2_kernel(values, row_codes, row_scales, column_codes, column_scales,
                             rows: tl.constexpr, width: tl.constexpr,
                             source_row_stride: tl.constexpr, source_column_stride: tl.constexpr,
                             row_scale_stride: tl.constexpr, column_scale_stride: tl.constexpr,
                             padded_rows: tl.constexpr, EPS: tl.constexpr):
    # Single pass: hold the tile in BF16 (abs/max are exact in BF16), run the
    # FP32 scale math identically to the reference, and stage the columnwise
    # codes through shared memory so both stores are fully coalesced.
    row_block = tl.program_id(0).to(tl.int64)
    column_block = tl.program_id(1).to(tl.int64)
    row_indices = row_block * 128 + tl.arange(0, 128)
    column_indices = column_block * 128 + tl.arange(0, 128)
    valid = (row_indices[:, None] < rows) & (column_indices[None, :] < width)
    block = tl.load(values + row_indices[:, None] * source_row_stride + column_indices[None, :] * source_column_stride,
                    mask=valid, other=0)
    amplitude = tl.abs(block)
    scale_rows = _safe_scale(tl.max(amplitude, axis=1).to(tl.float32), EPS)
    scale_columns = _safe_scale(tl.max(amplitude, axis=0).to(tl.float32), EPS)
    values32 = block.to(tl.float32)
    inverse_rows = tl.div_rn(1., scale_rows)
    quantized_rows = _scaled_values(values32, scale_rows[:, None], inverse_rows[:, None]).to(tl.float8e4nv)
    tl.store(row_codes + row_indices[:, None] * width + column_indices[None, :], quantized_rows, mask=valid)
    tl.store(row_scales + row_indices + column_block * row_scale_stride, scale_rows, mask=row_indices < rows)
    inverse_columns = tl.div_rn(1., scale_columns)
    quantized_columns = _scaled_values(values32, scale_columns[None, :], inverse_columns[None, :]).to(tl.float8e4nv)
    transposed = tl.trans(quantized_columns)
    tl.store(column_codes + column_indices[:, None] * padded_rows + row_indices[None, :], transposed,
             mask=(column_indices[:, None] < width) & (row_indices[None, :] < padded_rows))
    tl.store(column_scales + column_indices + row_block * column_scale_stride, scale_columns,
             mask=column_indices < width)


def _check_matrix(values, *, weight=False):
    if not values.is_cuda or torch.cuda.get_device_capability(values.device)[0] != 9:
        raise RuntimeError("fused ternary-FP8 kernels require Hopper SM90")
    if values.ndim != 2 or min(values.shape) <= 0 or values.shape[1] % 128:
        raise ValueError("fused quantizers need a nonempty matrix with 128-aligned width")
    if weight and (values.shape[0] % 128 or values.dtype != torch.float32):
        raise ValueError("ternary weights must remain FP32 with 128-aligned dimensions")
    if values.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("fused quantizers accept FP32/BF16 inputs")


def _scales(rows, groups, device):
    stride = triton.cdiv(rows, 4) * 4
    return torch.empty_strided((rows, groups), (1, stride), device=device, dtype=torch.float32)


@torch.no_grad()
def quantize_directions(values, *, rowwise=True, transposed=False, eps=1e-5, separate=False):
    _check_matrix(values)
    if not (rowwise or transposed) or not math.isfinite(eps) or eps < 0:
        raise ValueError("invalid quantization direction or scale floor")
    rows, width = values.shape
    padded_rows = triton.cdiv(rows, 128) * 128
    primary = (torch.empty((rows, width), device=values.device, dtype=torch.float8_e4m3fn),
               _scales(rows, width // 128, values.device)) if rowwise else None
    secondary = (torch.empty((width, padded_rows), device=values.device, dtype=torch.float8_e4m3fn),
                 _scales(width, padded_rows // 128, values.device)) if transposed else None
    block_rows = 128 if transposed else 16
    row_codes, row_scales = primary if rowwise else (values, values)
    column_codes, column_scales = secondary if transposed else (values, values)
    if separate and transposed:
        # Independent shapes avoid the register/shuffle pressure of doing both
        # reductions and an FP8 transpose in a 128x128 CTA. This spends one
        # extra input read when both directions are needed; measure the trade.
        if rowwise:
            _quantize_directions_kernel[(triton.cdiv(rows, 16), width // 128)](
                values, row_codes, row_scales, values, values, rows, width,
                values.stride(0), values.stride(1), row_scales.stride(1), 0,
                padded_rows, True, False, eps, 16, 128,
                num_warps=4, enable_fp_fusion=False)
        _quantize_directions_kernel[(triton.cdiv(rows, 128), width // 32)](
            values, values, values, column_codes, column_scales, rows, width,
            values.stride(0), values.stride(1), 0, column_scales.stride(1),
            padded_rows, False, True, eps, 128, 32,
            num_warps=4, enable_fp_fusion=False)
        if rows != padded_rows:
            # Triton 3.4's transposed store can leave some masked-input lanes
            # unwritten. The padded operand must be zero for native GEMM.
            secondary[0][:, rows:].zero_()
        return primary, secondary
    _quantize_directions_kernel[(triton.cdiv(rows, block_rows), width // 128)](
        values, row_codes, row_scales, column_codes, column_scales, rows, width,
        values.stride(0), values.stride(1), row_scales.stride(1), column_scales.stride(1),
        padded_rows, rowwise, transposed, eps, block_rows, 128,
        num_warps=8 if transposed else 4, enable_fp_fusion=False)
    if transposed and rows != padded_rows:
        secondary[0][:, rows:].zero_()
    return primary, secondary


@torch.no_grad()
def quantize_dual(values, *, eps=1e-5, chunk=32, num_warps=16, variant="single"):
    """Dual rowwise+columnwise activation FP8 quantization in one kernel launch.

    Bit-exact with two separate _quantize_directions_kernel passes (same scale
    math, same FMA residual rounding, same output layouts), minus one full HBM
    read of the operand. variant="single" holds the tile in BF16 with an smem
    transpose; variant="twopass" re-reads the tile from L2 instead.
    """
    _check_matrix(values)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("invalid scale floor")
    if variant not in {"single", "twopass"}:
        raise ValueError("unknown dual-quantizer variant")
    if chunk <= 0 or 128 % chunk:
        raise ValueError("chunk must divide the 128-row group")
    rows, width = values.shape
    padded_rows = triton.cdiv(rows, 128) * 128
    primary = (torch.empty((rows, width), device=values.device, dtype=torch.float8_e4m3fn),
               _scales(rows, width // 128, values.device))
    secondary = (torch.empty((width, padded_rows), device=values.device, dtype=torch.float8_e4m3fn),
                 _scales(width, padded_rows // 128, values.device))
    if variant == "single":
        _quantize_dual_v2_kernel[(triton.cdiv(rows, 128), width // 128)](
            values, primary[0], primary[1], secondary[0], secondary[1], rows, width,
            values.stride(0), values.stride(1), primary[1].stride(1), secondary[1].stride(1),
            padded_rows, eps, num_warps=num_warps, enable_fp_fusion=False)
    else:
        _quantize_dual_kernel[(triton.cdiv(rows, 128), width // 128)](
            values, primary[0], primary[1], secondary[0], secondary[1], rows, width,
            values.stride(0), values.stride(1), primary[1].stride(1), secondary[1].stride(1),
            padded_rows, eps, chunk, 128 // chunk,
            num_warps=num_warps, enable_fp_fusion=False)
    if rows != padded_rows:
        # Same Triton 3.4 transposed-store lane gap as _quantize_directions_kernel;
        # DeepGEMM requires zeroed padding.
        secondary[0][:, rows:].zero_()
    return primary, secondary


@torch.no_grad()
def quantize_ternary(weight, *, transposed=False):
    _check_matrix(weight, weight=True)
    rows, width = weight.shape
    primary = (torch.empty((rows, width), device=weight.device, dtype=torch.float8_e4m3fn),
               torch.empty((rows // 128, width // 128), device=weight.device, dtype=torch.float32))
    _quantize_ternary_kernel[(rows // 128, width // 128)](
        weight, *primary, rows, width, weight.stride(0), weight.stride(1),
        num_warps=8, enable_fp_fusion=False)
    # The combined 128x128 reduction/FP8-transpose store left unwritten lanes
    # on the tested Triton 3.4 SM90 build. Copy the already rounded codes exactly;
    # this is paid once per optimizer update when the versioned cache is used.
    secondary = (primary[0].T.contiguous(), primary[1].T.contiguous()) if transposed else None
    return primary, secondary


@torch.no_grad()
def quantize_dense(weight, *, transposed=False):
    _check_matrix(weight, weight=True)
    rows, width = weight.shape
    primary = (torch.empty((rows, width), device=weight.device, dtype=torch.float8_e4m3fn),
               torch.empty((rows // 128, width // 128), device=weight.device, dtype=torch.float32))
    _quantize_dense_kernel[(rows // 128, width // 128)](
        weight, *primary, rows, width, weight.stride(0), weight.stride(1),
        num_warps=8, enable_fp_fusion=False)
    secondary = (primary[0].T.contiguous(), primary[1].T.contiguous()) if transposed else None
    return primary, secondary
