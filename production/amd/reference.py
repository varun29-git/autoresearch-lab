"""Independent CPU/GPU oracle for the AMD FNUZ operand recipe, never a fast path."""
import torch
import torch.nn.functional as F

FP8 = torch.float8_e4m3fnuz
FP8_MAX = torch.finfo(FP8).max
BLOCK = 128


@torch.no_grad()
def quantize(values, *, group_rows=1, eps=0.0):
    if values.ndim != 2 or min(values.shape) <= 0 or group_rows not in (1, 128):
        raise ValueError("expected a nonempty matrix with 1x128 or 128x128 scaling")
    m, k = values.shape
    padded = F.pad(values.float(), (0, -k % BLOCK, 0, -m % group_rows))
    blocks = padded.reshape(padded.shape[0] // group_rows, group_rows, -1, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(eps)
    scale = torch.where(amax == 0, torch.ones_like(amax),
                        (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny))
    expanded = scale.repeat_interleave(group_rows, 0).repeat_interleave(BLOCK, 1)
    quotient = padded / expanded
    quotient = torch.where(torch.isfinite(quotient), quotient.clamp(-FP8_MAX, FP8_MAX),
                           torch.full_like(quotient, float("nan")))
    return quotient.to(FP8)[:m, :k].contiguous(), scale


@torch.no_grad()
def quantize_ternary(weight):
    """H100 ternary rule with FNUZ operand codes; independent of the Triton kernel."""
    if (weight.ndim != 2 or min(weight.shape) <= 0 or weight.dtype != torch.float32
            or weight.shape[0] % BLOCK or weight.shape[1] % BLOCK):
        raise ValueError("ternary reference needs an aligned FP32 master matrix")
    rows, width = weight.shape
    blocks = weight.reshape(rows // BLOCK, BLOCK, width // BLOCK, BLOCK)
    scales = blocks.abs().mean(dim=(1, 3)).clamp_min(1e-6)
    expanded = scales.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    rounded = (weight / expanded).round().clamp(-1, 1)
    codes = torch.where(torch.isfinite(weight) & torch.isfinite(expanded), rounded,
                        torch.full_like(rounded, float("nan")))
    return codes.to(FP8).contiguous(), scales


def dequantize(operand, group_rows=1):
    codes, scale = operand
    return codes.float() * scale.repeat_interleave(group_rows, 0).repeat_interleave(BLOCK, 1)[:codes.shape[0], :codes.shape[1]]


def directional(values, *, rowwise=True, transposed=False, eps=1e-5):
    rows = quantize(values, eps=eps) if rowwise else None
    columns = quantize(F.pad(values.T, (0, -values.shape[0] % BLOCK)).contiguous(), eps=eps) if transposed else None
    return rows, columns


def gemm(lhs, rhs, *, rhs_group_size=128, out_dtype=torch.bfloat16, **_):
    with torch.autocast(device_type=lhs[0].device.type, enabled=False):
        return (dequantize(lhs) @ dequantize(rhs, rhs_group_size).T).to(out_dtype)
