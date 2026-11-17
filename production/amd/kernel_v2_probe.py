#!/usr/bin/env python
"""Self-contained GPU probe for the v2 kernel candidates. Run on the MI300X pod.

Usage (cluster venv, repo root on PYTHONPATH):

    PYTHONPATH=$PWD python production/amd/kernel_v2_probe.py
    PYTHONPATH=$PWD python production/amd/kernel_v2_probe.py --kernels both,columns --iters 20

For each kernel it builds production-shape seeded inputs on cuda, runs the
current implementation and every v2 variant, checks outputs bitwise (or
reports max relative error for the reduction-order-sensitive kernels), and
times each with CUDA events (median of --iters after --warmup). One JSON line
per case: {kernel, variant, shape, bit_exact, max_rel_err, baseline_ms,
variant_ms}. Baselines are also emitted (variant="baseline").

Numerics policy: quantize_both / quantize_columns and the gathers must be
bit-exact. The norm/swiglu/rope/gain kernels keep FP32 accumulation but change
reduction order, so they are graded on max_rel_err (must stay < 1e-5).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import triton

FP8 = torch.float8_e4m3fnuz


# ---------------------------------------------------------------------------
# compare / timing helpers


def _bit_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        return bool((a.view(torch.uint8) == b.view(torch.uint8)).all().item())
    except (RuntimeError, TypeError):
        af, bf = a.float(), b.float()
        return bool(((af == bf) | (af.isnan() & bf.isnan())).all().item())


def _rel_err(a, b):
    if not a.is_floating_point():
        return 0.0 if bool((a == b).all().item()) else float("inf")
    # fp8 -> float64 cast is not implemented everywhere; go through float32.
    af, bf = a.float(), b.float()
    both_nan = af.isnan() & bf.isnan()
    diff = (af - bf).abs().masked_fill(both_nan, 0.0)
    denom = bf.abs().max().clamp_min(1e-30)
    return float(diff.max() / denom)


def _flatten(out):
    if out is None:
        return []
    if isinstance(out, torch.Tensor):
        return [out]
    tensors = []
    for item in out:
        tensors.extend(_flatten(item))
    return tensors


def compare(reference, candidate):
    ref, got = _flatten(reference), _flatten(candidate)
    if len(ref) != len(got):
        return False, float("inf")
    bit = True
    rel = 0.0
    for a, b in zip(ref, got):
        bit = bit and _bit_equal(a, b)
        rel = max(rel, _rel_err(a, b))
    return bit, rel


def time_median(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def run_case(kernel, shape, baseline_fn, variants, iters, warmup):
    reference = baseline_fn()
    baseline_ms = time_median(baseline_fn, iters, warmup)
    print(json.dumps({"kernel": kernel, "variant": "baseline", "shape": shape,
                      "bit_exact": True, "max_rel_err": 0.0,
                      "baseline_ms": baseline_ms, "variant_ms": baseline_ms}), flush=True)
    for name, fn in variants:
        candidate = fn()
        bit, rel = compare(reference, candidate)
        variant_ms = time_median(fn, iters, warmup)
        print(json.dumps({"kernel": kernel, "variant": name, "shape": shape,
                          "bit_exact": bit, "max_rel_err": rel,
                          "baseline_ms": baseline_ms, "variant_ms": variant_ms}), flush=True)


# ---------------------------------------------------------------------------
# production/amd FP8 quantizers (bit-exactness required)


def ref_both(x, num_warps):
    from production.amd.fp8_kernels import _both, FP8_MAX
    m, k = x.shape
    padded = triton.cdiv(m, 128) * 128
    row_codes = torch.empty((m, k), device=x.device, dtype=FP8)
    row_scales = torch.empty((k // 128, m), device=x.device, dtype=torch.float32).T
    col_codes = torch.empty((k, padded), device=x.device, dtype=FP8)
    col_scales = torch.empty((padded // 128, k), device=x.device, dtype=torch.float32)
    _both[(padded // 128, k // 128)](x, row_codes, row_scales, col_codes, col_scales, m, k, padded,
                                    *x.stride(), *row_scales.stride(), 1e-5, FP8_MAX,
                                    num_warps=num_warps, enable_fp_fusion=False)
    return (row_codes, row_scales), (col_codes, col_scales.T)


def ref_columns(x, num_warps):
    from production.amd.fp8_kernels import _columns, FP8_MAX
    m, k = x.shape
    padded = triton.cdiv(m, 128) * 128
    codes = torch.empty((k, padded), device=x.device, dtype=FP8)
    scales = torch.empty((padded // 128, k), device=x.device, dtype=torch.float32)
    _columns[(padded // 128, triton.cdiv(k, 32))](x, codes, scales, m, k, padded, *x.stride(),
                                                  1e-5, FP8_MAX, num_warps=num_warps,
                                                  enable_fp_fusion=False)
    return codes, scales.T


def case_both(iters, warmup):
    from production.amd.fp8_kernels import quantize_both
    from production.amd.fp8_kernels_v2 import quantize_both_v2

    shapes = [((65536, 2304), torch.bfloat16), ((65536, 3840), torch.bfloat16),
              ((3840, 2304), torch.bfloat16), ((11520, 2304), torch.bfloat16),
              ((2304, 5760), torch.bfloat16), ((2304, 2304), torch.bfloat16),
              ((4000, 2304), torch.bfloat16),   # row-padding path
              ((3840, 2304), torch.float32)]    # FP32 master-weight path
    for (m, k), dtype in shapes:
        torch.manual_seed(1234)
        x = torch.randn(m, k, device="cuda", dtype=torch.float32).to(dtype)
        variants = [
            ("ref_w4", lambda: ref_both(x, 4)),
            ("ref_w16", lambda: ref_both(x, 16)),
            ("strip_c32_w4", lambda: quantize_both_v2(x, variant="strip", chunk=32, num_warps=4)),
            ("strip_c32_w8", lambda: quantize_both_v2(x, variant="strip", chunk=32, num_warps=8)),
            ("strip_c64_w4", lambda: quantize_both_v2(x, variant="strip", chunk=64, num_warps=4)),
            ("strip_c64_w8", lambda: quantize_both_v2(x, variant="strip", chunk=64, num_warps=8)),
            ("hold_w4", lambda: quantize_both_v2(x, variant="hold", num_warps=4)),
            ("hold_w8", lambda: quantize_both_v2(x, variant="hold", num_warps=8)),
            ("hold_w16", lambda: quantize_both_v2(x, variant="hold", num_warps=16)),
            ("persist_c32_w8_p2048", lambda: quantize_both_v2(x, variant="persistent", chunk=32,
                                                              num_warps=8, programs=2048)),
            ("persist_c64_w8_p4096", lambda: quantize_both_v2(x, variant="persistent", chunk=64,
                                                              num_warps=8, programs=4096)),
        ]
        run_case("quantize_both", f"{m}x{k}:{str(dtype).split('.')[-1]}",
                 lambda: quantize_both(x), variants, iters, warmup)
        del x
        torch.cuda.empty_cache()


def case_columns(iters, warmup):
    from production.amd.fp8_kernels import quantize_columns
    from production.amd.fp8_kernels_v2 import quantize_columns_v2

    for m, k in [(65536, 2304), (65536, 3840), (4000, 2304)]:
        torch.manual_seed(1234)
        x = torch.randn(m, k, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        variants = [
            ("ref_w8", lambda: ref_columns(x, 8)),
            ("v2_bc64_t_w4", lambda: quantize_columns_v2(x, bc=64, trans_store=True, num_warps=4)),
            ("v2_bc128_t_w4", lambda: quantize_columns_v2(x, bc=128, trans_store=True, num_warps=4)),
            ("v2_bc128_t_w8", lambda: quantize_columns_v2(x, bc=128, trans_store=True, num_warps=8)),
            ("v2_bc128_s_w8", lambda: quantize_columns_v2(x, bc=128, trans_store=False, num_warps=8)),
            ("v2_bc64_t_w8", lambda: quantize_columns_v2(x, bc=64, trans_store=True, num_warps=8)),
        ]
        run_case("quantize_columns", f"{m}x{k}:bfloat16", lambda: quantize_columns(x),
                 variants, iters, warmup)
        del x
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# packed QKV rope + learned-gain backward (drift-graded)


def _rope_inputs(batched):
    torch.manual_seed(1234)
    q, kv, dim, batch, length = 18, 6, 128, 16, 4096
    packed = torch.randn(batch, length, (q + 2 * kv) * dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    tables = batch if batched else 1  # batched rope: one (1, L, D/2) table per document stream
    cos = torch.randn(tables, 1, length, dim // 2, device="cuda", dtype=torch.float32).to(torch.bfloat16).contiguous()
    sin = torch.randn(tables, 1, length, dim // 2, device="cuda", dtype=torch.float32).to(torch.bfloat16).contiguous()
    q_gain = torch.randn(dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    k_gain = torch.randn(dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    return packed, cos, sin, q_gain, k_gain, q, kv, dim, batch, length


def case_rope_fwd(iters, warmup):
    from terminus.qkv_kernels import packed_qkv_forward
    from terminus.qkv_kernels_v2 import packed_qkv_forward_v2

    for batched in (True, False):
        packed, cos, sin, q_gain, k_gain, q, kv, dim, batch, length = _rope_inputs(batched)
        variants = [
            (f"v2_p{p}_w{w}", (lambda p=p, w=w: packed_qkv_forward_v2(
                packed, cos, sin, q, kv, 1e-5, True, q_gain, k_gain, pairs=p, num_warps=w)))
            for p in (4, 8, 16) for w in (2, 4)
        ]
        run_case("packed_rope_fwd", f"16x4096x3840:batched={batched}",
                 lambda: packed_qkv_forward(packed, cos, sin, q, kv, 1e-5, True, q_gain, k_gain),
                 variants, iters, warmup)
        del packed, cos, sin
        torch.cuda.empty_cache()


def case_rope_bwd(iters, warmup):
    from terminus.qkv_kernels import _packed_rope, packed_qkv_forward
    from terminus.qkv_kernels_v2 import packed_qkv_backward_rope_v2

    packed, cos, sin, q_gain, k_gain, q, kv, dim, batch, length = _rope_inputs(True)
    _, _, _, inverse = packed_qkv_forward(packed, cos, sin, q, kv, 1e-5, True, q_gain, k_gain)
    torch.manual_seed(5678)
    dq = torch.randn(batch, q, length, dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    dk = torch.randn(batch, kv, length, dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    dv = torch.randn(batch, kv, length, dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    rows = batch * length

    def baseline():
        output = torch.empty_like(packed)
        _packed_rope[(rows, triton.cdiv(q + 2 * kv, 4))](
            packed, cos, sin, inverse, None, dq, dk, dv, output, q_gain, k_gain,
            q, kv, dim, length, *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3], 0., True,
            4, True, True, True, num_warps=4, enable_fp_fusion=False)
        return output

    variants = [
        (f"v2_p{p}_w{w}", (lambda p=p, w=w: packed_qkv_backward_rope_v2(
            dq, dk, dv, packed, cos, sin, inverse, q, kv, True, q_gain, k_gain, pairs=p, num_warps=w)))
        for p in (4, 8, 16) for w in (4, 8)
    ]
    run_case("packed_rope_bwd", "16x4096x3840:batched=True", baseline, variants, iters, warmup)
    del packed, cos, sin, dq, dk, dv
    torch.cuda.empty_cache()


def case_gain(iters, warmup):
    from terminus.qkv_kernels import _packed_gain_tiled, packed_qkv_forward
    from terminus.qkv_kernels_v2 import tiled_gain_backward_v2

    packed, cos, sin, q_gain, k_gain, q, kv, dim, batch, length = _rope_inputs(True)
    _, _, _, inverse = packed_qkv_forward(packed, cos, sin, q, kv, 1e-5, True, q_gain, k_gain)
    torch.manual_seed(5678)
    dq = torch.randn(batch, q, length, dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    dk = torch.randn(batch, kv, length, dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    def baseline():
        gradients = []
        for heads, start, gradient in ((q, 0, dq), (kv, q, dk)):
            programs = triton.cdiv(batch * length * heads, 64)
            partial = torch.empty((programs, dim), device=packed.device, dtype=torch.float32)
            _packed_gain_tiled[(programs,)](
                packed, cos, sin, inverse, gradient, partial, q, kv, dim, length,
                *gradient.stride()[:3], batch * length, heads, start, True, 64,
                num_warps=4, enable_fp_fusion=False)
            gradients.append(partial.sum(0))
        return gradients

    variants = [
        (f"v2_t{t}_w{w}", (lambda t=t, w=w: tiled_gain_backward_v2(
            dq, dk, packed, cos, sin, inverse, q, kv, tile=t, num_warps=w)))
        for t in (64, 128, 256) for w in (4, 8)
    ]
    run_case("packed_gain_tiled", "16x4096:q18+kv6", baseline, variants, iters, warmup)
    del packed, cos, sin, dq, dk
    torch.cuda.empty_cache()


def case_gain_serial(iters, warmup):
    from terminus.qkv_kernels import _packed_gain_backward, packed_qkv_forward

    packed, cos, sin, q_gain, k_gain, q, kv, dim, batch, length = _rope_inputs(True)
    _, _, _, inverse = packed_qkv_forward(packed, cos, sin, q, kv, 1e-5, True, q_gain, k_gain)
    torch.manual_seed(5678)
    dq = torch.randn(batch, q, length, dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    dk = torch.randn(batch, kv, length, dim, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    rows = batch * length

    def launch(programs, warps):
        partial = torch.empty((programs, 2 * dim), device=packed.device, dtype=torch.float32)
        _packed_gain_backward[(programs,)](packed, cos, sin, inverse, dq, dk, q_gain, k_gain, partial,
                                           q, kv, dim, length, *dq.stride()[:3], *dk.stride()[:3],
                                           rows, programs, True, num_warps=warps,
                                           enable_fp_fusion=False)
        reduced = partial.sum(0)
        return reduced[:dim], reduced[dim:]

    variants = [(f"ref_p{p}", (lambda p=p: launch(p, 4))) for p in (1024, 2048, 4096, 8192)]
    variants.append(("ref_p2048_w8", lambda: launch(2048, 8)))
    run_case("packed_gain_serial", "16x4096:q18+kv6", lambda: launch(512, 4), variants, iters, warmup)
    del packed, cos, sin, dq, dk
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# MLP / norm pointwise kernels (drift-graded except flat swiglu)


def case_swiglu_packed(iters, warmup):
    from terminus.mlp_kernels import packed_backward, packed_forward
    from terminus.mlp_kernels_v2 import packed_backward_v2, packed_forward_v2

    torch.manual_seed(1234)
    packed = torch.randn(65536, 11520, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn((), device="cuda", dtype=torch.float32)
    _, inverse, _ = packed_forward(packed, weight, 1e-5)
    torch.manual_seed(5678)
    dy = torch.randn(65536, 5760, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    fwd_variants = [
        (f"v2_c{c}_w{w}", (lambda c=c, w=w: packed_forward_v2(packed, weight, 1e-5, chunk=c, num_warps=w)))
        for c in (1024, 2048) for w in (4, 8)
    ]
    run_case("packed_swiglu_fwd", "65536x11520:norm", lambda: packed_forward(packed, weight, 1e-5),
             fwd_variants, iters, warmup)
    bwd_variants = [
        (f"v2_c{c}_w{w}", (lambda c=c, w=w: packed_backward_v2(dy, packed, weight, inverse, chunk=c, num_warps=w)))
        for c in (1024, 2048) for w in (4, 8)
    ]
    run_case("packed_swiglu_bwd", "65536x11520:norm", lambda: packed_backward(dy, packed, weight, inverse),
             bwd_variants, iters, warmup)
    plain_variants = [
        (f"v2_c{c}_w{w}", (lambda c=c, w=w: packed_backward_v2(dy, packed, None, None, chunk=c, num_warps=w)))
        for c in (1024, 2048) for w in (4, 8)
    ]
    run_case("packed_swiglu_bwd", "65536x11520:plain", lambda: packed_backward(dy, packed, None, None),
             plain_variants, iters, warmup)
    del packed, dy
    torch.cuda.empty_cache()


def case_swiglu_flat(iters, warmup):
    from terminus.pointwise_kernels import swiglu_backward, swiglu_forward
    from terminus.pointwise_kernels_v2 import swiglu_backward_v2, swiglu_forward_v2

    torch.manual_seed(1234)
    gate = torch.randn(65536, 5760, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    up = torch.randn(65536, 5760, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    dy = torch.randn(65536, 5760, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    fwd_variants = [
        (f"v2_bc{b}_w{w}", (lambda b=b, w=w: swiglu_forward_v2(gate, up, bc=b, num_warps=w)))
        for b in (512, 1024) for w in (4, 8)
    ]
    run_case("swiglu_flat_fwd", "65536x5760", lambda: swiglu_forward(gate, up), fwd_variants, iters, warmup)
    bwd_variants = [
        (f"v2_bc{b}_w{w}", (lambda b=b, w=w: swiglu_backward_v2(dy, gate, up, bc=b, num_warps=w)))
        for b in (512, 1024) for w in (4, 8)
    ]
    run_case("swiglu_flat_bwd", "65536x5760", lambda: swiglu_backward(dy, gate, up), bwd_variants, iters, warmup)
    del gate, up, dy
    torch.cuda.empty_cache()


def case_norm(iters, warmup):
    from terminus.pointwise_kernels import norm_backward, norm_forward
    from terminus.pointwise_kernels_v2 import norm_backward_v2, norm_forward_v2

    torch.manual_seed(1234)
    x = torch.randn(65536, 2304, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(2304, device="cuda", dtype=torch.float32)
    _, inverse, flat, _ = norm_forward(x, weight, 1e-5)
    dy = torch.randn(65536, 2304, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    fwd_variants = [
        (f"v2_w{w}", (lambda w=w: norm_forward_v2(x, weight, 1e-5, num_warps=w)))
        for w in (2, 4, 8)
    ]
    run_case("norm_forward", "65536x2304:gain2", lambda: norm_forward(x, weight, 1e-5),
             fwd_variants, iters, warmup)
    bwd_variants = [
        ("ref_p2048", lambda: norm_backward(dy, flat, weight, inverse, programs_override=2048)),
        ("ref_p8192", lambda: norm_backward(dy, flat, weight, inverse, programs_override=8192)),
    ] + [
        (f"v2_p{p}_w{w}", (lambda p=p, w=w: norm_backward_v2(dy, flat, weight, inverse,
                                                             programs=p, num_warps=w)))
        for p in (512, 2048, 8192) for w in (4, 8)
    ]
    run_case("norm_backward", "65536x2304:gain2", lambda: norm_backward(dy, flat, weight, inverse),
             bwd_variants, iters, warmup)
    del x, dy, flat
    torch.cuda.empty_cache()


def case_producer_norm(iters, warmup):
    from production.amd.producer_kernels import norm_forward
    from production.amd.producer_kernels_v2 import norm_forward_v2

    torch.manual_seed(1234)
    x = torch.randn(65536, 2304, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(2304, device="cuda", dtype=torch.float32)
    variants = [
        (f"v2_c{c}_w{w}", (lambda c=c, w=w: norm_forward_v2(x, weight, 1e-5, chunk=c, num_warps=w)))
        for c in (1024, 2048) for w in (2, 4, 8)
    ]
    run_case("producer_norm_fwd", "65536x2304:gain2+fnuz", lambda: norm_forward(x, weight, 1e-5),
             variants, iters, warmup)
    del x
    torch.cuda.empty_cache()


def case_producer_swiglu(iters, warmup):
    from production.amd.producer_kernels import packed_forward
    from production.amd.producer_kernels_v2 import packed_forward_v2

    torch.manual_seed(1234)
    packed = torch.randn(65536, 11520, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn((), device="cuda", dtype=torch.float32)
    variants = [
        (f"v2_c{c}_w{w}", (lambda c=c, w=w: packed_forward_v2(packed, weight, 1e-5, chunk=c, num_warps=w)))
        for c in (1024, 2048) for w in (4, 8)
    ]
    run_case("producer_swiglu_fwd", "65536x11520:norm+fnuz", lambda: packed_forward(packed, weight, 1e-5),
             variants, iters, warmup)
    del packed
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# aten gather replacements (bit-exact)


def case_embed(iters, warmup):
    import torch.nn.functional as F
    from terminus.gather_kernels_v2 import embed_gather_v2

    torch.manual_seed(1234)
    ids = torch.randint(0, 65536, (16, 4096), device="cuda", dtype=torch.int64)
    for dtype in (torch.float32, torch.bfloat16):
        weight = torch.randn(65536, 2304, device="cuda", dtype=torch.float32).to(dtype)
        variants = [
            ("v2_r1_bc256_w4", lambda: embed_gather_v2(ids, weight, rows_per_cta=1, bc=256, num_warps=4)),
            ("v2_r1_bc512_w4", lambda: embed_gather_v2(ids, weight, rows_per_cta=1, bc=512, num_warps=4)),
            ("v2_r1_bc1024_w4", lambda: embed_gather_v2(ids, weight, rows_per_cta=1, bc=1024, num_warps=4)),
            ("v2_r4_bc512_w4", lambda: embed_gather_v2(ids, weight, rows_per_cta=4, bc=512, num_warps=4)),
            ("v2_r4_bc512_w8", lambda: embed_gather_v2(ids, weight, rows_per_cta=4, bc=512, num_warps=8)),
            ("v2_r8_bc256_w4", lambda: embed_gather_v2(ids, weight, rows_per_cta=8, bc=256, num_warps=4)),
        ]
        run_case("embed_gather", f"65536->65536x2304:{str(dtype).split('.')[-1]}",
                 lambda: F.embedding(ids, weight).to(torch.bfloat16), variants, iters, warmup)
        del weight
        torch.cuda.empty_cache()


def case_rope_gather(iters, warmup):
    from terminus.gather_kernels_v2 import rope_gather_v2

    torch.manual_seed(1234)
    cos = torch.randn(1, 1, 4096, 64, device="cuda", dtype=torch.float32).to(torch.bfloat16).contiguous()
    sin = torch.randn(1, 1, 4096, 64, device="cuda", dtype=torch.float32).to(torch.bfloat16).contiguous()
    positions = torch.randint(0, 4096, (16, 4096), device="cuda", dtype=torch.int64)

    def baseline():
        return (cos[0, 0][positions].unsqueeze(1).contiguous(),
                sin[0, 0][positions].unsqueeze(1).contiguous())

    def expected():
        return cos[0, 0][positions], sin[0, 0][positions]

    variants = [
        (f"v2_r{r}_w{w}", (lambda r=r, w=w: rope_gather_v2(positions, cos, sin,
                                                           rows_per_cta=r, num_warps=w)))
        for r in (8, 16, 32) for w in (1, 2, 4)
    ]
    # rope_gather_v2 returns (16, 4096, 64) rows; the baseline unsqueezes a
    # singleton head axis — compare against the pre-unsqueeze gather.
    reference = expected()
    baseline_ms = time_median(baseline, iters, warmup)
    print(json.dumps({"kernel": "rope_gather", "variant": "baseline", "shape": "65536->4096x64",
                      "bit_exact": True, "max_rel_err": 0.0,
                      "baseline_ms": baseline_ms, "variant_ms": baseline_ms}), flush=True)
    for name, fn in variants:
        bit, rel = compare(reference, fn())
        variant_ms = time_median(fn, iters, warmup)
        print(json.dumps({"kernel": "rope_gather", "variant": name, "shape": "65536->4096x64",
                          "bit_exact": bit, "max_rel_err": rel,
                          "baseline_ms": baseline_ms, "variant_ms": variant_ms}), flush=True)
    del cos, sin, positions
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------

CASES = {
    "both": case_both,
    "columns": case_columns,
    "rope_fwd": case_rope_fwd,
    "rope_bwd": case_rope_bwd,
    "gain": case_gain,
    "gain_serial": case_gain_serial,
    "swiglu_packed": case_swiglu_packed,
    "swiglu_flat": case_swiglu_flat,
    "norm": case_norm,
    "producer_norm": case_producer_norm,
    "producer_swiglu": case_producer_swiglu,
    "embed": case_embed,
    "rope_gather": case_rope_gather,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", default=",".join(CASES),
                        help=f"comma-separated subset of: {','.join(CASES)}")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA device required; run this probe on the GPU pod"}))
        return 1
    print(json.dumps({"device": torch.cuda.get_device_name(0),
                      "hip": str(torch.version.hip), "torch": torch.__version__}), flush=True)
    selected = [name.strip() for name in args.kernels.split(",") if name.strip()]
    unknown = [name for name in selected if name not in CASES]
    if unknown:
        print(json.dumps({"error": f"unknown kernels {unknown}; choose from {sorted(CASES)}"}))
        return 2
    for name in selected:
        CASES[name](args.iters, args.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
