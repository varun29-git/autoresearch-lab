"""Native AMD kernel/composition gates. A passing receipt is not a speed claim."""
import argparse
from dataclasses import asdict, replace
import json

import torch

from production.artifacts import external_path
from terminus.fp8_validation import projection_shapes, relative_error
from terminus.integrity import fingerprint, source_hashes, write_json_atomic
from terminus.kernels import attention, sdpa_reference_attention
from terminus.model import BitNetLlama, BitLinear, RMSNorm, apply_rotary
from terminus.optim import build_optimizer, OptimizerConfig
from terminus.preflight import count_model, model_numerical_preflight, optimized_kernel_preflight
from terminus.train import load_config, runtime_contract
from . import backend, reference
from .recipe import SELECTED_DENSE_OPTIMIZER, SELECTED_QAT_OPTIMIZER, changes
from .runtime import require_gfx942


def identity(config, device):
    return {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device)}


def validate_report(path, config, device):
    if path is None:
        raise ValueError("AMD native preflight receipt is required")
    report = json.loads(external_path(path).read_text())
    if (report.get("kind") != "ci1_amd_preflight" or report.get("schema") != 1
            or report.get("status") != "passed" or report.get("identity") != identity(config, device)):
        raise ValueError("AMD preflight is stale, incompatible or did not pass")
    required = ("attention", "document_attention", "model_numerics", "optimized_kernels", "packed_learned_qkv", "optimizer")
    if config.fused_fp8_producers:
        required += ("fp8_producers",)
    if config.linear_backend.startswith("rocm_fp8"):
        required += ("fp8",)
        if config.weight_mode == "ternary":
            required += ("ternary",)
    if any(not report.get(key) for key in required):
        raise ValueError("AMD preflight lacks required native evidence")
    if config.weight_mode == "ternary":
        ternary = report["ternary"]
        if (not isinstance(ternary, dict) or ternary.get("weight_mode") != "ternary"
                or ternary.get("fp8_operand_gate") is not True
                or ternary.get("reduced_model_numerics") is not True
                or ternary.get("optimizer") is not True
                or not isinstance(report.get("fp8"), dict)
                or report["fp8"].get("weight_mode") != "ternary"
                or report["fp8"].get("fp32_wgrad") is not True
                or report["fp8"].get("weight_cache_refresh") is not True):
            raise ValueError("AMD preflight lacks ternary FP8/STE evidence")
    if report.get("report_sha256") != fingerprint({k: v for k, v in report.items() if k != "report_sha256"}):
        raise ValueError("AMD preflight receipt digest mismatch")
    return report


def _errors(actual, expected, *, limit=.03, atol=2e-5):
    if len(actual) != len(expected) or not actual:
        raise AssertionError("numerical gate needs matching nonempty result lists")
    errors = []
    for a, b in zip(actual, expected):
        if a.shape != b.shape:
            raise AssertionError("numerical gate result shapes differ")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError("nonfinite result in numerical gate")
        absolute = float((a.float() - b.float()).abs().max())
        relative = relative_error(a, b)
        if absolute > atol and relative >= limit:
            raise AssertionError(f"native numerical mismatch: relative={relative}, max_absolute={absolute}")
        errors.append({"relative": relative, "max_absolute": absolute})
    return errors


def attention_gate(config, device):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    results = []
    for selected in sorted({config.attention_backend, config.local_attention_backend or config.attention_backend}):
        for length in sorted({1, config.local_attention_window - 1, config.local_attention_window,
                              config.local_attention_window + 1, config.max_seq_len}):
            for window in (None, config.local_attention_window):
                for qlen in sorted({1, min(3, length), length}):
                    shapes = [(1, config.n_heads, qlen, config.head_dim),
                              (1, config.n_kv_heads, length, config.head_dim)]
                    xs = [torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
                          for shape in (shapes[0], shapes[1], shapes[1])]
                    refs = [x.detach().clone().requires_grad_() for x in xs]
                    actual = attention(*xs, backend=selected, local_window=window, query_position_offset=length - qlen)
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = sdpa_reference_attention(*refs, local_window=window, query_position_offset=length - qlen)
                    grad = torch.randn_like(actual)
                    ag = torch.autograd.grad(actual, xs, grad)
                    eg = torch.autograd.grad(expected, refs, grad)
                    errors = _errors([actual, *ag], [expected, *eg])
                    results.append({"backend": selected, "length": length, "query_length": qlen,
                                    "window": window, "errors": errors})
    return results


def document_attention_gate(config, device):
    """Native varlen forward/backward, local boundaries, GQA and pad isolation."""
    from terminus.document_attention import DocumentLayout
    from torch.nn.attention import SDPBackend, sdpa_kernel

    length = config.max_seq_len
    cut = min(length-1, config.local_attention_window+1)
    positions = torch.zeros(2, length, dtype=torch.long)
    segments = torch.full_like(positions, -1)
    for row, lengths in enumerate(((cut, length-cut), (1, max(1, min(cut, length-2))))):
        start = 0
        for segment, size in enumerate(lengths):
            positions[row, start:start+size] = torch.arange(size)
            segments[row, start:start+size] = segment
            start += size
    layout = DocumentLayout.from_cpu(positions, segments).to(device)
    cases = []
    for selected in sorted({config.attention_backend, config.local_attention_backend or config.attention_backend}):
        for window in (None, config.local_attention_window):
            shapes = [(2, config.n_heads, length, config.head_dim),
                      (2, config.n_kv_heads, length, config.head_dim)]
            xs = [torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
                  for shape in (shapes[0], shapes[1], shapes[1])]
            refs = [x.detach().clone().requires_grad_() for x in xs]
            actual = attention(*xs, backend=selected, local_window=window, document_layout=layout)
            with sdpa_kernel(SDPBackend.MATH):
                expected = attention(*refs, backend="sdpa_reference", local_window=window, document_layout=layout)
            gradient = torch.randn_like(actual)
            errors = _errors([actual, *torch.autograd.grad(actual, xs, gradient)],
                             [expected, *torch.autograd.grad(expected, refs, gradient)])
            perturbed = [x.detach().clone() for x in xs]
            for x in perturbed:
                x[0, :, :cut] = torch.randn_like(x[0, :, :cut])*3
            isolated = attention(*perturbed, backend=selected, local_window=window, document_layout=layout)
            if not torch.equal(isolated[0, :, cut:], actual.detach()[0, :, cut:]):
                raise AssertionError("native packed attention leaked between documents")
            cases.append(dict(backend=selected, window=window, errors=errors, isolation=True))
    return {"cases": cases, "reset_positions": True, "gqa": True, "padding_excluded": True}


def fp8_gate(config, device):
    from .fp8_kernels import quantize, quantize_directions, quantize_ternary

    quantizers, cases = [], []
    ternary = config.weight_mode == "ternary"
    weight_quantize = quantize_ternary if ternary else lambda w: quantize(w, group_rows=128)
    weight_reference = reference.quantize_ternary if ternary else lambda w: reference.quantize(w, group_rows=128)
    for rows in (1, 17, 129, 256):
        for amplitude in (0., 1e-7, 1., 1000.):
            x = torch.randn(rows, 256, dtype=torch.bfloat16, device=device) * amplitude
            native = quantize_directions(x, transposed=True)
            expected = reference.directional(x, transposed=True)
            errors = _errors([reference.dequantize(n) for n in native],
                             [reference.dequantize(e) for e in expected], limit=.004, atol=0.)
            # Padding is a memory-safety contract, not merely a numerical tolerance.
            if torch.count_nonzero(native[1][0].float()[:, rows:]):
                raise AssertionError("nonzero padded columnwise FP8 lanes")
            quantizers.append({"rows": rows, "amplitude": amplitude, "errors": errors})
    for special in (float("nan"), float("inf"), -float("inf")):
        x = torch.zeros(128, 128, device=device)
        x[0, 0] = special
        if not torch.isnan(weight_quantize(x)[0].float()[0, 0]):
            raise AssertionError("nonfinite master weight was hidden by quantization")
    if ternary:
        # Integer values avoid accidental near-ties from reduction-order ulps.
        pattern = torch.randint(-2, 3, (128, 256), device=device, dtype=torch.int32).float()
        pattern[:, 128:] = 0
        native_codes, native_scales = quantize_ternary(pattern)
        oracle_codes, oracle_scales = reference.quantize_ternary(pattern)
        if not torch.equal(native_codes.float(), oracle_codes.float()):
            raise AssertionError("native ternary FNUZ codes differ from the FP32 reference")
        _errors([native_scales], [oracle_scales], limit=1e-6, atol=1e-6)
        if torch.count_nonzero(native_codes.float()[:, 128:]) or not torch.all(native_scales[:, 1] == 1e-6):
            raise AssertionError("zero ternary block did not preserve the scale floor")
        stride_codes, stride_scales = quantize_ternary(pattern.T)
        if (not torch.equal(stride_codes.float(), native_codes.float().T.contiguous())
                or not torch.equal(stride_scales, native_scales.T.contiguous())):
            raise AssertionError("ternary quantizer mishandled a strided master matrix")
        ties = torch.full((128, 128), 2., device=device)
        ties[0, 0], ties[0, 1] = 1., 3.  # Exact mean abs = 2, so 1/2 rounds to even zero.
        tie_codes, tie_scales = quantize_ternary(ties)
        if tie_scales[0, 0] != 2 or tie_codes[0, 0].float() != 0:
            raise AssertionError("ternary half-integer must round to nearest even")
    shapes = [(rows, outputs, width) for outputs, width in sorted(set(projection_shapes(config).values()))
              for rows in (17, 256)]
    # Exercise wgrad's long reduction axis without allocating a full model.
    # The selected microbatch has 16 * 4096 token rows per GPU.
    shapes.append((16 * config.max_seq_len, 256, 256))
    for rows, outputs, width in shapes:
        for gradient_scale in (1., 2. ** -24):
            x = torch.randn(rows, width, dtype=torch.bfloat16, device=device, requires_grad=True)
            w = (torch.randn(outputs, width, device=device) * .02).requires_grad_()
            g = torch.randn(rows, outputs, dtype=torch.bfloat16, device=device) * gradient_scale
            actual = backend.linear(x, w, backend=config.linear_backend, splits=config.fp8_wgrad_splits or 1,
                                    weight_mode=config.weight_mode)
            dx, dw = torch.autograd.grad(actual, (x, w), g)
            xr, xt = reference.directional(x.detach(), transposed=True)
            wr = weight_reference(w.detach())
            wt = wr[0].T.contiguous(), wr[1].T.contiguous()
            gr, gt = reference.directional(g, transposed=True, eps=0.)
            expected = [reference.gemm(xr, wr), reference.gemm(gr, wt),
                        reference.gemm(gt, xt, rhs_group_size=1, out_dtype=torch.float32)]
            if dw.dtype != torch.float32:
                raise AssertionError("wgrad must retain FP32 output")
            cases.append({"rows": rows, "outputs": outputs, "width": width,
                          "gradient_scale": gradient_scale,
                          "errors": _errors([actual, dx, dw], expected, limit=.025, atol=0.)})
    # Weight version/cache refresh across an actual in-place master update.
    layer = BitLinear(256, 256, weight_mode=config.weight_mode, linear_backend=config.linear_backend,
                      weight_scale_layout="blockwise", quantize_activations=True,
                      fp8_backward="full", fp8_quantizer="triton_split", fp8_cache_weights=True).to(device)
    x = torch.randn(17, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    before = layer(x)
    with torch.no_grad():
        layer.weight.mul_(1.5)
    cached = layer(x)
    layer.clear_fp8_weight_cache()
    refreshed = layer(x)
    _errors([cached], [refreshed], limit=1e-6)
    if torch.equal(before, cached):
        raise AssertionError("weight cache did not observe the master weight version")
    return {"cases": cases, "quantizers": quantizers, "fp32_wgrad": True,
            "weight_cache_refresh": True, "weight_mode": config.weight_mode,
            "format": "FNUZ E4M3, max 240"}


def _packed_gate(config, device, *, batched_rope):
    from terminus.pointwise import packed_qkv_rope

    q, kv, d = config.n_heads, config.n_kv_heads, config.head_dim
    x = torch.randn(2, 17, (q + 2 * kv) * d, dtype=torch.bfloat16, device=device, requires_grad=True)
    cos, sin = [torch.randn(2 if batched_rope else 1, 1, 17, d // 2, device=device) for _ in range(2)]
    qnorm, knorm = [RMSNorm(d, gain="channel").to(device) for _ in range(2)]
    with torch.no_grad():
        qnorm.weight.copy_(torch.linspace(.5, 1.5, d, device=device))
        knorm.weight.copy_(torch.linspace(1.5, .5, d, device=device))
    actual = packed_qkv_rope(x, cos, sin, q, kv, config.rms_norm_eps,
                             q_gain=qnorm.weight, k_gain=knorm.weight)
    expected = []
    for i, part in enumerate(x.split((q * d, kv * d, kv * d), -1)):
        part = part.view(2, 17, q if i == 0 else kv, d).transpose(1, 2)
        expected.append(apply_rotary((qnorm if i == 0 else knorm)(part), cos, sin) if i < 2 else part)
    grads = [torch.randn_like(y) for y in actual]
    operands = (x, qnorm.weight, knorm.weight)
    return _errors([*actual, *torch.autograd.grad(actual, operands, grads, retain_graph=True)],
                   [*expected, *torch.autograd.grad(expected, operands, grads)], limit=.012)


def packed_gate(config, device):
    return {"shared_rope": _packed_gate(config, device, batched_rope=False),
            "per_batch_reset_rope": _packed_gate(config, device, batched_rope=True)}


def producer_gate(config, device):
    from .producers import rms_norm_fp8, packed_swiglu_norm_fp8
    from .fp8_kernels import quantize
    from terminus.pointwise import rms_norm, packed_swiglu_norm

    results = []
    for kind in ("norm", "swiglu"):
        width = config.dim if kind == "norm" else config.ffn_dim
        for rows in (17, 129):
            for amplitude in (0., 1e-7, 1., 1000.):
                x = (torch.randn(rows, width * (2 if kind == "swiglu" else 1), device=device,
                                 dtype=torch.bfloat16) * amplitude).requires_grad_()
                gain = torch.ones(width if kind == "norm" else (), device=device, requires_grad=True)
                xr, gr = x.detach().clone().requires_grad_(), gain.detach().clone().requires_grad_()
                fused = rms_norm_fp8 if kind == "norm" else packed_swiglu_norm_fp8
                separate = rms_norm if kind == "norm" else packed_swiglu_norm
                out, codes, scales = fused(x, gain, config.rms_norm_eps)
                expected = separate(xr, gr, config.rms_norm_eps)
                expected_codes, expected_scales = quantize(expected.reshape(-1, width), eps=1e-5)
                if codes.dtype != torch.float8_e4m3fnuz or not torch.equal(codes.float(), expected_codes.float()):
                    raise AssertionError("fused producer codes differ from separate AMD quantization")
                if not torch.equal(scales, expected_scales):
                    raise AssertionError("fused producer scales differ from separate AMD quantization")
                gradient = torch.randn_like(out)
                errors = _errors([out, *torch.autograd.grad(out, (x, gain), gradient)],
                                 [expected, *torch.autograd.grad(expected, (xr, gr), gradient)], limit=.012)
                results.append({"kind": kind, "rows": rows, "amplitude": amplitude, "errors": errors})
    return {"cases": results, "fp8_format": "float8_e4m3fnuz", "separate_quantizer_exact": True}


def optimizer_gate(config, device):
    reduced = replace(config, dim=256, n_heads=4, n_kv_heads=2, ffn_dim=640,
                      vocab_size=257, n_layers=6, max_seq_len=16, local_attention_window=8,
                      pad_token_id=0, loss_chunk_size=7)
    a, b = BitNetLlama(reduced).to(device), BitNetLlama(reduced).to(device)
    b.load_state_dict(a.state_dict())
    selected = SELECTED_QAT_OPTIMIZER if config.weight_mode == "ternary" else SELECTED_DENSE_OPTIMIZER
    oa, ob = [build_optimizer(m, OptimizerConfig(**selected)) for m in (a, b)]
    ob.batched = False
    errors = []
    for _ in range(3):
        before = [p.detach().clone() for p in a.parameters()]
        for pa, pb in zip(a.parameters(), b.parameters()):
            pa.grad = torch.randn_like(pa) * .01
            pb.grad = pa.grad.clone()
        before_b = [p.detach().clone() for p in b.parameters()]
        oa.step()
        ob.step()
        errors.extend(_errors([p - old for p, old in zip(a.parameters(), before)],
                              [p - old for p, old in zip(b.parameters(), before_b)], limit=.03))
    return {"updates": 3, "max_update_relative_error": max(x["relative"] for x in errors),
            "scope": "Native batched vs per-logical-matrix spectral NS5; synthetic gradients, reduced geometry"}


def run(config, device):
    require_gfx942(device)
    if config.attention_backend not in {"rocm_aiter", "rocm_ck"}:
        raise ValueError("AMD preflight requires an explicit AMD attention backend")
    torch.manual_seed(1337)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        report = {"schema": 1, "kind": "ci1_amd_preflight", "status": "passed",
                  "counts": count_model(config), "changes_from_h100": changes(config),
                  "attention": attention_gate(config, device),
                  "document_attention": document_attention_gate(config, device),
                  "fp8": fp8_gate(config, device) if config.linear_backend.startswith("rocm_fp8") else None,
                  "packed_learned_qkv": packed_gate(config, device),
                  "optimized_kernels": optimized_kernel_preflight(config, device),
                  "model_numerics": model_numerical_preflight(config, device),
                  "optimizer": optimizer_gate(config, device),
                  "fp8_producers": producer_gate(config, device) if config.fused_fp8_producers else None,
                  "scope": "Kernel and reduced-model gate only. Full-size training, quality, DDP and recovery remain unmeasured."}
        if config.weight_mode == "ternary":
            report["ternary"] = {
                "weight_mode": "ternary",
                "fp8_operand_gate": report["fp8"]["weight_mode"] == "ternary",
                "reduced_model_numerics": bool(report["model_numerics"]),
                "optimizer": bool(report["optimizer"]),
            }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    # Capture after native libraries have loaded/JIT caches are warm.
    report["identity"] = identity(config, device)
    report["report_sha256"] = fingerprint(report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output")
    p.add_argument("--count-only", action="store_true")
    args = p.parse_args()
    config, _, _ = load_config(external_path(args.config))
    if not args.count_only and not args.output:
        p.error("native preflight requires an external --output receipt")
    result = count_model(config) if args.count_only else run(config, torch.device("cuda", 0))
    if args.output:
        write_json_atomic(external_path(args.output, fresh=True), result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
