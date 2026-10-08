"""End-of-line GPU validation for the optimizations package.

Run on the H100 (Hopper SM90) before wiring anything into terminus/:

    python -m optimizations.gpu_check            # parity gates + timings
    python -m optimizations.gpu_check --quick    # parity only, no timings

Every parity gate compares the fused/optimized path against the eager model
math it replaces and exits nonzero on failure. Timings are reported, not
asserted; speed claims become real only when this prints them.
"""
from __future__ import annotations

import argparse
import sys

import torch

DIM = 2304          # stack width (25 x 2304 candidate; layer count unchanged)
VOCAB = 65_536
EPS = 1e-5


def time_cuda(fn, *, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / repeat


def check_stack_fusion(device, quick):
    from terminus.model import RMSNorm

    from optimizations.stack_fusion import block_boundary, out_norm_residual, value_mix

    torch.manual_seed(61)
    residual = torch.randn(4, 512, DIM, device=device)
    branch = torch.randn(4, 512, DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    out_gain = (torch.randn(DIM, device=device) * 0.2 + 1.0).requires_grad_()
    in_gain = (torch.randn(DIM, device=device) * 0.2 + 1.0).requires_grad_()

    norm_a, norm_i = RMSNorm(DIM, EPS).to(device), RMSNorm(DIM, EPS).to(device)
    with torch.no_grad():
        norm_a.weight.copy_(out_gain.detach())
        norm_i.weight.copy_(in_gain.detach())

    eager_xn = residual + norm_a(branch)
    eager_h = norm_i(eager_xn)
    fused_xn, fused_h = block_boundary(residual, branch, out_gain, in_gain, EPS)
    torch.testing.assert_close(fused_xn, eager_xn, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(fused_h, eager_h, atol=2e-2, rtol=1e-2)

    eager_xn.backward(torch.ones_like(eager_xn))
    eager_branch_grad = branch.grad.clone()
    branch.grad = None
    fused_single = out_norm_residual(residual, branch, out_gain, EPS)
    fused_single.backward(torch.ones_like(fused_single))
    torch.testing.assert_close(branch.grad, eager_branch_grad, atol=2e-2, rtol=1e-2)

    v1 = torch.randn(2, 6, 1024, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    vn = torch.randn_like(v1).requires_grad_()
    lambdas = torch.tensor([0.5, 0.5], device=device, requires_grad=True)
    eager_v = lambdas[0] * v1 + lambdas[1] * vn
    fused_v = value_mix(v1, vn, lambdas)
    torch.testing.assert_close(fused_v, eager_v, atol=1e-2, rtol=1e-2)

    report = {"block_boundary_parity": True, "value_mix_parity": True}
    if not quick:
        def eager_block():
            xn = residual + norm_a(branch)
            return norm_i(xn)

        report["eager_boundary_ms"] = time_cuda(eager_block)
        report["fused_boundary_ms"] = time_cuda(
            lambda: block_boundary(residual, branch, out_gain, in_gain, EPS))
        report["eager_value_mix_ms"] = time_cuda(lambda: lambdas[0] * v1 + lambdas[1] * vn)
        report["fused_value_mix_ms"] = time_cuda(lambda: value_mix(v1, vn, lambdas))
    return report


def check_head_loss(device, quick):
    from optimizations.fused_head_loss import fused_head_loss
    from terminus.loss import streaming_cross_entropy_sum

    torch.manual_seed(67)
    rows, batch, length = 4096, 4, 1024
    hidden = torch.randn(batch, length, DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(VOCAB, DIM, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, VOCAB, (batch, length), device=device)
    targets[:, :7] = -100

    fused = fused_head_loss(hidden, weight, targets, tile_size=8192)
    fused.backward()
    fused_hidden_grad, fused_weight_grad = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    reference = streaming_cross_entropy_sum(hidden, weight, targets, 1024, backend="triton")
    reference.backward()
    torch.testing.assert_close(fused, reference, atol=1e-1, rtol=1e-3)
    torch.testing.assert_close(fused_hidden_grad, hidden.grad, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(fused_weight_grad, weight.grad, atol=5e-2, rtol=2e-2)

    fp8 = fused_head_loss(hidden.detach(), weight.detach(), targets, tile_size=8192, fp8_head=True)
    relative = abs(fp8 - fused.detach()) / fused.detach()
    report = {"head_loss_parity": True, "fp8_head_relative_loss_delta": relative.item()}
    if not quick:
        report["streaming_head_ms"] = time_cuda(
            lambda: streaming_cross_entropy_sum(hidden.detach(), weight.detach(), targets, 1024,
                                                backend="triton"))
        report["fused_head_ms"] = time_cuda(
            lambda: fused_head_loss(hidden.detach(), weight.detach(), targets, tile_size=8192))
        report["fused_head_fp8_ms"] = time_cuda(
            lambda: fused_head_loss(hidden.detach(), weight.detach(), targets, tile_size=8192,
                                    fp8_head=True))
        report["head_rows"] = rows
    return report


def check_batched_muon(device, quick):
    from muon import muon_update

    from optimizations.update_fp8_pipeline import batched_muon_update

    torch.manual_seed(71)
    # The stack's projection shapes: qkv (3456x2304), o (2304x2304), gate_up (11520x2304), down.
    shapes = [(3456, 2304)] * 25 + [(2304, 2304)] * 25 + [(11520, 2304)] * 25 + [(2304, 5760)] * 25
    grads = [torch.randn(shape, device=device) for shape in shapes]
    momentums = [torch.zeros(shape, device=device) for shape in shapes]
    updates = batched_muon_update(grads, momentums, beta=0.95)
    reference_momentum = torch.zeros(shapes[0], device=device)
    reference = muon_update(grads[0].clone(), reference_momentum, beta=0.95)
    torch.testing.assert_close(updates[0], reference, atol=1e-4, rtol=1e-3)

    report = {"batched_muon_parity": True}
    if not quick:
        def sequential():
            for grad, shape in zip(grads, shapes):
                muon_update(grad.clone(), torch.zeros(shape, device=device), beta=0.95)

        report["sequential_ns_ms"] = time_cuda(sequential, warmup=2, repeat=5)
        report["batched_ns_ms"] = time_cuda(
            lambda: batched_muon_update(grads, [torch.zeros(s, device=device) for s in shapes],
                                        beta=0.95), warmup=2, repeat=5)
    return report


def check_owner_optimizer(device):
    from optimizations.owner_optimizer import (OwnerMuonOptimizer, SingleProcessCollectives,
                                               replicated_state_bytes, ring_volume_bytes)

    torch.manual_seed(73)
    shapes = {"qkv": (512, 256), "down": (256, 768)}
    initial = {name: torch.randn(shape) for name, shape in shapes.items()}
    grads = {name: torch.randn(shape, device=device) for name, shape in shapes.items()}
    optimizer = OwnerMuonOptimizer(shapes, comm=SingleProcessCollectives(), weight_mode="dense",
                                   device=device)
    optimizer.initialize_masters({name: w.to(device) for name, w in initial.items()})
    optimizer.step(grads, max_norm=1.0)
    replica = optimizer.slots["qkv"]["replica"]["forward"]
    assert replica[0].dtype == torch.float8_e4m3fn and replica[0].is_cuda

    params = 1_400_000_000
    baseline_volume = ring_volume_bytes(params, 8, scheme="allreduce_fp32")
    owner_volume = ring_volume_bytes(params, 8, scheme="owner")
    saved = replicated_state_bytes(params, 8, scheme="allreduce_fp32") - replicated_state_bytes(
        params, 8, scheme="owner")
    return {"owner_step_single_rank": True,
            "owner_volume_ratio_vs_allreduce": owner_volume / baseline_volume,
            "owner_state_saved_gib_at_8_ranks": saved / (1 << 30)}


def check_cuda_graph_microstep(device):
    from terminus.cuda_graph import CapturedMicrostep
    from terminus.model import BitNetLlama, BitNetLlamaConfig

    torch.manual_seed(79)
    config = BitNetLlamaConfig(vocab_size=1024, dim=256, n_layers=4, n_heads=4, n_kv_heads=2,
                               ffn_dim=512, max_seq_len=128, local_attention_window=64,
                               global_attention_every=2, branch_norm="peri", value_residual="learned",
                               residual_dtype="bf16")
    model = BitNetLlama(config).to(device).train()
    inputs = torch.randint(0, 1024, (2, 128), device=device)
    targets = torch.randint(0, 1024, (2, 128), device=device)
    captured = CapturedMicrostep(model, inputs, targets)
    first = captured.run(inputs, targets, 1.0)
    replay = captured.run(inputs, targets, 1.0)
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    return {"cuda_graph_stack_microstep": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="parity gates only, no timings")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable: gpu_check must run on the H100")
        return 2
    device = torch.device("cuda", 0)
    capability = torch.cuda.get_device_capability(device)
    print(f"device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")

    checks = [
        ("stack fusion (Peri-LN boundary + value mix)", check_stack_fusion),
        ("fused head loss (idea 2)", check_head_loss),
        ("batched Newton-Schulz (idea 5)", check_batched_muon),
        ("owner optimizer (idea 1, single rank)", lambda d: check_owner_optimizer(d)),
        ("CUDA graph microstep on the stack", lambda d: check_cuda_graph_microstep(d)),
    ]
    failures = 0
    for label, check in checks:
        try:
            report = check(device, args.quick)
        except Exception as exc:
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        details = ", ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                            for key, value in report.items())
        print(f"[ok]   {label}: {details}")
    if failures:
        print(f"{failures} check(s) failed")
        return 1
    print("all GPU checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
