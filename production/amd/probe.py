"""Human-run exact-production-shape component timing, never whole-model tok/s."""
import argparse
import json
import statistics

import torch

from production.artifacts import external_path
from terminus.fp8_validation import projection_shapes
from terminus.integrity import write_json_atomic, source_hashes
from terminus.kernels import attention
from terminus.loss import streaming_cross_entropy_sum
from terminus.train import load_config, runtime_contract
from . import backend
from .runtime import require_gfx942


def measure(call, device, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(elapsed), "samples_ms": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--component", choices=("linear", "attention", "head", "hbm"), required=True)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    if min(args.microbatch, args.iterations) <= 0 or args.warmup < 0:
        p.error("invalid probe counts")
    output = external_path(args.output, fresh=True)
    model, _, _ = load_config(external_path(args.config))
    device = torch.device("cuda", 0)
    require_gfx942(device)
    torch.manual_seed(1337)
    rows = args.microbatch * model.max_seq_len
    results = {}
    if args.component == "linear":
        for name, (n, k) in projection_shapes(model).items():
            x = torch.randn(rows, k, device=device, dtype=torch.bfloat16, requires_grad=True)
            w = (torch.randn(n, k, device=device) * .02).requires_grad_()
            g = torch.randn(rows, n, device=device, dtype=torch.bfloat16)
            operands = backend.quantize_weight(w, transposed=True) if model.linear_backend.startswith("rocm_fp8") else None

            def call():
                x.grad = w.grad = None
                if operands is None:
                    y = torch.nn.functional.linear(x, w.to(torch.bfloat16))
                else:
                    y = backend.linear(x, w, weight_operands=operands, backend=model.linear_backend,
                                       splits=model.fp8_wgrad_splits or 1)
                y.backward(g)

            results[name] = {"m": rows, "n": n, "k": k, "includes": "forward+dgrad+wgrad+activation/gradient quantization; warm weight cache",
                             **measure(call, device, args.warmup, args.iterations)}
            del x, w, g, operands
    elif args.component == "attention":
        for window in (None, model.local_attention_window):
            selected = model.attention_backend if window is None else model.local_attention_backend or model.attention_backend
            q = torch.randn(args.microbatch, model.n_heads, model.max_seq_len, model.head_dim,
                            device=device, dtype=torch.bfloat16, requires_grad=True)
            k, v = [torch.randn(args.microbatch, model.n_kv_heads, model.max_seq_len, model.head_dim,
                                device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
            g = torch.randn_like(q)

            def call():
                q.grad = k.grad = v.grad = None
                attention(q, k, v, backend=selected, local_window=window).backward(g)

            results[str(window)] = {"backend": selected, "window": window,
                                    **measure(call, device, args.warmup, args.iterations)}
            del q, k, v, g
    elif args.component == "head":
        x = torch.randn(rows, model.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(model.vocab_size, model.dim, device=device) * .02).requires_grad_()
        labels = torch.randint(model.vocab_size, (rows,), device=device)

        def call():
            x.grad = w.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = streaming_cross_entropy_sum(x, w, labels, model.loss_chunk_size, backend="triton")
            (loss / rows).backward()

        results["head"] = measure(call, device, args.warmup, args.iterations)
    else:
        # 4 GiB exceeds the advertised 256 MB LLC; copy bytes count read+write.
        x = torch.empty(2**30, dtype=torch.float32, device=device).fill_(1.)
        y = torch.empty_like(x)
        results["copy"] = measure(lambda: y.copy_(x), device, args.warmup, args.iterations)
        results["copy"]["effective_tb_s"] = 2 * x.nbytes / (results["copy"]["median_ms"] / 1000) / 1e12
    report = {"schema": 1, "kind": "ci1_amd_component_probe", "component": args.component,
              "microbatch": args.microbatch, "source": source_hashes(), "runtime": runtime_contract(device),
              "results": results, "scope": "Synthetic component timing; no model throughput or production quality claim"}
    write_json_atomic(output, report)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
