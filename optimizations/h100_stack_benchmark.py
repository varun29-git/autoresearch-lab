"""H100 throughput benchmark: 1.5B stack config, baseline vs the optimizations.

Remote entry point for the Azure benchmark cycle (research.azure_cycle
benchmark). Runs the optimizations/gpu_check.py parity gates first — a broken
fused kernel fails the cycle before any timing — then measures synthetic-step
training throughput of configs/terminus_25l_2304d_64k_h100_stack.json through
terminus.benchmark.training_benchmark in these variants:

- baseline: the committed stack config, unchanged.
- fused: BitNetBlock runs block_boundary (Peri-LN out-norm + residual + FFN
  in-norm) and out_norm_residual (FFN side), attention uses value_mix for the
  V' = l0*V1 + l1*Vn mix, the head loss is optimizations.fused_head_loss, and
  the Muon step batches equal-shaped Newton-Schulz slices.
- fused_graphs: fused + captured CUDA-graph microsteps.
- fused_fp8_head: fused + FP8 E4M3 head tiles (numerics experiment; reported,
  not adopted).

Patches are monkeypatches applied inside this process only; terminus/ is not
modified. development_mode is forced on in the derived configs because the
benchmark feeds synthetic tokens and must not require a campaign preflight
report or data manifest.

    python -m optimizations.h100_stack_benchmark --output report.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STACK_CONFIG = REPO_ROOT / "configs" / "terminus_25l_2304d_64k_h100_stack.json"


def _derived_config(directory, *changes):
    value = json.loads(STACK_CONFIG.read_text())
    for section, key, new in changes:
        value[section][key] = new
    value["training"]["development_mode"] = True
    path = Path(directory) / f"benchmark_{len(list(Path(directory).glob('*.json')))}.json"
    path.write_text(json.dumps(value))
    return path


def _patch_fused(fp8_head=False):
    from terminus import loss as loss_module
    from terminus import model as model_module
    from terminus.kernels import attention
    from terminus.logical_muon import LogicalMuonWithAuxAdam

    from optimizations.fused_head_loss import fused_head_loss
    from optimizations.stack_fusion import block_boundary, out_norm_residual, value_mix
    from optimizations.update_fp8_pipeline import batched_muon_update

    def fused_attention_forward(self, hidden, cos, sin, attention_mask=None, cache=None, *,
                                value_source=None):
        # terminus.model.BitNetAttention.forward with the value-residual mix as
        # one fused kernel; everything else identical.
        batch, length, _ = hidden.shape
        if self.config.fused_qkv:
            from terminus.pointwise import packed_qkv_rope

            query, key, value = packed_qkv_rope(self.qkv_proj(hidden), cos, sin,
                self.config.n_heads, self.config.n_kv_heads, self.config.rms_norm_eps,
                normalize_qk=self.config.qk_norm_mode == "unit_rms")
        else:
            if self.config.fuse_projections:
                kv_width = self.config.n_kv_heads * self.config.head_dim
                query, key, value = self.qkv_proj(hidden).split((self.config.dim, kv_width, kv_width), dim=-1)
            else:
                query, key, value = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
            query = query.view(batch, length, self.config.n_heads, self.config.head_dim).transpose(1, 2)
            key = key.view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
            value = value.view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
            if self.config.fused_pointwise and self.config.qk_norm_mode == "unit_rms":
                from terminus.pointwise import qk_rope

                query, key = (qk_rope(values, cos, sin, self.config.rms_norm_eps) for values in (query, key))
            else:
                from terminus.model import apply_rotary

                query, key = apply_rotary(self.q_norm(query), cos, sin), apply_rotary(self.k_norm(key), cos, sin)
        if value_source is not None:
            value = value_mix(value_source, value, self.value_lambda)
        if self.value_residual:
            self.values_used = value
        offset = 0
        if cache is not None:
            key, value, offset = cache.attention_inputs(key, value)
        probe = getattr(self, "stability_probe", None)
        if probe is not None and probe.active and cache is None:
            probe.record_attention(self.stability_layer, query, key, self.is_global_layer)
        output = attention(query, key, value, backend=self.config.attention_backend,
                           local_window=None if self.is_global_layer else self.config.local_attention_window,
                           query_position_offset=offset, attention_mask=attention_mask,
                           dropout_p=self.config.dropout if self.training else 0.0)
        if cache is not None:
            cache.commit(key, value)
        output = output.transpose(1, 2).reshape(batch, length, self.config.dim)
        return self.o_proj(self.attn_sub_norm(output))

    def fused_block_forward(self, hidden, cos, sin, attention_mask=None, cache=None, *,
                            value_source=None):
        # BitNetBlock.forward with idea-3 fusion: attention out-norm + residual
        # + FFN in-norm in one kernel, FFN out-norm + residual in one kernel.
        branch = self.attn(self.attn_norm(hidden), cos, sin, attention_mask, cache,
                           value_source=value_source)
        hidden, ffn_input = block_boundary(hidden, branch, self.attn_out_norm.weight,
                                           self.ffn_norm.weight, self.attn_norm.eps)
        branch = self.ffn(ffn_input)
        return out_norm_residual(hidden, branch, self.ffn_out_norm.weight, self.ffn_norm.eps)

    def fused_loss(hidden, weight, targets, chunk_size, *, backend="torch"):
        return fused_head_loss(hidden, weight, targets, tile_size=max(int(chunk_size), 8192),
                               fp8_head=fp8_head)

    @torch.no_grad()
    def batched_muon_step(self, closure=None):
        # LogicalMuonWithAuxAdam.step with equal-shaped Newton-Schulz slices
        # batched into one stacked GEMM loop (idea 5). Adam groups unchanged.
        from muon import adam_update

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        pending, grads, momentums, betas = [], [], [], []
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for parameter in group["params"]:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                state = self.state[parameter]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum = state["momentum_buffer"]
                start = 0
                for rows in group.get("logical_rows", (parameter.shape[0],)):
                    stop = start + rows
                    pending.append((parameter, start, stop, group))
                    grads.append(parameter.grad[start:stop])
                    momentums.append(momentum[start:stop])
                    betas.append(group["momentum"])
                    start = stop
        if pending:
            if len(set(betas)) != 1:
                raise ValueError("batched Muon step requires one momentum value across groups")
            updates = batched_muon_update(grads, momentums, beta=betas[0])
            for (parameter, start, stop, group), update in zip(pending, updates):
                weight = parameter[start:stop]
                weight.mul_(1 - group["lr"] * group["weight_decay"])
                weight.add_(update.reshape(weight.shape), alpha=-group["lr"])
        for group in self.param_groups:
            if group["use_muon"]:
                continue
            for parameter in group["params"]:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] = 0
                state["step"] += 1
                update = adam_update(parameter.grad, state["exp_avg"], state["exp_avg_sq"],
                                     state["step"], group["betas"], group["eps"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss

    model_module.BitNetAttention.forward = fused_attention_forward
    model_module.BitNetBlock.forward = fused_block_forward
    loss_module.streaming_cross_entropy_sum = fused_loss
    LogicalMuonWithAuxAdam.step = batched_muon_step


def run_parity_gates():
    from optimizations import gpu_check

    device = torch.device("cuda", 0)
    gates = [("stack_fusion", gpu_check.check_stack_fusion),
             ("head_loss", gpu_check.check_head_loss),
             ("batched_muon", gpu_check.check_batched_muon),
             ("owner_optimizer", lambda d: gpu_check.check_owner_optimizer(d)),
             ("cuda_graph_microstep", lambda d: gpu_check.check_cuda_graph_microstep(d))]
    return {name: gate(device, True) for name, gate in gates}


def run_benchmark(*, batch_size, warmup_steps, measured_steps, skip_gates=False):
    from terminus.benchmark import training_benchmark

    report = {"stack_config": STACK_CONFIG.name, "batch_size": batch_size,
              "warmup_steps": warmup_steps, "measured_steps": measured_steps}
    if not skip_gates:
        report["parity_gates"] = run_parity_gates()
    variants = {}
    with tempfile.TemporaryDirectory(prefix="stack-benchmark-") as directory:
        baseline_path = _derived_config(directory)
        variants["baseline"] = training_benchmark(baseline_path, batch_size=batch_size,
                                                  warmup_steps=warmup_steps,
                                                  measured_steps=measured_steps)
        _patch_fused()
        fused_path = _derived_config(directory)
        variants["fused"] = training_benchmark(fused_path, batch_size=batch_size,
                                               warmup_steps=warmup_steps,
                                               measured_steps=measured_steps)
        graphs_path = _derived_config(directory, ("training", "cuda_graphs", True))
        variants["fused_graphs"] = training_benchmark(graphs_path, batch_size=batch_size,
                                                      warmup_steps=warmup_steps,
                                                      measured_steps=measured_steps)
        _patch_fused(fp8_head=True)
        fp8_path = _derived_config(directory)
        variants["fused_fp8_head"] = training_benchmark(fp8_path, batch_size=batch_size,
                                                        warmup_steps=warmup_steps,
                                                        measured_steps=measured_steps)
    report["variants"] = {
        name: {"tokens_per_second": value["tokens_per_second"],
               "median_step_seconds": value["median_step_seconds"],
               "peak_allocated_bytes": value["peak_allocated_bytes"],
               "losses": value["losses"], "gradient_norms": value["gradient_norms"],
               "physics": value["physics"]}
        for name, value in variants.items()}
    base = report["variants"]["baseline"]["tokens_per_second"]
    for name, value in report["variants"].items():
        value["speedup_vs_baseline"] = value["tokens_per_second"] / base
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--skip-gates", action="store_true",
                        help="skip parity gates (only for local debugging)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable: the stack benchmark must run on the H100")
        return 2
    report = run_benchmark(batch_size=args.batch_size, warmup_steps=args.warmup_steps,
                           measured_steps=args.steps, skip_gates=args.skip_gates)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
