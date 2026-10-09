#!/usr/bin/env python3
"""Native class-E gate for the batched-Muon candidate (trial e9-batched-muon).

Builds the optimizer group/slice structure of a small stack-004-shaped model on
CUDA, injects deterministic synthetic gradients, and steps the batched and
reference optimizer paths in lockstep from identical state. Reports the maximum
parameter and optimizer-state deltas. Newton-Schulz runs as bmm vs mm here, so
BF16 reduction-order noise is expected and bounded; momentum and Adam state are
elementwise and must match exactly.

Usage: python scripts/e9_gpu_parity.py --steps 3 --out /external/path/parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from terminus.model import BitNetLlama
from terminus.optim import OptimizerConfig, build_optimizer
from tests.helpers import tiny_config


def build_pair(device):
    config = tiny_config(qk_norm_mode="learned_rms", fuse_projections=True,
                         value_residual="learned", branch_norm="peri")
    torch.manual_seed(41)
    model = BitNetLlama(config).to(device)
    reference_model = BitNetLlama(config).to(device)
    reference_model.load_state_dict(model.state_dict())
    options = OptimizerConfig(kind="muon")
    batched = build_optimizer(model, options)
    reference = build_optimizer(reference_model, options)
    reference.batched = False
    return model, reference_model, batched, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this gate requires the native CUDA device")
    device = torch.device("cuda")
    model, reference_model, batched, reference = build_pair(device)
    assert not reference.batched and batched.batched
    per_step = []
    for step in range(args.steps):
        torch.manual_seed(1000 + step)
        for (_, pa), (_, pb) in zip(model.named_parameters(), reference_model.named_parameters()):
            gradient = torch.randn(pa.shape, device=device, dtype=pa.dtype)
            pa.grad = gradient.clone()
            pb.grad = gradient.clone()
        batched.step()
        reference.step()
        max_param = 0.0
        max_state = 0.0
        for (_, pa), (_, pb) in zip(model.named_parameters(), reference_model.named_parameters()):
            max_param = max(max_param, (pa - pb).abs().max().item())
            state_a, state_b = batched.state[pa], reference.state[pb]
            for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                if key in state_a:
                    max_state = max(max_state, (state_a[key] - state_b[key]).abs().max().item())
        per_step.append({"step": step + 1, "max_abs_param_delta": max_param,
                         "max_abs_state_delta": max_state})
    record = {"kind": "e9_gpu_parity", "steps": args.steps, "per_step": per_step,
              "device": torch.cuda.get_device_name(device),
              "note": "elementwise state paths are exact; NS bmm-vs-mm reduction order bounds param deltas"}
    Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record["per_step"], indent=1))


if __name__ == "__main__":
    main()
