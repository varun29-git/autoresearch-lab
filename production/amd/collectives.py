"""Human-run RCCL payload probe with correctness and rank timing, no cloud API."""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from production.artifacts import external_path
from terminus.integrity import write_json_atomic
from .runtime import require_gfx942


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--sizes-mib", nargs="+", type=int, default=[1, 25, 64, 256, 1024])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or any(s <= 0 for s in args.sizes_mib):
        p.error("invalid probe dimensions")
    output = external_path(args.output, fresh=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    require_gfx942(device)
    dist.init_process_group("nccl")  # PyTorch maps this name to RCCL on HIP.
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        cases = []
        for mib in args.sizes_mib:
            count = mib * 1024 * 1024 // 4
            x = torch.empty(count, device=device, dtype=torch.float32)
            elapsed = []
            for iteration in range(args.warmup + args.iterations):
                x.fill_(rank + 1)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                dist.all_reduce(x)
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - start
                if not torch.all(x == world * (world + 1) // 2):
                    raise AssertionError("RCCL all-reduce result is incorrect")
                if iteration >= args.warmup:
                    elapsed.append(seconds)
            gathered = [None] * world
            dist.all_gather_object(gathered, elapsed)
            slow = [max(t[i] for t in gathered) for i in range(args.iterations)]
            mean = sum(slow) / len(slow)
            cases.append({"bytes": count * 4, "mean_slowest_rank_seconds": mean,
                          "algorithm_gb_s": count * 4 / mean / 1e9,
                          "ring_normalized_bus_gb_s": count * 4 / mean / 1e9 * 2 * (world - 1) / world,
                          "rank_seconds": gathered})
        if rank == 0:
            result = {"schema": 1, "kind": "ci1_amd_collectives", "world_size": world,
                      "cases": cases, "scope": "Standalone FP32 all-reduce; does not establish DDP overlap or hierarchical algorithm selection"}
            write_json_atomic(output, result)
            print(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
