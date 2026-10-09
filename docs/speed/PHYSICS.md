# Single-H100 physical bounds for the dense stack-004 loop

This document supports the five-hour, **one-H100 NVL dense-throughput** campaign. Count each valid loss-bearing target exactly once. Kimi should optimize synchronized complete optimizer updates on the real token stream. The 32-GPU/QAT budget analysis is archived in [PRODUCTION_CONTEXT.md](PRODUCTION_CONTEXT.md) and is not a task or success criterion in this campaign.

## Count the fixed work

The model has 25 layers, width 2304, MLP width 5760, 18 query heads, 6 KV heads, 4K context, 1024-token local window in 20 layers and 5 global layers, tied 65,536-token embedding/head, Peri-LN and value residual. There are 1,500,456,496 trainable parameters. The tied weight still incurs the output classifier projection and both its hidden and weight gradients.

For the selected execution, the block linear forward/backward work is approximately **8.0953 GFLOPs per token with FP8 operands**. The exact vocabulary head adds **0.9060 GFLOPs per token in BF16**. The useful attention QK/PV and operand-gradient work is **0.7787 GFLOPs per token in BF16**, before score/softmax recomputation, masked tile waste and pointwise work. This is a useful-work count, not a count of all executed instructions. Confirm any changed kernel's actual FLOPs and precision rather than moving BF16 work into the FP8 denominator.

Causal valid attention pairs per sequence are 8,390,656 for a global layer and 3,670,528 for a local layer, including the diagonal. That is 115,363,840 pairs over all 25 layers. Attention efficiency must compare **issued** and useful pairs; the legal mask does not make invalid tiles free.

## Hardware ceilings are optimistic, not performance targets

The H100 NVL product figures are about 1,670.5 dense FP8 tensor TFLOP/s, 835.5 dense BF16 tensor TFLOP/s, 3.9 TB/s nominal HBM bandwidth and 94 GB nominal memory. These are dense rates: NVIDIA's larger published sparse rates do not apply to this model. Confirm the VM's actual SKU, usable memory, power cap, clocks, ECC/MIG state, background use and measured bandwidth at campaign start.

Because the FP8 and BF16 operations share tensor-core resources, an optimistic arithmetic service floor is

`t_TC/token >= 8.0953/1670.5 + (0.9060+0.7787)/835.5 = 0.006862 ms/token`.

This corresponds to about **145,700 tokens/s** or **3.60 s of tensor-core service per 524,288-token update** at nominal peak. It excludes attention recomputation, FP8 conversion, scales, elementwise work, HBM traffic, synchronization, optimizer, launch gaps and nonideal achieved tensor throughput. It is a bound to challenge a proposed bottleneck, **not** an attainable rate or a stop threshold. The archived selected-stack rate of 41,075 tokens/s is roughly 28% of this optimistic ceiling, which does not imply the rest is all recoverable.

For each dominant kernel, estimate `max(executed_FLOPs/achievable_compute, moved_bytes/achievable_bandwidth)`, then check occupancy, tile waves, register/shared-memory limits and dependencies. A whole-update lower bound must account for overlapping service and the longest dependency path; do not add categories that execute concurrently. Measure clocks and event gaps in the same complete-update run used for the score. A profile that changes the raw update time needs its own overhead estimate.

## Memory and exposed-time decisions

Approximate persistent FP32 weights, gradients and Muon/Adam state total **18.6 GB** before FP8 caches, cast copies, activations, temporaries and allocator/context use. A packed gate/up activation for one 4K sequence is 90 MiB per layer; an unchunked BF16 4K-by-65,536 logits tensor is 512 MiB. Storage aliases and lifetimes matter more than summing independent tensor sizes. Record torch allocated/reserved **and device-wide** use for each candidate.

The research GPU has 94 GB nominal memory. The old 74 GB production-planning envelope is not a gate here. Try faster microbatch/layout schedules if they fit **measured usable memory with headroom for repeatable full updates**. Keep the fixed valid-token count and compare complete updates; a smaller peak matters only if it enables speed or stability. Recompute heavy GEMMs only when the saved memory buys a larger whole-update gain.

Rank ideas by recoverable, **exposed** milliseconds. If a component occupies fraction `f` of the complete update and is accelerated by `s`, its maximum isolated speedup is `1/(1-f+f/s)`. An attractive microbenchmark may produce no end-to-end gain if its time was already overlapped or replaced by new conversion/launch work. After each material win, reprofile briefly and move to the newly dominant exposed cost. Other implementations are candidate sources; the hardware and dependency bounds determine how far to push.

A justified early saturation claim needs sustained complete-update measurements, a credible achieved-hardware bound for the remaining dominant components, and quantitative falsification of untried high-upside paths. The five-hour cap otherwise ends the campaign.
