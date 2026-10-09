# Honest 32-GPU transfer model for the dense stack-004 speed continuation

This is a scenario calculation, **not a 32-GPU result**. The production objective is 1.5T minimum or 2T target valid tokens in 6,000 aggregate H100-hours across dense training and later QAT. Thirty-two GPUs give 187.5 wall hours at full allocation; more GPUs shorten wall time but do not create extra GPU-hours. The required mean per-GPU production rates before overhead are 69,444 and 92,593 tokens/s.

## Single-GPU evidence and its limits

The new base is E3+E9+E4 at mb4 x accum32, 524,288 valid tokens/update on an H100 NVL. Three complete 20-update candidate receipts give 40,964, 40,967 and 41,058 tokens/s, versus paired incumbent rates of 36,851, 36,639 and 36,680. Candidate mean is **40,996 tokens/s**, with **55.5 GB** peak torch allocation. A checksum-verified archived `soak.out` line from an 80-update run reports **41,075 tokens/s**, **12.76 s/update**, **55.5 GB**. Its full JSON receipt was not copied locally; do not treat the line as a complete source-hashed receipt.

The previous 8-update rate of 41,936 tokens/s was optimistic relative to the 80-update line by 2.1%. The earlier phase probe (forward 5.09 s, backward 8.66 s, optimizer 0.276 s per update) measured a separate instrumented **incumbent** at 14.04 s/update. Those phase times must not be added to the optimized candidate's 12.76 s raw update or used as its measured optimizer cost. Reprofile the optimized candidate briefly if phase timing will drive a new implementation.

At 41,075 tokens/s sustained on every GPU with zero overhead, 6,000 GPU-hours would deliver **887B dense tokens**. This is 613B below the 1.5T minimum. QAT, distributed communication, evaluation, checkpointing and downtime consume some of those hours. The present implementation is not production-feasible under the stated token budget.

## Per-rank compute scenarios

For a global batch `B` valid tokens/update and 32 data-parallel ranks, each rank handles `B/32`. At mb4 x 4K context, one local microstep handles 16,384 tokens. The optimized raw 32-microstep update takes 12.76 s. Let `F` be its *unmeasured* per-update fixed time (optimizer, clipping, cache and host costs). A simple local model is:

`microsteps = B / (32 × 16,384)`

`t_rank ≈ microsteps × (12.76 − F)/32 + F + exposed_communication`

`F` and the transfer of per-microstep cost to fewer local microsteps are **assumptions**. The table excludes communication and shows why short local accumulation may lose throughput even if the network were free:

| Global B | Microsteps/rank | F=0 | F=0.15 s | F=0.30 s |
| ---: | ---: | ---: | ---: | ---: |
| 2,097,152 | 4 | ~41.1k tok/s/GPU | ~38.0k | ~35.3k |
| 4,194,304 | 8 | ~41.1k tok/s/GPU | ~39.7k | ~38.4k |

None approaches 69.4k tokens/s/GPU. The optimizer batching E9 may matter more at low local accumulation, but its production-local benefit is a hypothesis until measured. A larger global batch must also satisfy optimization/quality requirements; it is not a free way to meet the GPU-hour budget.

## Communication and memory

If all 1.5005B trainable gradients are reduced in FP32 once per update, a 32-rank ring all-reduce moves `2 × 31/32 × 4P ≈ 11.63 GB` per rank per update. At effective 50, 100, 200 or 350 GB/s per rank, raw service times are about 233, 116, 58 or 33 ms. The exposed part depends on real gradient-bucket readiness, network topology and overlap with backward; none has been measured. Do not apply an assumed 70–90% overlap as a fact or count the same communication bytes twice.

The optimized mb4 candidate peaked at 55.5 GB allocated on the research NVL. The mb8 variant used **87.4 GB** and exceeds the 74 GB planning envelope; it may not fit an 80 GB production SXM at all. Exact production SKU, power cap, fabric, and global batch remain unknown. Persistent model/optimizer state and temporary NCCL buffers reduce available activation space.

## Required later measurements

On the actual 32 GPUs, measure full dense update rate across all ranks at the chosen global batch, real sharded data throughput, NCCL payload/bucket overlap, slowest-rank time, numerical update equivalence, and checkpoint/restart cost. Then measure the separate ternary-QAT phase under the same accounting. Weight dense and QAT rates by their actual token fractions and include setup, evaluation and downtime before claiming that 6,000 GPU-hours can cover 1.5T or 2T. The five-hour one-GPU loop must keep improving the local rate and refresh this model from complete receipts; it cannot certify the production target alone.
