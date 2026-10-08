# What the budget and H100 physically permit

All throughput here counts valid, loss-bearing training targets exactly once. All hours are **GPU-hours** unless explicitly called wall hours. Fused multiply-add counts as two FLOPs. Hardware rates and bandwidth use decimal units; label GiB separately. These calculations are optimistic ceilings or declared scenarios, not evidence of achieved speed.

## 1. The requirement

For total tokens `T`, aggregate GPU-hours `H`, GPU count `G`:

`required_per_GPU = T / (3600 H)`; `cluster_rate = G × required_per_GPU`; `wall_hours = H/G`.

| Budget assumption | 1.5T minimum | 2T target |
| --- | ---: | ---: |
| 6,000 hours, no overhead: tokens/s/GPU | 69,444 | 92,593 |
| 32 GPUs, no overhead: aggregate tokens/s | 2,222,222 | 2,962,963 |
| 10% unavailable time: useful tokens/s/GPU | 77,160 | 102,881 |
| Above, plus 90% scaling efficiency: one-GPU equivalent tokens/s | 85,734 | 114,312 |
| Above, plus 80% scaling efficiency | 96,451 | 128,601 |

The allocation is **187.5 production wall hours**, or 7 days 19 hours 30 minutes. The user's 800B projection corresponds to 37,037 tokens/s/GPU *if it is already an end-to-end projection over the same 6,000 hours*. It is not an independently measured benchmark. Nominal improvement needed is 1.875× for the minimum and 2.5× for the target. If that 800B number excludes QAT/downtime, the true gap is larger. Establish its provenance at S0.

Experiments consume their own budget. Production evaluation, checkpointing, transition, recoveries, setup on allocated GPUs and idle ranks consume the 6,000 hours. A 32-GPU checkpoint pause of 60 seconds costs 0.533 GPU-hours. A one-hour production debugging session costs 32. Buying more GPUs changes wall time, not the ideal aggregate compute budget; worse scaling can reduce tokens per GPU-hour.

## 2. Count the actual computation

Target fixture: `L=25, d=2304, f=5760, Hq=18, Hkv=6, dh=128, k=768, V=65536, S=4096, W=1024`. Five layers are global, twenty local. Preserve learned channel Q/K gains shared across heads as in the winner, four channel RMSNorms per block, final RMSNorm and 48 free mixing scalars.

Block matrix parameter count is:

`P_block = L(2d² + 2dk + 3df) = 1,349,222,400`.

Tied embedding/head parameters: `Vd = 150,994,944`. Norms/scalars make the total **1,500,456,496**. Tying storage saves parameters; it does **not** remove the vocabulary projection, its hidden gradient or its weight gradient. Embedding lookup is not another dense `V×d` matmul. Do not count the tied head twice in parameter totals.

For `M = microbatch_sequences × S`, these are the relevant forward GEMMs; also benchmark dgrad and wgrad orientations independently:

| Operation | Logical output/input dimensions | Packed forward output width |
| --- | --- | ---: |
| Q, K, V projections | `(2304,2304)`, `(768,2304)`, `(768,2304)` | 3840 |
| Attention output | `(2304,2304)` | 2304 |
| Gate and up | each `(5760,2304)` | 11520 |
| Down | `(2304,5760)` | 2304 |
| Tied output classifier | `(65536,2304)` | 65536 |

Three GEMMs per dense linear forward/backward give `6 × parameters` FLOPs/token. The block projection total is **8.0953 GFLOPs/token in FP8 operands**, while the current vocabulary path contributes **0.9060 GFLOPs/token in BF16**. FP8 operands do not mean all accumulators are FP8. Do not move BF16 work into an FP8 denominator to inflate utilization.

Global causal valid pairs, including the diagonal, are `S(S+1)/2 = 8,390,656`. Local pairs are `W(W+1)/2 + (S−W)W = 3,670,528`. This is the exact triangular prefix plus the full sliding window, not `SW` everywhere. The local window includes self and 1023 preceding tokens. The total is 115,363,840 valid pairs per sequence over all layers.

For attention, QK and PV forward plus both operand gradients cost `12d` FLOPs per valid pair, yielding **0.7787 GFLOPs/token in BF16**. GQA reduces KV projection/storage but query-head attention still uses all 18 query heads. FlashAttention backward normally recomputes scores and softmax; the compute floor deliberately excludes this extra work. A common score recompute adds another `2d` FLOPs/pair, before elementwise work. Measure actual kernels and masked-tile waste rather than treating the valid-pair count as issued operations.

Also inventory norm/softmax/SwiGLU/SFU work, casting/scales, output-norm/value gradients, optimizer, checkpoint rematerialization and padding tiles. `6N` is a useful rough estimate, not a complete performance model. Report **useful model FLOPs** separately from **executed hardware FLOPs**, especially when checkpointing or padding changes.

## 3. Mixed-precision compute ceiling

The [NVIDIA product table](https://www.nvidia.com/en-us/data-center/h100/) marks its tensor-core numbers **with sparsity**. Dense model weights do not get those doubled figures.

| Per GPU | H100 SXM | H100 NVL |
| --- | ---: | ---: |
| Dense FP8 tensor TFLOP/s | 1979 | 1670.5 |
| Dense BF16 tensor TFLOP/s | 989.5 | 835.5 |
| Nominal HBM bandwidth | 3.35 TB/s | 3.9 TB/s |
| Nominal memory | 80 GB | 94 GB |

NVL is not interchangeable with SXM: the research campaign's NVL can hold more memory while having a lower compute ceiling. “H100” alone is insufficient. Confirm power cap, clocks, SKU, ECC, MIG status, driver, fabric and background processes. Nominal bandwidth is not measured sustained bandwidth.

FP8 and BF16 operations share tensor-core resources. An optimistic mixed-path compute service time is:

`t_TC/token >= F_FP8/P_FP8 + F_BF16/P_BF16`.

For this fixture on SXM, this is **5.793 microseconds/token**, or an optimistic **172,617 tokens/s/GPU**. Ideal compute-only hours are **2,414 for 1.5T** and **3,218 for 2T**. The corresponding rate for NVL is available from the calculator. Thus the stated target is not ruled out by the simple compute bound, but there is no proof that the substantial remaining nonidealities can be eliminated.

At 10% unavailable time and 90% scaling efficiency, 2T needs 114,312 single-GPU tokens/s: about **66.2% of this optimistic mixed-compute ceiling**. The minimum requires about 49.7%. These percentages are not conventional MFU; call them “fraction of the mixed-precision compute ceiling.” Real ceilings are lower after attention recompute, memory movement, softmax, optimizer and pipeline dependencies.

Changing head/attention arithmetic changes this model and belongs in a numerical subcampaign. Even infinitely fast head computation could remove only its exposed service time. For example, changing just the head's compute from BF16 to FP8 would save only about 0.458 microseconds/token in this optimistic model, an ~8.6% ideal total gain, before quantization overhead and quality risk. It cannot alone explain a 2.5× improvement. Similarly, recovering the proxy stack's entire recorded 5.8% throughput penalty would be useful but insufficient.

## 4. Memory, IO and launch bounds

For a kernel `i`, a roofline lower bound is `max(F_i/P_i, bytes_i/BW_i)`. For a dependent chain, sum nonoverlapping segment bounds. For the whole update, the maximum of total resource service times and longest dependency-path latency is a lower bound; adding every traced category can double-count overlaps. Use a DAG/timeline, not a pie chart that sums simultaneous kernels and NCCL as if serial. Memory or launch savings cannot exceed their exposed portion.

For an illustrative one-sequence microbatch (`M=4096`) in BF16:

- Residual/branch tensor `M×2304`: 18,874,368 bytes (~18 MiB).
- Raw layer-0 values `M×768`: 6,291,456 bytes (6 MiB), before saved gradients/consumer tensors.
- Packed gate/up `M×11520`: 94,371,840 bytes (90 MiB) per layer.
- Vocabulary logits `M×65536`: 536,870,912 bytes (512 MiB) per chunk; FP32 intermediates double that. The existing streaming loss bounds this allocation but does not make classifier computation free.

All scale with microbatch. V₁ itself is small relative to all MLP saves; repeated mixed-V saves, retained autograd graphs and module attributes can be the real issue. Record a storage-level allocation/lifetime ledger, de-duplicate aliases and check post-backward frees. `allocated`, `reserved` and device-used memory mean different things. A smaller peak is valuable only if it increases sustained speed, avoids recomputation or makes the production envelope safe.

FP32 latent parameters alone cost ~6.002 GB. FP32 gradients cost another ~6.002 GB. Muon momentum for block matrices is ~5.397 GB; two FP32 Adam moments for the tied embedding are ~1.208 GB, with small extra norm/scalar state. This gives roughly **18.6 GB of persistent training state before FP8 caches, cast copies, activations, temporaries, allocator/context and communication**. Verify dtypes in the real checkpoint; do not assume a compact ternary forward makes training weights or optimizer states two-bit.

Maintain a memory/time Pareto surface over microbatch, accumulation, selected saved tensors and checkpoint regions. Recomputing elementwise functions can be cheap; recomputing tensor-core-heavy MLPs can destroy the throughput objective. For every checkpoint plan, calculate `saved bytes / extra exposed seconds` and actual resulting batch/throughput. Memory optimization is a constrained scheduling problem, not a contest for the smallest allocation.

Hopper provides asynchronous WGMMA, TMA, warp specialization, thread-block clusters, up to 227 KB shared memory per block, and 50 MB L2. These are opportunities, not unlimited storage. Register spilling or one active CTA/SM can erase a fusion gain. Cross-layer megakernels cannot fit arbitrary live activation sets in L2/shared memory. Estimate resident tiles, registers, occupancy and waves before implementation; verify with compiler and profiler reports.

## 5. Dense/QAT and distributed accounting

If fraction `q` of **tokens** is QAT, dense and QAT speeds are `r_d,r_q` on one GPU, scaling efficiency is `eta`, and unavailable production-time fraction is `u`:

`r_single_mix = 1 / ((1−q)/r_d + q/r_q)`

`T_projected = 6000 × 3600 × (1−u) × eta × r_single_mix`.

The simplified calculator applies one `eta` to both phases. In the production worksheet use separate phase efficiencies and budgets if they differ:

`H = T(1−q)/(3600 eta_d r_d) + Tq/(3600 eta_q r_q) + H_nontraining`.

Do not subtract checkpoint/evaluation time both inside an end-to-end measured rate and again via `u`. Do not hide recompilation or transition under a dense/QAT arithmetic mean. QAT allocation is set for model quality elsewhere and then frozen here.

Illustrative scenarios with `eta=0.9`, `u=0.1` (all rates invented for planning):

| Dense / QAT tokens/s/GPU | QAT token share | Projected tokens |
| --- | ---: | ---: |
| 100K / 80K | 40% | 1.591T |
| 120K / 90K | 40% | 1.853T |
| 130K / 110K | 40% | 2.120T |

The table describes what to aim for, not evidence that any row is reachable. Run sensitivity grids over `q=0.1,0.4,0.6`, `eta=0.7…0.95`, `u=0.05…0.15` until actual controls/measurements replace them. The prior short QAT probe cannot determine a 2T-run phase allocation.

For pure data parallelism, global tokens/update `B = 32 × microbatch × S × accumulation`. One-GPU timing at the production *local* accumulation covers `B/32` tokens per update; a one-GPU quality test accumulating `B` tokens is a different workload. Label both explicitly; never use the slower full-global single-GPU optimizer amortization as if it measured production local work, or vice versa. The campaign schema can lock either measurement profile via `global_tokens_per_update` (the complete research update); record the intended production mapping in the quality protocol. Use separate campaign locks for different batch profiles.

With ~1.5B FP32 gradients, a ring all-reduce transfers approximately `2(G−1)/G × 4P ≈ 11.63 GB/rank/update` at G=32. BF16 gradients would halve bytes but would be a changed numerical contract unless already used. Dividing this by an **effective unidirectional per-rank** network bandwidth gives only a communication service-time bound. The 900 GB/s NVLink and 128 GB/s PCIe product figures are not 32-rank all-reduce bandwidths. Account for eight-GPU node topology, shared NICs, cross-node bisection and latency. Gradient buckets overlap with backward only when they are ready; the tied head and V₁ backward fan-in can delay readiness.

A one-GPU profiler can identify bucket readiness and model a delay, but cannot validate overlap, NCCL deadlocks or scaling. Test DDP first for this model size; FSDP/ZeRO/TP are hypotheses only when memory or exposure makes their extra communication worthwhile. Fixed global batch means changing world size changes local work; extrapolate under that constraint. Production acceptance later must measure all ranks, use the slowest rank wall time, and count unique global tokens once.

## 6. Choose research by recoverable time

If a category occupies exposed fraction `f`, accelerating it by `s` yields at most `1/(1−f+f/s)` overall. A 2× kernel speedup on a 10% category is only 1.053×. To achieve 2.5× overall, 60% of baseline time must disappear even with infinitely fast optimized regions. If the trace does not contain that much removable cost, a collection of small fusions will not meet the target; investigate a larger scheduling/memory bottleneck or report the limit.

For each idea estimate `[low,central,high]` whole-step savings and research cost. A production saving of 1% is ~60 GPU-hours at a fixed workload, but experiment cost is still separate and finite. Re-evaluate after every accepted implementation because the dominant bottleneck moves. Use other implementations to challenge a poor local kernel; use the architecture-specific physics model to challenge complacency with everybody else's baseline.
