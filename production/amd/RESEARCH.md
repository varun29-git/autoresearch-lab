# Architecture, software and budget review

Review date: 2026-10-24. Baseline checkout: `5703c77` on `production/ci-1`.
This review concerns the selected production branch, not the old architecture
or Azure/HPO speed campaigns. Source keys refer to [SOURCES.md](SOURCES.md).

## Findings that change the migration plan

1. **The month is an eight-GPU problem.** The pasted writeup models 32 GPUs
   and 6,000 GPU-hours. The current allocation is 8 × 24 × 30 = 5,760 hours.
   Optimize one full xGMI node first. Four-node tuning is a subsequent scaling
   exercise; it does not create more aggregate compute within this budget.
2. **The vocabulary head is BF16.** `production/recipe.py` chooses
   `loss_backend="triton"`. `terminus/loss.py` implements BF16 head GEMMs and
   streaming CE with FP32 classifier-gradient accumulation. The experimental
   FP8 head is a different numerical candidate. The pasted table incorrectly
   makes its migration look like a mandatory FP8-head problem.
3. **AITER assembly is not the local-attention solution.** In the inspected
   `mha.py`, `can_impl_fmha_v3_fwd` and gfx942 backward checks exclude `swa`.
   The selected model has twenty local layers. AITER's CK route or separately
   built FA-CK must handle those layers; global layers may use assembly. [S3]
4. **FNUZ is a numerical change.** E4M3 FNUZ has maximum 240 versus FN's 448,
   different special-value/subnormal behavior, and different quantization
   thresholds under absmax scaling. Merely replacing the import or dtype
   creates the wrong quantizer unless every scale limit is also changed.
   Matching block shapes does not make the two formats numerically identical.
5. **There is now a documented full TE path, but it is not the only possible
   implementation.** ROCm TE v2.17 documents gfx942 1D×2D and 1D×1D FP8 with
   FP32 output through HipKittens. This permits forward, dgrad and wgrad.
   The new local Triton implementation supplies an independent complete path;
   AITER 1D×2D plus local wgrad supplies a third. No one has measured their
   relative cost for CI-1 yet. [S4–S6]
6. **Do not reserve weeks of this month for speculative optimization.** Build
   and stage on CPU beforehand. Use a bounded native acceptance phase and
   stop spending optimization hours when their measured break-even exceeds
   the remaining useful training time.

## Repository inventory and reuse decisions

The inspected repository contains 43 `terminus` Python modules (~9.7k lines),
41 original test modules (~6.6k lines), production runtime,
checkpoint/evaluation/export tools, 20 Python scripts, plus separate research,
speed and HPO controllers. Archived documents describe multiple incompatible
geometries and numerical campaigns. They are evidence/context, not runtime
defaults. The migration follows the executable recipe and architecture review.

| Area | Code and relevant behavior | AMD decision |
| --- | --- | --- |
| Model construction | `terminus/model.py`: packed projections, GQA, mixed local/global layers, Peri-LN, free value coefficients, tied head | Reuse; add named AMD dispatch only |
| Projection quantization | `quantization.py`, `fp8_kernels.py`: 448/FN, Hopper guards, DeepGEMM adapter, per-update weight cache | Isolated FNUZ quantizers/autograd; preserve group direction and rounding boundaries |
| Attention | `kernels.py`: FA3 SM90 gate, SDPA reference, Flex candidate | Add AITER/CK named backends; preserve masks and offsets |
| Packed Q/K | `qkv_kernels.py`, `pointwise.py`: learned shared gains, BF16 rotary roundings, packed gradient layout | Reuse portable Triton; independently gate 18:6 and both gain gradients |
| MLP | `mlp_kernels.py`: packed gate/up, SwiGLU, optional scalar SubLN | Reuse BF16 operations; selected Peri-LN does not enable old SubLN |
| Norm/value residual | `pointwise_kernels.py`: explicit intermediate BF16 roundings, FP32 norm reductions, value-source gradient fan-in | Reuse with HIP libdevice import compatibility; native numerics/occupancy gates |
| Loss | `loss.py`, `loss_kernels.py`: chunked BF16 logits, online LSE, dlogits reuse, FP32 head wgrad | Keep selected backend; tune chunk size with FP32 accumulation verified |
| Optimizer | `optim.py`, `logical_muon.py`: pinned upstream NS5, batched logical slices, foreach Adam | Reuse exact implementation; no alternate Muon, matrix regrouping or state quantization |
| DDP | `train.py`: `nccl`, gradient bucket views, `no_sync`, last-microstep reduction, FP32 token mean | ROCm maps NCCL API to RCCL; retain pure DP |
| Input path | `data.py`: mmap shards, deterministic router, rank-disjoint global indices, pinned prefetch, resumable cursor | Reuse; add immutable staging receipts to remove repeated full-corpus scans |
| Checkpoints | `checkpoint.py`, `production/checkpoints.py`: atomic payload/receipt, contract identity, verified backup/prune | Reuse; bind AMD sources/runtime/tuning; test real recovery |
| Preflight/runtime | `preflight.py`, `train.runtime_contract`, `production/runtime.py`: originally H100-only | Add AMD-specific full identity and numerical receipt; leave H100 wrapper intact |
| Telemetry | `stability.py`, `event_profile.py` | HIP events work; fix monitor to report FNUZ rather than FN quantization error |
| Evaluation | `production/evaluate.py`, `compare_eval.py`, `task_eval.py` | Reuse frozen-input NLL comparisons; generation metrics remain separate |
| QAT/export | `transition.py`, `packed_cpu.py`, `production/export.py` | Preserve existing implementation; AMD dense FP8 QAT is not enabled without a reviewed format/phase plan |
| Historical automation | `research/azure_cycle.py`, `speed_session.py`, `hpo`, old shell bootstrap | Do not use; CUDA installs, Azure lifecycle and obsolete campaign recipes |

No cloud control tool, corpus download, checkpoint conversion, source-corpus
copy, trial-result commit or optimizer retuning was needed for this port.

## Exact computational contract

For each projection `Y = X Wᵀ`, with `X[M,K]`, `W[N,K]`:

| Operation | Left codes/scales | Right codes/scales | Output |
| --- | --- | --- | --- |
| Forward | `X[M,K]`, per row/per 128 K | `W[N,K]`, per 128 N × 128 K | BF16 |
| Input gradient | `dY[M,N]`, per row/per 128 N | Exact transpose of W codes and block scales | BF16 |
| Weight gradient | Quantize `dYᵀ[N,padded M]`, 1×128 | Independently quantize `Xᵀ[K,padded M]`, 1×128 | FP32 |

FP32 scales are absmax/240. Activation amax uses the existing `1e-5` floor;
gradient and weight quantization use zero floor, with zero-amax groups assigned
scale 1. Thus an all-zero activation group uses the activation floor's scale.
Nonfinite inputs propagate NaNs. Weight blocks are already aligned and their
transposed codes/scales are reusable exactly. Padded directional lanes must
be written as zero, not left as uninitialized allocator memory.

The local GEMM accumulates each 128-element dot product in FP32, applies that
block's row/weight scales, and sums the scaled partials in FP32. On gfx942,
software-applied block scales are part of the cost; `tl.dot_scaled`/native
MXFP8 shortcuts for newer architectures are not interchangeable. Explicit
split-K uses FP32 temporary outputs plus a deterministic ordered reduction,
not atomic BF16 partials. The default four splits is a **test starting point**,
not an AMD winner; compare 1/2/4/8 at actual wgrad shapes.

TE receives zero-copy uint8 views of those FNUZ codes and its documented
physical scale layout. Its `Float8BlockQuantizer` defaults to power-of-two
scales, so the adapter does **not** invoke that quantizer or TE Linear. The
code uses the pinned storage wrapper and `general_gemm(..., layout="TN")`
over operands already oriented for each operation. This avoids replacing
parameters, tied embeddings, optimizer groups or cached scale policy. [S6]

The exact production projection shapes are:

| Projection | N | K | FP8 wgrad output |
| --- | ---: | ---: | --- |
| Packed QKV | 3840 | 2304 | 3840 × 2304 |
| Attention output | 2304 | 2304 | 2304 × 2304 |
| Packed gate/up | 11520 | 2304 | 11520 × 2304 |
| Down | 2304 | 5760 | 2304 × 5760 |
| Tied BF16 head | 65536 | 2304 | 65536 × 2304 |

`M = microbatch × 4096`; wgrad reduces over that M. These N/K values are
128-aligned. Padding width 2304 or FFN 5760 to a different model size is
unnecessary and would change the selected architecture. A kernel can pad
physical tiles while preserving the original logical dimensions.

## Hardware advantages and their limits

AMD's MI325X specification lists 256 GB HBM3E, 6 TB/s bandwidth, 256 MB LLC,
304 compute units, 1,307.4 BF16 and 2,614.9 FP8 dense TFLOP/s, and 1,000 W
typical board power. Sparse figures are twice the dense figures and do not
apply to this model. The module is OAM; vendor module data does not guarantee
the cooling, clocks or power policy of a rented server. [S1]

Compared with the repository's H100 NVL specification of 94 GB, 3.9 TB/s,
835.5 BF16 and approximately 1,670.5 FP8 dense TFLOP/s, MI325X offers roughly
2.72× capacity, 1.54× bandwidth and 1.57× peak tensor compute. Multiplying
these ratios together would be invalid: a phase bottleneck uses one resource
or a combination, and vendor peaks are not sustained application rates.

The useful capacity advantage is retained activations, larger microbatches,
FP8 directional caches and transient optimizer/wgrad workspaces without
tensor or pipeline parallelism. It does not pool all eight GPUs' HBM for
one DDP rank. The useful fabric advantage is a full eight-GPU node, avoiding
NVL pair-to-pair PCIe bottlenecks. Neither advantage proves that a specific
GEMM, local-attention backward or optimizer update is faster.

The approximate compute/bandwidth ridge points are similar: MI325X ~218
BF16/~436 FP8 FLOPs per byte, NVL ~214/~428. A component that was dominated by
HBM traffic on NVL may remain bandwidth-bound on AMD; the hardware ratio is
a ceiling on that component's simple scaling, not a whole-model speedup.

The production parameter/gradient memory floor is 11.572 GB/rank for two
FP32 copies. Projection Muon momentum adds ~5.181 GB and tied/small-group
Adam's two moments add ~1.210 GB. Dense FP8 forward/transposed weight caches
add ~2.591 GB. This is ~20.55 GB before activations, DDP buckets, logits,
stacked NS5 intermediates, allocator reservations and library workspaces.
All numbers are decimal GB. PyTorch/driver GiB reports must be converted.

The historical 25-layer mb4/mb8 allocations (55.5/87.4 GB) suggest a rough
23.6 GB intercept plus 7.975 GB per microbatch. That extrapolates to ~151 GB
at mb16 and ~279 GB at mb32. It is **not** a prediction for this 24-layer
AMD path: attention kernels, FP8 caches, DDP and allocator behavior differ.
It is enough to prioritize mb8→16 and require a profile before mb32.

## FLOP bounds, not expected throughput

Applying `speed_research.physics.operation_inventory` to 24 layers/window512:

- Projection forward/backward: **7.7715 GFLOPs/token**, FP8 in the FP8 paths.
- Tied head: **0.9060 GFLOPs/token**, BF16.
- Valid attention QK/PV forward/backward pairs: **0.4920 GFLOPs/token**, BF16.
- Total counted work: **9.1695 GFLOPs/token**; 72,889,344 valid attention
  pairs per sequence across all layers.

FP8 and BF16 share matrix resources. Adding their ideal service times gives
`7.7715e9/2.6149e15 + 1.3980e9/1.3074e15`, an idealized bound of about
**247k tokens/s/GPU**. An all-BF16 version's equivalent bound is ~143k.
Both omit FA score recomputation, tile waste, scaling, softmax, norm/rotary,
HBM, optimizer, launches, data, network and operating overhead. They are
deliberately loose upper bounds, not targets or forecasts. They also show
why the pasted scenario table is not a proof that 1.5T is physically
impossible on MI325X: measured execution efficiency decides the outcome.

## Reconcile the H100 evidence before declaring a win

`docs/speed/TRANSFER_32GPU.md` reports three complete paired 20-update
receipts averaging **40,996 tokens/s**, mb4×accum32, on the older 25-layer
stack. It explicitly treats an 80-update 41,075 tok/s line as lacking the
full local receipt. The supplied writeup quotes 43,873.9 at mb8 and detailed
phase/power data; its accompanying 12-document dossier and raw receipts were
not attached. Those figures are **user-supplied context**, not newly verified
measurements in this review.

Earlier README snapshots also mention synthetic 56.8k and different 30-layer
43.9k results. These are different model/optimizer/workload histories. None
establishes a selected 24-layer H100 baseline or an eight-GPU throughput rate.
Use source-hashed matching receipts if they are available externally. Report
an AMD result against the exact historical workload used, with the geometry
and global-batch caveat visible. Never advertise a fair hardware speedup
by comparing an AMD synthetic component to an H100 real-data full update.

## The current token budget

Eight cards for thirty days give **20,736,000 GPU-seconds**. Required gross
average rates are 1.5T/20,736,000 = **72,337.96** and 2T/20,736,000 =
**96,450.62** tokens/s/GPU. These include every billed second.

| Measured steady 8-rank rate per GPU | Gross tokens, no overhead | Tokens with 320 GPU-hour reserve and 90% duty cycle |
| ---: | ---: | ---: |
| 40k | 0.829T | 0.705T |
| 45k | 0.933T | 0.793T |
| 50k | 1.037T | 0.881T |
| 55k | 1.140T | 0.969T |
| 60k | 1.244T | 1.058T |
| 65k | 1.348T | 1.146T |
| 75k | 1.555T | 1.322T |
| 85k | 1.763T | 1.498T |
| 90k | 1.866T | 1.586T |
| 100k | 2.074T | 1.763T |

Every row is arithmetic, not an estimate of achievable AMD speed. The
reserve/duty cycle is a planning example, not a selected spend authorization.
Use actual-cluster rates so communication is already counted. Apply duty
cycle only to remaining cadence/recovery/idle costs; do not charge setup
again if it is in the reserve. If the node is reserved as eight GPUs, a
one-GPU probe lasting an hour still consumes eight allocation GPU-hours.

For a chosen dense target `Td` and QAT target `Tq`, the cost is
`reserve + (Td/rd + Tq/rq)/(3600 × duty_cycle)`. A harmonic, token-weighted
rate is appropriate; the arithmetic mean of dense and QAT rates is wrong.
The follow-up [QAT_ALLOCATION.md](QAT_ALLOCATION.md) recommends 60/40 by tokens;
the native QAT rate remains unmeasured. If the hardware misses the
rate gate, revise the token horizon/data exposure and WSD decay schedule
before production starts, or obtain more compute. Do not silently shorten
context, layers, global-attention frequency, Muon iterations or quality checks.

## Eight-GPU first, later scale-out

Start DP=8, TP=PP=CP=1. Gradients are 5,785,912,504 bytes/rank; a simple ring
model sends `2×7/8×G = 10,125,346,882` bytes/rank/update. At a measured bus
bandwidth B, raw service time is `2×7/8×G/B`. The actual exposed time depends
on bucket readiness, RCCL scheduling and backward overlap. A standalone
collective is a diagnostic, not an additive fixed DDP penalty.

The existing 25 MiB DDP bucket default means many collectives; the AMD config
builder exposes `--ddp-bucket-cap-mb` to compare 25/64/128 with matched
numerical/recovery gates. Do not assume one giant all-reduce, that
all communication overlaps, or that a single NIC's full line rate belongs
to each rank. FP32 gradient reduction is retained.

Use xGMI topology/NUMA affinity, peer bandwidth and RCCL correctness to catch
partitioning, host-fallback or cabling issues. Verify the actual SKU and
256 GB capacity; gfx942 also includes MI300X and does not by itself identify
MI325X. Inventory SPX/CPX/NPS settings rather than imposing a copied MI350X
tuning recipe. AMD's later RCCL channel defaults for MI350X are not a
universal MI325X channel recommendation. [S8]

For 2/4 nodes, preserve global batch while changing accumulation. Set
`NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NPROC_PER_NODE=8` and
the expected total world size explicitly. Freeze identical code/images/data
on every node and create staging/runtime receipts per node. Check RDMA
devices, GPU/NIC affinity, GID/MTU/RoCE settings and all rails with the provider.
Capture RCCL logs to verify the selected algorithm rather than asserting it
is hierarchical. Then repeat real-update rate, slowest-rank, loss/update
equivalence, checkpoint and worker-loss recovery gates.
