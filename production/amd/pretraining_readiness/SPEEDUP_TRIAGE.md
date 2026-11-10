# MI325X speedup triage

Reviewed 2026-10-26 against the selected 24-layer, width-2304, eight-GPU
training path and the supplied MI325X optimization note. **No native MI325X
speedup has been measured.** This is a short queue for the bounded hardware
qualification period, not a claim that every idea will improve training.
Use complete-update time and final quality per allocated GPU-hour as the
decision metrics. See [OPTIMIZATION.md](../OPTIMIZATION.md) for numerical gates
and [BATCH_STABILITY.md](../BATCH_STABILITY.md) for global-batch learning gates.

## What the pasted note gets wrong for this workload

| Claim | Code-based check | Decision |
| --- | --- | --- |
| The model uses under 10% of 256 GB, leaving 220+ GB free | Parameters, gradients, Muon/Adam state and FP8 caches have a ~20.55 GB floor **before** activations, DDP buckets and workspaces. Older H100 allocation extrapolates to ~151 GB at microbatch 16, with large uncertainty on AMD. | Measure warm allocated, reserved and device-level peak; do not size a data cache from parameter memory alone. |
| A 2304-ish projection cannot occupy 304 CUs because its N dimension has only 18 tiles | At microbatch 16, `M=65,536`. Even the smallest 2304-wide forward projection has `512 × 18 = 9,216` nominal 128×128 output tiles. The attention-output wgrad has only `18 × 18 = 324` output tiles, where split-K may help. | Profile wgrad specifically; forward GEMM is not starved merely by N=2304. Tile count is a screen, not an occupancy measurement. |
| Store a 50–100B-token corpus on each GPU or a full 800B-token corpus across eight cards | 800B `uint16` tokens need 1.6 TB before runtime copies. Allocating 200 GB per card leaves only 56 GB for model training and its activations. The existing loader sends approximately 4 MiB of `int64` input/target tensors per GPU per 2M-token update and already pins/prefetches them on a separate stream. | Keep immutable mmap shards on local storage; consider GPU-resident data only if measured data wait/page faults materially expose the GPU. |
| Turn on HIP graphs or `torch.compile(..., mode="reduce-overhead")` for an automatic large win | The selected AMD validator rejects compile/graphs; the trainer's captured microsteps require **one GPU**. The older H100 optional graph path did not improve its larger-microbatch winner. Static buffers, RCCL capture, FP8 weight refresh and partial tails need new eight-rank validation. | Prototype only after a trace shows host launch gaps on the chosen batch. No universal 20–50% memory or speed estimate is established. |
| Force a 512 MB–1 GB hipBLASLt workspace and split-K everywhere | The FP8 lead uses TransformerEngine HipKittens for GEMMs, not hipBLASLt. The local FP8 wgrad already supports explicit splits and shape-specific Triton tiles; the BF16 head is the plausible hipBLASLt tuning target. Extra splits require FP32 scratch reads/reductions and can lose. The proposed `HIPBLASLT_WORKSPACE_SIZE` switch is not established by the inspected [official hipBLASLt environment-variable page](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html). | Tune the backend that actually executes each traced GEMM; compare whole updates and FP32 wgrad error. |
| Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95,max_split_size_mb:512` to guarantee zero allocator stalls | The [pinned PyTorch 2.12 allocator documentation](https://docs.pytorch.org/docs/2.12/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf) documents `PYTORCH_ALLOC_CONF` and a compatibility alias `PYTORCH_CUDA_ALLOC_CONF`, not the proposed HIP spelling. It calls `expandable_segments` experimental for changing allocation sizes, and `max_split_size_mb` a last resort for OOM with inactive split blocks. A 0.95 garbage threshold does not prevent all alloc/free calls or guarantee zero stalls. | Capture allocator snapshots and native traces first; verify each option on the exact HIP build and test one at a time only if fragmentation or synchronization is measured. |
| Pin all model weights in the 256 MB last-level cache, or run parallel microsteps on separate streams | The cache is hardware managed; full FP8 projection caches alone are ~2.59 GB/rank. Parallel microsteps compete for the same CUs/HBM and can raise activation peak. | Keep per-update FP8 weight caching already present; treat stream concurrency as a research candidate only after profiler evidence. |

The input-transfer figure assumes `16 × 4096` tokens per rank, two `int64`
tensors and four microsteps. Real page faults, CPU routing or storage waits
could still matter; measure them instead of assuming either zero cost or a
dataset-HBM windfall. External sources are in [SOURCES.md](../SOURCES.md);
the allocator and hipBLASLt checks above are API audits, not hardware tests.

## Remaining opportunities, ordered for this run

| Rank | Opportunity and present state | Promote only if the eight-GPU evidence shows |
| --- | --- | --- |
| 1 | **FP8 backend selection.** TE HipKittens, AITER+local wgrad and Triton paths are implemented; none has native MI325X full-update timing or numerical acceptance. | A backend passes forward/dgrad/FP32-wgrad and complete-model tests, then wins paired *quantization-inclusive* real-data updates at the same batch. This is the first likely high-leverage choice. |
| 2 | **Microbatch at fixed global batch.** Profiles for 4/12 at the 393k HPO batch, 8/16/32 at 2M, and 16/24 at 3.146M exist; fit and speed are unmeasured. | At least 15% actual usable HBM remains through optimizer, checkpoint and validation, and complete-update throughput improves without quality regression. A new global batch gets its own learning bridge. |
| 3 | **Twenty local-attention layers.** AITER-CK and independent FA-CK are wired as candidates; the selected local window is 512, while only four layers are global. | Real packed-stride forward **and backward** traces show attention is exposed, and the replacement preserves the left-window mask, GQA and value-residual gradients. |
| 4 | **Wgrad reduction and tile choices.** Local split-K values 1/2/4/8 and a small Triton tuning table exist; TE owns its own split policy. `autotune --run` sweeps those four counts per weight-gradient shape. A schema-2 table applies the measured count; until it is loaded, the config split is unchanged. | The chosen per-shape kernel lowers full-update time after scale conversion and FP32 scratch reduction, with unchanged error bounds and headroom. Do not force one split count across QKV, O, MLP and down. |
| 5 | **BF16 tied head.** Loss chunks 2048/4096/8192 and offline hipBLASLt tuning are available to test; an FP8 head would be a new numerical candidate. | The full head forward/backward is a measured bottleneck and a different chunk/algorithm improves whole updates with exact tied-weight, ignored-token and FP32-gradient checks. Gate any FP8 head on a separate quality bridge. |
| 6 | **RCCL and DDP overlap.** Bucket caps 25/64/128 MiB are exposed; topology and slowest-rank overlap remain unmeasured. | A trace finds exposed all-reduce or poor xGMI routing and the alternative reduces update tail without delaying backward overlap. |
| 7 | **Pointwise/quantizer fusion.** Native Triton pointwise kernels are reused, but selected norm→FNUZ producer fusion is not implemented. | Pointwise launches or HBM traffic are a significant exposed share. Preserve the existing BF16 rounding, 1×128 scales and norm/value/gain gradients in native tests. |
| 8 | **Optimizer and cadence.** NS5 batching is already present. Provider-aware NS5 workspace and live verified background checkpoint upload remain incomplete. | Optimizer work or synchronous checkpoint/upload is a measured fraction of allocated time; changes preserve spectral Muon and recoverability. |
| 9 | **Eight-rank HIP graph capture.** Not implemented; the existing capture is single-GPU only. | Host dispatch gaps remain material after the cheaper changes, and an explicit RCCL/static-buffer implementation passes changing-input, FP8-cache, DDP and exact-resume tests. |
| 10 | **Selective activation checkpointing.** The selected run retains activations; layer-zero value-source handling makes broad recomputation delicate. | A measured memory limit blocks a faster microbatch and selective recomputation wins complete-update time despite extra forward work. |

Do not convert a microbenchmark win into a speedup claim. The benchmark in
[train.py](../train.py) times real data plus complete optimizer updates but
reports checkpoint/validation cadence separately. Include that cadence,
recovery and idle allocation in the final tokens-per-GPU-hour comparison.
For perspective, halving a component that occupies 20% of update time yields
only `1 / (0.8 + 0.2 / 2) = 1.11×` overall speed. A trace should determine
which component deserves engineering time before implementing the next idea.
