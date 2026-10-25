# MI325X optimization plan and stopping rules

This is an ordered test plan, not a set of achieved speedups. It preserves
the selected model and spectral optimizer; explicit numerical candidates are
identified separately. Commands are human-operated and use external outputs.
References and the hardware analysis are in [SOURCES.md](SOURCES.md) and
[RESEARCH.md](RESEARCH.md). [CONFIG_SELECTION.md](CONFIG_SELECTION.md) now
selects the initial batch, backend, schedule and named comparison profiles.

## Optimize tokens per allocated GPU-hour

The objective is sustained useful tokens and held-out quality within 5,760
GPU-hours. Peak GEMM TFLOP/s, maximum microbatch, MFU and memory utilization
are diagnostics. Increasing any one of them can make full updates slower.

Charge image build/JIT, tuning, failed trials, idle cards, data staging,
validation, checkpointing and recovery to the right budget. Prebuild and
fully validate data on CPU before the paid month where possible. On a node
reserved as eight GPUs, timing a single card does not pause the other seven
cards' allocation clocks.

If an experiment costs `H` allocated GPU-hours and produces a speedup `s`,
the remaining baseline workload must exceed `H / (1 - 1/s)` GPU-hours for
the experiment to pay back. A 5% improvement costing 100 GPU-hours needs
over 2,100 baseline GPU-hours of useful work remaining. A statistically
unclear 1% change is usually a poor use of a short allocation.

## Ordered opportunities

| Priority | Test | Advantage being used | Evidence required to retain it |
| --- | --- | --- | --- |
| 0 | Correct image, SKU, xGMI, data snapshot, FP32 loss/gradients | Avoid losing the month to an invalid environment or storage path | Native preflight and one full-size real update |
| 1 | `fp8_te` vs `fp8` vs `fp8_triton` vs `bf16` | gfx942 MFMA, TE HipKittens, AITER tuned kernels | Quantization-inclusive projection timings, then paired full updates and numerical/quality gates |
| 2 | Microbatch 8→16, then feasible larger sizes at fixed global batch | 256 GB HBM, fewer launches/microsteps and better GEMM occupancy | Peak allocated **and reserved**, all-rank update time, checkpoint memory headroom |
| 3 | Global AITER; local AITER-CK versus independent FA-CK | Tune the actual 20-local/4-global workload | Local/global forward **and backward**, 18:6 GQA, real packed strides |
| 4 | Wgrad splits and shape-specific tiles | Long reduction dimension, many gfx942 CUs, LLC reuse | FP32 wgrad error, scratch memory, quantization-inclusive and end-to-end time |
| 5 | BF16 head chunk sizes and hipBLASLt selection | Large vocabulary GEMMs and more available HBM | Exact sum/ignore-index/tied gradient behavior, actual full update |
| 6 | DDP buckets and provider-appropriate RCCL settings | Full eight-GPU fabric and backward overlap | Slowest rank, bucket readiness/last-reduction trace, FP32 update equivalence |
| 7 | Pointwise launch tuning and bounded producer fusion | HBM bandwidth and reduced intermediate traffic | Preserve BF16 rounding boundaries and all norm/value/gain gradients |
| 8 | NS5 batching/workspace and data/checkpoint cadence | Avoid fixed per-update overhead | Unchanged optimizer state/update math, measured allocation-level benefit |

## GEMM and FP8 path

The dominant projection shapes are QKV 3840×2304, output 2304×2304,
gate/up 11520×2304 and down 2304×5760. At microbatch 16, forward/dgrad M is
65,536; wgrad reduces over 65,536 tokens and produces the parameter matrix.
Those workloads need different scheduling, not a single arbitrary GEMM tile.

The delivered AITER adapter uses current configuration-first dispatch for
forward/dgrad. Its inspected fallback chooses Triton for sufficiently large
M on gfx942. The TE candidate supports all three GEMMs through HipKittens.
The local implementation keeps all three operations testable without relying
on a vendor's forward-only API. It is a correctness-oriented starting kernel
with tunable launch dimensions; no claim is made that its default tile wins.

For the local path, use `--wgrad-splits 1`, `2`, `4` and `8` in distinct configs
at an unchanged global batch. Four FP32 partials of the largest 11520×2304
wgrad occupy ~424.7 MB; eight occupy ~849.3 MB, excluding final output and
live activations. Split-K costs another read/reduction and can lose even on
a GPU with spare capacity. TE owns its own scheduling and uses config value
zero for this field. Never concatenate and requantize microsteps to make
wgrad larger without a separate numerical review.

`CI1_AMD_GEMM_TUNING` can name an external JSON table for the local GEMM.
Keys are `M,N,K,right_scale_group_rows`; entries specify validated launch
fields. For example, a candidate launch for QKV wgrad at microbatch 16 has
key `3840,2304,65536,1`. The format is:

```json
{
  "schema": 1,
  "arch": "gfx942",
  "tiles": {
    "3840,2304,65536,1": {
      "bm": 32, "bn": 64, "num_warps": 4,
      "num_stages": 2, "waves_per_eu": 0
    }
  }
}
```

This example reproduces the default; it is not a tuned result. Candidate
families include BM 16/32/64, BN 32/64/128, four/eight warps, one/two/three
stages, and waves-per-EU 0–4. Select a small set informed by occupancy/LDS/
register pressure rather than running the full Cartesian product. A table
changes runtime identity; use a new process and repeat preflight/receipts.
The kernel fixes the scale reduction group at 128 and uses 64-bit large
matrix offsets. Tile padding does not add model parameters or training tokens.

Rowwise/per-tensor FP8, E5M2 gradients, stochastic rounding, FP8 head GEMMs,
power-of-two scales and quantized optimizer states are **not** enabled as
performance toggles. They change the training computation. The BF16
candidate already provides a useful alternative when fine-grained FP8 is
inefficient; promote it only after explicit numerical/quality comparison.

## Attention: the largest portability trap

Profile both the twenty 512-local layers and the four global layers. AITER
assembly's SWA exclusion means a fast global benchmark says little about
most layers. Also capture actual packed QKV and mixed-value strides: a
contiguous synthetic attention benchmark can select a different implementation.
The provided probe is useful for screening, and `rocprofv3` on a real update
must confirm the path the full model uses.

Build FA-CK with dimensions 64 and 128: 128 is production; 64 supports the
reduced native optimizer/model tests. Do not install a forward-only attention
package or set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` and assume its sliding
window backward is supported. Preserve the exact left window `(511,0)`,
bottom-right cached mask and three query heads per KV head.

Native acceptance must include length 1, 511, 512, 513 and 4096, plus cached
query lengths 1 and 3 against longer keys. Check both learned gain gradients
and the raw layer-zero value source's fan-in. Dropping local masks, expanding
the window to a tile boundary or discarding a value-residual gradient would
invalidate a throughput improvement.

## Use memory to reduce overhead, with a fixed update

At 8 ranks and the selected global batch 2,097,152, these are comparable full updates:

| Microbatch per rank | Accumulation | Tokens/rank/update |
| ---: | ---: | ---: |
| 4 | 16 | 262,144 |
| 8 | 8 | 262,144 |
| 16 | 4 | 262,144 |
| 32 | 2 | 262,144 |

This is a selected starting batch, not a measured winner. The 4,194,304-token
global batch is a separate quality-affecting `large_batch` comparison.
Microbatch 24 does not divide either batch and must not be silently rounded.
Larger batches can also hurt GEMM latency, resource overlap, allocator
fragmentation or attention workspaces. Test steady state after optimizer
state, both weight orientations and attention/BLAS workspaces have been used.
Reserve enough memory for checkpoint copies and peak NS5 stack intermediates;
do not operate against the nominal 256 GB limit.

The default retains activations. Broad activation checkpointing trades the
memory advantage for recomputation; it is not the initial MI325X strategy.
If a measured memory issue requires selective rematerialization, preserve
the special layer-zero value-source handling and revalidate the gradient.
Do not add tensor/pipeline/FSDP sharding to solve a memory problem that this
1.446B model does not yet have on a 256 GB card.

## Vocabulary head and pointwise operations

The selected head executes 0.906 GFLOPs/token in BF16. It computes dlogits
while streaming forward, so profiler “forward” time includes gradient work.
The head weight is cast once per loss invocation and its FP32 wgrad receives
all chunks plus the input-embedding gradient. Benchmark loss chunk sizes
2048, 4096 and 8192 using `--loss-chunk-size`, keeping the same token batch.
At 4096×65536, the BF16 logits buffer alone is 512 MiB; 8192 doubles it.
Changing the chunk size may change floating-point reduction order and must
pass the complete gradient gate.

Use the pinned PyTorch/hipBLASLt BF16 path first. Offline hipBLASLt tuning or
PyTorch TunableOp is appropriate only when the trace shows heuristic misses
on these head/NS5 shapes. Capture the actual tuned file and environment in
the runtime receipt. Disable live tuning once measurement begins; do not
include tuning updates as steady-state performance. The current native CE
preflight explicitly exercises BF16 input with FP32 `addmm` output/accumulation
on the installed ROCm build. A version that cannot execute this contract
has not passed preflight.

Norm, SwiGLU, learned Q/K/RoPE and value-mix Triton kernels are reused with
the existing BF16 store/multiply boundaries and `enable_fp_fusion=False`
where selected by the original code. Retune block/warp occupancy only after
the profile shows a material cost. A future norm→FNUZ producer should emit
codes of the already-BF16-rounded norm output, preserving 1×128 scaling and
the gradient needed by the norm. The H100 OCP producer sidecar is explicitly
rejected by the AMD adapter; the delivered default does not call it.

## Communication, host work and storage

Start with the 25 MiB DDP bucket cap. `--ddp-bucket-cap-mb 64` and `128` make
the comparison explicit in the config/checkpoint contract. Larger buckets
can reduce launch latency but delay the first reduction and increase exposed
tail time. Do not compress the FP32 gradient or change token normalization
to improve a communication benchmark.

Use one process per GPU; bind CPU/data-loader work to the provider's actual
NUMA topology. Do not blindly force NCCL/RCCL rings, channels, GID indices,
network interfaces, CPX or NPS settings from an MI350X example. Run the
provided FP32 collective probe at bucket-sized payloads, then use rccl-tests
for large buffers on the actual topology. Confirm peer traffic traverses
xGMI, not a host staging path. The run-level metric is still complete updates.

Shard reads are already mmap/prefetched. Immutable staging receipts eliminate
the old per-rank full-corpus hash scan at launch. For a 3 TB snapshot, eight
full independent scans would logically examine 24 TB before training.
Receipt verification does not make mutable network storage safe: stage an
immutable snapshot and measure page faults, disk stalls and restart behavior.

Checkpoint and validation intervals are in updates. At the example 4.194M
token batch, 1,000 updates is 4.194B tokens; choose a cadence from measured
save duration, acceptable lost compute and storage durability. A background
uploader must receive an immutable committed checkpoint, never live tensors.
Keep durable backup/retention receipts and test recovery before relying on
them. Use `production.checkpoints`; do not prune the sole recoverable copy.

## Quality and bounded campaign plan

Compare candidates both at matched tokens and matched allocated GPU-hours.
Track language/domain NLL, loss/gradient finite checks, clipping frequency,
Q/K norm gains, value coefficients, projection update norms and token
exposure. A lower English average can hide regressions in another language,
math or code. Keep development and final evaluation sets separate. A model
trained for more tokens is not automatically better if the additional stream
increases repetition, leakage or low-quality mixture exposure.

The following **optional planning envelope** uses at most 40 node-hours
(320 GPU-hours) of the paid month. It is not a launch authorization or a
prediction of required engineering time:

| Window | Maximum node-hours | Decision |
| --- | ---: | --- |
| Runtime/device/native bring-up | 6 | Complete kernel gates; stop for an unresolved compiler/numerics failure |
| Component profiles and backend screening | 4 | Identify the two candidates worth full-size comparison |
| Microbatch/attention/wgrad and 8-rank screening | 8 | Select a measured execution configuration |
| Matched-token numerical/quality controls | 10 | Check early trajectory and multilingual/domain regressions |
| Soak, cadence, failure and recovery | 8 | Verify production operations on the actual node |
| Contingency and final forecast | 4 | Freeze the recipe, image, source and attainable token horizon |

Build time should occur before this envelope where possible. A failed gate
is a reason to repair or revise the plan, not to silently consume the month
searching a broad kernel grid. Twenty/forty-update tests establish execution
behavior, not production quality; continue scheduled quality evaluation during
the real run. QAT needs its own chosen rates, data policy, budget and native
format/quality tests. It is not enabled by this dense handoff.
