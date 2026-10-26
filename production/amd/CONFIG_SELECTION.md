# Selected eight-MI325X configuration

Decision: 2026-10-25. These are concrete starting settings for the selected
1,446,478,126-parameter model. They are ready to generate and test; they are
not native measurements or permission to launch paid hardware. The executable
choices are in [profiles.py](profiles.py), with [prepare.py](prepare.py)
writing actual manifests/configs outside Git. For a final ternary model,
[QAT_ALLOCATION.md](QAT_ALLOCATION.md) now recommends 60% dense / 40% QAT by
tokens and a combined 480 GPU-hour qualification reserve. The forecasts in
this document remain standalone dense-only scenarios. The lower-level builder remains
available for explicitly reviewed experiments.
See [BATCH_STABILITY.md](BATCH_STABILITY.md) for the batch/LR-transfer and
dense-to-QAT decision sequence, including the supplied QAT candidate.

**Microbatch 16 is a starting point, not an optimum.** Neither complete-update
throughput nor peak memory has been measured on MI325X. In particular, it is
incorrect to claim that 65,536 tokens per GPU saturate the compute units or
fit in cache. One BF16 residual tensor alone is
`16 × 4096 × 2304 × 2 = 301,989,888` bytes (288 MiB), larger than the
advertised 256 MB LLC. Kernels tile and stream data through HBM; LLC capacity
and the 256 GB HBM capacity are different constraints. Filling HBM is not
itself an optimization objective.

## Main choice: `balanced`

| Setting | Selected value | Reason |
| --- | --- | --- |
| Node / parallelism | One node, 8 MI325X, DDP; TP=PP=1, no FSDP/ZeRO | Replicate this small model and use xGMI for gradients; avoid parameter gathers and pipeline bubbles |
| Context / microbatch | 4096 tokens / 16 sequences per GPU | Measurable starting point; larger microbatches remain candidates |
| Accumulation | 4 microsteps | 8 × 16 × 4096 × 4 = **2,097,152 tokens/update**, 512 sequences |
| Projection backend | `fp8_te`, pinned ROCm TE HipKittens | Vendor implementation of forward, dgrad and FP32 wgrad; test against the independent AITER path |
| FP8 recipe | E4M3 FNUZ; activation/gradient scales 1×128; weight scales 128×128; FP32 scales | Preserve the existing explicit AMD numerical candidate; no per-tensor/delayed/MXFP8 substitution |
| Attention | `rocm_aiter`; its CK route for local layers | Preserve 20 local layers, exact 512-token window and four global layers |
| Head / residuals | BF16 streaming head, chunk 4096; BF16 residuals | Preserve tied-head and rounding semantics; this is not an all-FP8 model |
| Parameters / optimizer | FP32 master weights, accumulated gradients and state | Preserve spectral Muon and auxiliary AdamW; memory capacity makes state compression unnecessary |
| DDP | FP32 reductions, bucket cap 25 MiB, existing bucket views and `no_sync` accumulation | Known baseline; larger buckets are conditional experiments, not assumed wins |
| Memory execution | Retain activations; no full-model compile, graphs or rematerialization | Use the extra HBM to remove recomputation and reduce microsteps |
| Data / host | Immutable local mmap shards; prefetch 4 batches/rank; OMP 4 from pinned runtime | Keep S3 transfer and tokenization out of the training critical path |
| Reproducibility | Seed 1337, frozen source/image/data identities | Same initialization and global stream for comparisons |

The model geometry, local/global pattern, Peri-LN, learned Q/K gains, learned
value residual and optimizer algorithm are unchanged. The initial optimizer
rates remain Muon **0.017**, embedding AdamW **0.003**, ordinary norms
**0.0003**, Q/K norms **0.0001**. Preserve momentum 0.95, Adam betas
(0.9, 0.95), epsilon 1e-8, five NS iterations, logical Q/K/V and gate/up
updates, matrix weight decay 0.02, embedding decay 0 and global clipping 1.0.
No automatic linear/square-root LR scaling is applied to the new batch.

TE is the selected **lead to test**, not a claimed winner. Its advantage over
our initial AITER candidate is that its inspected API covers the complete
block-scaled backward, including FP32 wgrad [S6]. Our AITER path uses local
Triton for wgrad, and its large-M forward dispatch may also choose Triton.
Those facts justify comparing complete updates; they do not establish which
implementation is fastest. If TE cannot build or pass preflight, explicitly
generate `aiter` and qualify it. There is no silent backend fallback.
To retain another profile's batch while explicitly using AITER, pass e.g.
`--profile mb8 --candidate fp8`. The decision record captures that override.

## Why 2M tokens instead of the largest batch that fits

The published **SmolLM2-1.7B production config** uses 256 data-parallel ranks,
microbatch 4, sequence length 2048 and accumulation 1: exactly **2,097,152
tokens/update** [S20]. That is unusually relevant evidence for a 1.446B dense
model. Its optimizer, tokenizer, context and architecture differ, so this is
a scale reference rather than proof of optimality for CI-1.

The repository's earlier optimizer HPO/700M bridge used **393,216
tokens/update** [S23]. The selected batch is 5⅓ times larger. This is a real
transfer: momentum averages and decoupled weight decay act per update, and
there are fewer updates per token. Keeping the numerical settings fixed
preserves the optimizer definition, but does not preserve its trajectory.
The archived recipe is evidence for the comparison, not authority to replace
the selected production architecture.

MiniCPM's batch experiments and the gradient-noise-scale research show why
the largest batch fitting memory need not provide the best loss per compute
[S21–S22]. More memory allows **larger microbatches at fixed global batch**;
it does not establish that arbitrarily large global batches learn efficiently.
There is no verified Muon-specific batch scaling law here.

The older 25-layer H100 run allocated 55.5 GB at microbatch 4 and 87.4 GB at
8. A linear extrapolation gives roughly **151 GB at 16, 215 GB at 24 and
279 GB at 32**. These are screening numbers, not MI325X predictions: the
selected model has 24 layers, different attention windows and AMD kernels.
They do show why leaving nominal HBM unused at 16 does not establish that
32 will fit with checkpoint/validation headroom.

At the selected batch, each rank processes 262,144 tokens/update. Moving from
microbatch 8 to 16 reduces eight microsteps to four, with the same number of
optimizer updates and gradient reductions. Moving the *global* batch to 4M
halves update/reduction counts per token, but changes learning dynamics.
For intuition, all FP32 gradients are about 5.79 GB; an ideal ring all-reduce
on eight ranks transfers roughly 10.13 GB/rank/update. Actual bucket overlap,
algorithm and topology must be measured; this is traffic accounting.

## Learning-rate schedule and training horizon

Select **WSD**, with linear warmup over **1,006,632,960 tokens** (480 balanced
updates), a constant plateau, then cosine decay over the **final 10%** of the
manifest's usable tokens, ending at **0.1 × each starting rate**. The decay
length rounds up to a 4096-token boundary. The original data horizon and
mixture phases are never rewritten by the generator.

The roughly 1B warmup is an engineering choice: hundreds of updates at the
selected batch and more tokens than the archived ~250M warmup. It is not a
published CI-1 optimum. SmolLM2 used 2,000 warmup updates and a final 10%
decay [S20]; we borrow the latter duration as a starting point while retaining
CI-1's selected cosine shape and 0.1 floor. We do not copy SmolLM2's AdamW
rates, linear-to-zero decay or 2,000-step warmup.

The accounting quantum is 6,291,456 tokens. The configured warmup, bridge
boundary and operational cadences are even multiples of that quantum and
therefore divisible by all five reviewed global batches (393k, 1M, 2M,
3.146M and 4M). This gives exact matched-token milestones. It is accounting precision,
not a claim that those exact token counts are statistically optimal.

`purpose=training` requires a roughly 10B-token or larger manifest; a 10B
representative subset is sufficient for a bounded quality bridge. Each
comparison uses the same horizon, data, seed and schedule. Stop the bridge
at **4,026,531,840 tokens**, after warmup and before decay:

| Profile | Microbatch | Global tokens/update | Accumulation | Bridge stop update |
| --- | ---: | ---: | ---: | ---: |
| `batch_bridge` | 4 | 393,216 | 3 | 10,240 |
| `batch_bridge_mb12` | 12 | 393,216 | 1 | 10,240 |
| `batch_1m_mb16` | 16 | 1,048,576 | 2 | 3,840 |
| `balanced` | 16 | 2,097,152 | 4 | 1,920 |
| `large_batch` | 16 | 4,194,304 | 8 | 960 |
| `batch_3m_mb16` | 16 | 3,145,728 | 6 | 1,280 |
| `batch_3m_mb24` | 24 | 3,145,728 | 4 | 1,280 |

The two 3.146M profiles are a fixed-global-batch execution pair: they process
the same token stream with six or four microsteps respectively. Both also
change the global batch relative to `balanced`, so promotion needs a
matched-token and matched-GPU-hour quality bridge. This compares the selected
full architecture at different batches, not a small archived model against a
new full model. A short plateau bridge tests early transfer only; it cannot
certify final convergence.

The final **dense token horizon remains conditional on measured throughput**.
Choosing 2T in a file cannot make it finish. Reserve **320 of 5,760 GPU-hours**
for bring-up/qualification and initially budget **90%** productive duty cycle
for the remaining allocation. These are planning allowances to replace with
actual costs, not measured overheads:

| Dense horizon | Required complete-update tokens/s/GPU on eight ranks |
| --- | ---: |
| 800B | 45,389 |
| 900B | 51,062 |
| 1T | 56,736 |
| 1.5T | 85,103 |
| 2T | 113,471 |

The generator records the requirement for the supplied horizon and labels
feasibility unmeasured. Use the remaining allocation after testing, and do
not charge measured cadence twice if the input throughput already includes
it. There is no QAT allowance inside these dense-only forecasts. The schedule
and manifest horizon are part of exact-resume identity: changing the horizon
later requires a reviewed migration, not editing JSON mid-run.

## Monitoring, evaluation and recovery settings

| Operation | Token cadence | Balanced updates |
| --- | ---: | ---: |
| Scalar training log | 25,165,824 | 12 |
| Expensive stability/gradient scans | 201,326,592 | 96 |
| Validation | 503,316,480 | 240 |
| Committed checkpoint | 2,013,265,920 | 960 |

All comparisons keep **1,024 full sequences globally per language/domain**,
4,194,304 scored tokens per scope key. The generator derives the per-rank
validation batch count: 32/16/8/4 for microbatch 4/8/16/32, or 6 for
microbatch 24 on eight ranks. The trainer caps the selected global sequences
at 1,024, so the last microbatch-24 evaluation step contains eight sequences
per rank. The cap is an operational setting; the trainer's default behavior
for older configurations remains unchanged. It refuses short or missing
held-outs instead of silently changing coverage and preserves the exact
evenly spaced held-out sequence selection. Full staging still verifies actual
bytes, token IDs, tokenizer, provenance and train/validation coverage.

For ten scope keys this is 41.94M validation tokens per check. Measure the
actual forward-only cost. Stability scans run at a lower cadence than the
old every-ten-update default; smoke runs explicitly retain every-update
diagnostics. The default 20-update benchmark may contain no expensive scan,
so qualify the production cadence separately over at least **1,920 balanced
updates** (two periodic checkpoint boundaries). Do not describe its short
timing window as fully amortized production throughput.

At an illustrative 55k tokens/s/GPU, checkpoints are about 76 minutes apart,
excluding pauses. At 85k they are about 49 minutes apart. Select last-three
local recoverable checkpoints plus verified durable backups, with daily,
phase-boundary and final milestones retained long term. **Live verified S3
upload/retention coordination still needs implementation/testing**; the
existing retention command requires a stopped run. This policy is not an
automatic cleanup promise.

If every periodic checkpoint is archived, 2T produces roughly 994 payloads;
at an illustrative 13 GB each that is **12.9 TB**, plus final/extra saves.
The corresponding 1T archive is about 6.5 TB. Measure payload size and size
storage from the actual retention policy. A 6 TB working disk cannot hold
4 TB of token shards plus a whole month of unpruned checkpoints. AWS S3 can
be the durable archive; training still uses staged filesystem shards.

## Named execution comparisons and promotion order

All names below are supported by `prepare --profile`. They change only the
stated field(s) relative to `balanced` and derive accumulation/cadence.

| Profile | Difference | When to run |
| --- | --- | --- |
| `mb4` | microbatch 4, accumulation 16 | First conservative full-size memory/update check |
| `mb8` | microbatch 8, accumulation 8 | Memory fallback and fixed-global-batch speed control |
| `mb32` | microbatch 32, accumulation 2 | Only after measured full-update HBM headroom |
| `batch_bridge` | HPO global batch 393,216, microbatch 4, accumulation 3 | Conservative execution and quality control |
| `batch_bridge_mb12` | HPO global batch 393,216, microbatch 12, accumulation 1 | Fixed-global-batch execution comparison against `batch_bridge` (microbatch 4, accumulation 3) |
| `batch_1m_mb16` | global batch 1,048,576, accumulation 2 | Intermediate quality candidate to bridge HPO 393k and selected 2M |
| `batch_3m_mb16` | global batch 3,145,728, accumulation 6 | Quality and execution control for the 3.146M batch |
| `batch_3m_mb24` | global batch 3,145,728, microbatch 24, accumulation 4 | Conditional larger-microbatch candidate at the matched 3.146M batch |
| `aiter` | AITER forward/dgrad, local wgrad split-K 4 | Primary backend comparison/fallback |
| `bf16` | Removes projection FP8 quantization | Independent quality and timing control |
| `triton` | Local Triton forward/dgrad/wgrad, splits 4 | Diagnosis if vendor kernels disagree; not mandatory tuning spend |
| `ck_local` | Independent FA-CK for local layers | If local attention is a measured bottleneck |
| `ddp64`, `ddp128` | Bucket cap 64 or 128 MiB | If traces show exposed communication; cap is not a hard maximum tensor size |
| `single_gpu` | One rank, accumulation 32 | Bounded rank-equivalence control at the same 2M global batch |

The selected runtime has OMP 4 and no speculative RCCL channel/protocol
overrides. Select NUMA affinity from the actual GPU/CPU topology. Do not
enable GPU partitioning or lower precision gradients to chase utilization.
Use the pinned PyTorch 2.12 DDP behavior as the API reference [S24].

Require measured peak **reserved and allocated** memory through optimizer,
validation and checkpoint paths, and initially keep at least **15% of actual
usable HBM free**, including device-level usage visible in `amd-smi` that
PyTorch's allocator counters do not capture. Nominal 256 GB capacity is not
an activation-memory estimate.
Test `batch_3m_mb24` and `mb32` only when they can meet that margin. If `mb16`
fails, use `mb8` or `mb4` at the same global batch before considering
rematerialization. Head chunk 4096 and TE-owned split-K 0 stay fixed initially;
further sweeps need
profile evidence, as described in [OPTIMIZATION.md](OPTIMIZATION.md).

Order: native preflight → conservative full update → RCCL/rank correctness →
`mb8`/`balanced` and TE/AITER paired complete updates → conditional
`batch_3m_mb16`/`batch_3m_mb24`, `mb32`, bucket and attention tests → batch
and BF16 quality bridges → cadence, backup and exact-resume rehearsal →
measured horizon selection. Prefer at
least **3% median paired throughput improvement**, no repeated p95 regression
and the same quality/memory gates before paying for a more complex execution
choice. Three alternating 20-update pairs and a 40-update soak remain the
minimum short timing evidence. A 1% ambiguous speedup does not justify a
large extra search inside this allocation.

For the quality bridge, select a conservative initial guardrail of **0.02
nats/token maximum increase** for every task/language's upper paired-bootstrap
confidence bound, using the existing frozen reference-NLL evaluator and
`production.compare_eval --max-loss-increase 0.02`. This is our engineering
acceptance tolerance, not one inferred from a paper. Check full held-out
language/domain NLL curves, nonfinite counts, clipping and learned-gain trends
as well. A wide confidence interval is insufficient evidence: extend the
comparison or use a second seed, rather than loosen the threshold after
seeing the result. This does not claim generated accuracy or final quality.

Cap initial qualification at the **40 node-hours** corresponding to 320 GPU-hours:
roughly 8 runtime/native checks, 6 execution comparisons, up to 12 conditional
batch-comparison hours, 6 BF16 quality work, 4 recovery/cadence work, and 4
contingency. The 2M profile is tested first; a full-model 393k control is
optional. Reuse a trial's cadence/recovery evidence where it actually
exercises the required paths. Charge all eight reserved cards even during a
single-card check. If measured cost exceeds the envelope, drop low-priority
sweeps before reducing correctness checks or consuming the production budget
blindly. Return unspent qualification time to production.

## Generate the configurations without a GPU

Use a real external manifest. The tool copies its metadata, changes only
microbatch, and writes `manifest.json`, `config.json`, `selection.json` and
`env.sh` into a **new external directory**:

```bash
python3 -m production.amd.prepare \
  --manifest /external/data/manifest.json \
  --profile balanced --output-dir /external/mi325x/balanced
source /external/mi325x/balanced/env.sh
export DATA_ROOT=/external/data
python3 -m production.amd.preflight --config "$CONFIG" --count-only
python3 -m production.amd.staging --config "$CONFIG" --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" --output "$STAGING_RECEIPT"
```

For the first small-data bring-up, pass `--purpose smoke --profile mb4` with
a separately validated test manifest. It needs at least **26 full updates**
(54,525,952 tokens at the 2M global batch) and the same fixed held-out coverage.
Its schedule warms up for two updates and decays over the final update;
logging/stability run every update and checkpoint/validation every ten.
This exercises the model and operations; it is **not** a quality comparison
or a checkpoint to resume into the long schedule. The minimum smoke horizon
covers five warmup plus twenty measured updates; prepare at least 45 full
updates for a five-warmup/40-measured soak.

Each alternative needs a distinct output directory, its own relevant
receipts and a fresh run. Set `OUTPUT_DIR` to a fresh external directory for
each timing repeat. Do not reuse a completed `run/` directory. The generated
environment defines a single-node launch but does not create receipts, start
hardware, verify an S3 backup or claim data readiness. Follow
[README.md](README.md) and [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md) for the
human-run native steps. A `selection.json` is a configuration decision record,
not a passed acceptance receipt.

## Research basis

Primary research and source code were inspected on 2026-10-25; links and
limits are [S20–S24 in SOURCES.md](SOURCES.md). Vendor kernel support comes
from the existing pinned S3–S6 sources. No external model's speed, batch or
loss establishes MI325X performance for CI-1. TinyLlama's current public
training script was also inspected; its batch settings differ from commonly
quoted launch recipes, so it was not used to justify an asserted final
TinyLlama production batch. The concrete SmolLM2 YAML supplies the batch
evidence used here.
