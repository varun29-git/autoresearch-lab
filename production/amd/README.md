# CI-1 on MI325X: native test handoff

**Start here before training:**
[pretraining_readiness/README.md](pretraining_readiness/README.md) tracks the
batch, dense-to-QAT and hardware decisions that must be settled before the
month-long run.

Prepared 2026-10-24 for **8 GPUs and 5,760 total GPU-hours (30 days)**.
This directory contains runnable migration candidates, native numerical gates,
real-data update measurement, staging/runtime receipts, and a scaling runbook.
It has **not** been executed on AMD hardware. No AMD throughput, long-run
quality, complete native build, or token-budget feasibility is certified.
The human launches and stops all paid hardware and commands below.

Read [RESEARCH.md](RESEARCH.md) for the code/hardware analysis,
[OPTIMIZATION.md](OPTIMIZATION.md) for the ordered tuning plan,
[BATCH_STABILITY.md](BATCH_STABILITY.md) for the fixed-LR batch and QAT
stability decision sequence,
[ACCEPTANCE.md](ACCEPTANCE.md) for the release gates, and
[SOURCES.md](SOURCES.md) for the checked upstream revisions and evidence limits.
The original H100 recipe in `production/recipe.py` remains unchanged.

Use [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md) for the ordered readiness list,
including production data, S3 backup integration and the native gates still due.
Use [CONFIG_SELECTION.md](CONFIG_SELECTION.md) for the researched eight-GPU
starting settings and named comparisons selected on 2026-10-25: microbatch 16,
accumulation 4, global batch 2,097,152, TE FP8, roughly 1B-token warmup and a
final 10% WSD decay. These choices await native fit, speed and quality evidence.
Microbatch 16 is not a measured optimum; the paired 3,145,728-token
`batch_3m_mb16`/`batch_3m_mb24` profiles test whether more HBM use reduces
complete-update time without conflating microbatch and global-batch effects.
At the HPO's 393,216-token global batch, `batch_bridge_mb12` uses one
12-sequence microstep per rank and can be compared with `batch_bridge`'s
three smaller microsteps without changing optimizer-update frequency.
For a final ternary model, [QAT_ALLOCATION.md](QAT_ALLOCATION.md) recommends
60% dense / 40% QAT by tokens and prices both stages together. Native AMD
ternary execution still needs implementation and qualification.

## What is selected, and what is a candidate

All candidates keep 24 layers, width 2304, FFN 5760, 18 query/6 KV heads,
head dimension 128, vocabulary 65,536, context 4096, local window 512 including
the current token, global layers 6/12/18/24, tied embeddings, Peri-LN,
learned Q/K gains, learned value residual, BF16 residuals, and exactly
**1,446,478,126 parameters**. Spectral Muon, five Newton–Schulz iterations,
logical Q/K/V and gate/up slices, auxiliary AdamW, LR starts, weight decay,
global token-mean loss, clipping, and checkpoint/data-cursor logic are reused.

| Config candidate | Projection forward/dgrad | Projection wgrad | Numerical change from H100 |
| --- | --- | --- | --- |
| `fp8` (independent comparison) | AITER block-scaled FP8 | Local Triton 1D×1D, FP32 output | E4M3 FNUZ/240 replaces E4M3 FN/448; explicit FP32 split-K |
| `fp8_triton` | Local Triton block-scaled FP8 | Same local Triton path | Same FNUZ recipe; independent GEMM implementation |
| `fp8_te` (selected lead to test) | Pinned ROCm TE HipKittens | TE 1D×1D, FP32 output | Same FNUZ operands and FP32 scales; TE owns scheduling |
| `bf16` | PyTorch BF16 GEMMs | PyTorch autograd | Removes FP8 activation/weight/gradient quantization; separate quality candidate |

FP8 scales remain 1×128 for activations and gradients and 128×128 for dense
weights. Wgrad independently quantizes transposed activations/gradients;
transposing already-quantized activation codes would be wrong. Master weights,
accumulated gradients and optimizer state remain FP32. The selected tied head
continues to use **BF16 streaming cross-entropy**, not the archived FP8 head.

`rocm_aiter` and `rocm_ck` are explicit attention choices. AITER's inspected
gfx942 assembly path excludes sliding windows: local layers use its CK route.
`--local-attention rocm_ck` permits an independent CK build for just those
layers. AITER BF16 conversion is explicitly RTNE. No attention/linear backend
silently falls back to a CPU/reference implementation.

These are numerical migration candidates for review against the selected
baseline. A faster candidate must pass matched-token quality checks before
the month-long run. The mixed-phase recommendation is a planning policy;
the AMD FP8 constructor still rejects ternary mode. Dense-to-QAT
quality/format migration is a separate implementation and qualification gate.

## Build without spending GPU time

The checked Docker Hub candidate is ROCm 7.14.1, Ubuntu 24.04, Python 3.12,
PyTorch 2.12.0, pinned by digest in [Dockerfile](Dockerfile). This avoids
inventing a ROCm/PyTorch 2.13 image tag from the pasted writeup. The build
retains the image's ROCm torch/Triton, pins AITER/FA-CK/TE/Muon source commits,
and captures the installed dependency set. Native compilation is **unrun**.

On a Linux x86-64 CPU build host with Docker:

```bash
docker build -f production/amd/Dockerfile -t ci1-mi325x:test .
```

`--build-arg INSTALL_TE=0` builds the AITER/Triton/BF16 candidates without TE.
Do not select `fp8_te` in that image. On an existing ROCm development image,
the equivalent build is:

```bash
RUNTIME_ROOT=/external/ci1-runtime INSTALL_TE=1 bash production/amd/build_runtime.sh
source /external/ci1-runtime/activate-runtime.sh
```

Keep source, compiler, image digest, package freeze, final built-image digest
and tuning files together outside Git. Resolve any `pip check` conflicts
before native tests. This is a source-audited build recipe, not a certified
binary distribution. Once built, freeze and reuse that image on all ranks.

The activation script/container puts AITER and Triton JIT caches outside the
installed source packages. Keep that separation: AITER otherwise writes its
build output into its package, changing the source identity after new shapes
compile. Preserve the warmed caches with the frozen runtime. The receipt
hashes installed source/binaries, including `aiter_meta`; it does not certify
arbitrary external cache contents or allow unreviewed source-directory overrides.

When the human supplies the MI325X host, expose `/dev/kfd` and `/dev/dri`,
the appropriate render/video group permissions, sufficient shared memory,
and the external data/output mounts to the container. Use the provider's
documented ROCm launch configuration. Do not assume a local Docker build
provides the host driver, xGMI topology or RDMA configuration.

## Prepare the data and configuration on CPU

Use the existing frozen tokenizer and immutable token shards. A 65,536-ID
vocabulary fits `uint16`; preserve token IDs, NFC/continuous-EOS packing,
mixture/phase order, deduplication, domain/language coverage and held-outs.
All manifests, configs, receipts, checkpoints, traces and tuning results must
be outside this repository. The manifest owns the data mixture and total
tokens. The selected-profile generator derives a new manifest changing only
its microbatch, plus the config, decision record and launcher environment:

```bash
python3 -m production.amd.prepare \
  --manifest /external/data/manifest.json \
  --profile balanced --output-dir /external/mi325x/balanced
source /external/mi325x/balanced/env.sh
export DATA_ROOT=/external/data
python3 -m production.amd.preflight --config "$CONFIG" --count-only
```

`balanced` requires a roughly 10B-token or larger manifest and 1,024 validation
sequences per language/domain. A representative 10B subset supports the
bounded quality bridge; it is not the final corpus. For small-data bring-up,
pass `--purpose smoke --profile mb4` using a separately validated test manifest
with at least 26 full global updates. Smoke has an explicit short schedule
and every-update diagnostics; do not use it as a convergence comparison.

Use distinct external output directories for `mb8`, `aiter`, `bf16` and the
other [named profiles](CONFIG_SELECTION.md). Each is an explicit run; there
is no automatic sweep. `balanced` selects TE and needs `INSTALL_TE=1`.
The lower-level `production.amd.build_config` remains available for explicit
experiments and retains its original `fp8` default; it does not apply the new
profile policies. Compare microbatch 8 and 16 first. If full-update memory
leaves headroom, compare `batch_3m_mb16` and `batch_3m_mb24` at the same
3.146M-token global batch; qualify that batch against the 2M baseline before
promotion. Test microbatch 32 only after a memory profile shows headroom.
Recreate data staging receipts for each derived manifest, even if its shards
are shared.

Before the paid interval, make shard/tokenizer **files** read-only on the
staged snapshot (or use an immutable read-only mount), then fully verify once
per node and candidate config:

```bash
python3 -m production.amd.staging \
  --config "$CONFIG" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
  --output "$STAGING_RECEIPT"
```

Staging performs the existing complete checksum/token-ID/tokenizer/provenance
validation. Startup checks all resolved file paths, inode/device, size, mode,
mtime and ctime against the receipt; it does not rescan terabytes eight times.
Replacements, writes, permission changes, config/source/manifest changes or
copying to another node require new receipts. Files must remain immutable
throughout training. This detects ordinary storage changes; it does not
attest against an administrator deliberately forging storage metadata.
On filesystems with unstable inode/ctime semantics, use the full validator
and establish a supported immutable snapshot scheme before production.

## Human-run native gates and measurement

Inside the frozen runtime on one GPU, first run:

```bash
python3 -m production.amd.preflight --config "$CONFIG" --output "$PREFLIGHT_REPORT"
python3 -m production.amd.runtime --output "$RUNTIME_REPORT"
```

Capture runtime **after** native imports/JIT initialization. The preflight
checks attention forward/backward and cached offsets at window boundaries,
FNUZ quantization/tails/nonfinite behavior, all production projection shapes,
FP32 wgrad, weight-cache refresh, learned packed Q/K gains, pointwise/loss
kernels, reduced model checkpoint/cached composition, and native batched
versus per-logical-matrix spectral Muon. It writes no passed receipt on error.
It does not allocate/train the full production model or certify convergence.

Optional focused probes, invoked individually after correctness passes:

```bash
python3 -m production.amd.probe --config "$CONFIG" --component linear \
  --microbatch 16 --output /external/probes/linear.json
python3 -m production.amd.probe --config "$CONFIG" --component attention \
  --microbatch 16 --output /external/probes/attention.json
python3 -m production.amd.probe --config "$CONFIG" --component head \
  --microbatch 16 --output /external/probes/head.json
python3 -m production.amd.probe --config "$CONFIG" --component hbm \
  --output /external/probes/hbm.json
```

The linear probe includes forward, both gradients and activation/gradient
quantization with a warm weight cache. It does not predict model tokens/s.
The HBM probe copies 4 GiB, exceeding LLC; its read+write bandwidth is not a
GEMM roofline or evidence about sustained clocks. Collect `amd-smi` telemetry
and `rocprofv3` traces in separate diagnostic windows; follow the installed
version's help for supported counters.

On the full node, validate collective sizes before a full-size update:

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  --module production.amd.collectives --output /external/probes/rccl.json
```

Then set `CONFIG`, `MANIFEST`, `DATA_ROOT`, `OUTPUT_DIR`, `RUNTIME_REPORT`,
`PREFLIGHT_REPORT`, `STAGING_RECEIPT`, `EXPECTED_WORLD_SIZE=8`, and
`EXPECTED_GLOBAL_BATCH_TOKENS` to the inspected values. A first real-update
measurement is:

```bash
bash production/amd/train.sh --warmup-updates 5 --benchmark-updates 20
```

Each measurement needs a fresh output directory and enough manifest tokens
for warmup plus measured updates. It trains the **full model** on real data
with spectral Muon, FP32 DDP reduction, no-sync accumulation, clipping, cache
invalidation and finite checks. Per-rank synchronized host times include
data waits and exposed communication. The summary sums the slowest rank's
time at each update and counts global tokens once. It records the global loss
in each rank's record, all training settings including stability monitoring,
p50/p95, and peak allocated/reserved memory. A final checkpoint and validation
are exercised separately and timed in `cadence.json`.

Use three alternating candidate/control 20-update pairs, then a 40-update
soak for finalists. Predeclare comparison order, seed, stream, global batch,
runtime, monitoring and tolerances. Repeated startup JIT is excluded from the
steady-state window but charged to the allocation budget. Production cadence
and recovery must pass separately. If the benchmark receives a stop request,
it aborts rather than issuing a complete measurement receipt.

After [ACCEPTANCE.md](ACCEPTANCE.md) is satisfied, the human can invoke normal
training with the same environment and a fresh output directory:

```bash
bash production/amd/train.sh
```

`--stop-after-updates N` supports a bounded training/recovery drill with a
committed final checkpoint. To resume, pass the exact committed checkpoint
with `--resume /external/run/checkpoints/step_00001000.pt`. Runtime, source,
world size, data identity and numerical candidate must match. Changing
backend or GPU count is a new reviewed migration, not an exact resume.
`production.status`, `production.checkpoints`, `production.evaluate` and
`production.compare_eval` remain the operating/evaluation tools.

## Forecast the month from measured rates

The example below is **dense-only**. For the recommended combined dense/QAT
program, use `python3 -m production.amd.phase_plan` with both phase rates and
the 480 GPU-hour total qualification reserve in [QAT_ALLOCATION.md](QAT_ALLOCATION.md).

```bash
python3 -m production.amd.budget --dense-rate 60000 \
  --reserve-gpu-hours 320 --duty-cycle 0.9
```

The displayed 60k rate is an **arithmetic example**, not an AMD measurement.
This scenario produces about **1.058T dense tokens**, below 1.5T. Required
gross averages are **72,338 tokens/s/GPU for 1.5T** and **96,451 for 2T**.
With the example reserve/duty cycle, the 1.5T requirement becomes **85,103**.
Increasing GPU count changes elapsed days, not tokens per GPU-hour.
Do not apply a second scaling-efficiency penalty to an already measured
8-rank rate. For a fixed mixed-phase target, pass both `--qat-tokens`
and its measured `--qat-rate`; QAT is not free additional time.

## Local validation status

CPU tests cover recipe preservation/count, FNUZ reference arithmetic and
full-backward wiring, accumulation, cache/inference refresh, budget accounting,
slowest-rank statistics, immutable-stage invalidation and receipt rejection.
Existing model, optimizer, training and production-operation regressions were
also run. See [ACCEPTANCE.md](ACCEPTANCE.md) for the final check record and the
explicit native checks left unrun. Nothing here launches a cloud VM.
