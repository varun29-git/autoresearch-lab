# New AMD optimization candidates and promotion tests

These changes are ready for native testing; none has a measured MI325X speedup.
The selected architecture, spectral Muon rule, BF16 residuals, FP32 latent
weights/optimizer state and ternary straight-through gradient remain intact.

## Fused RMSNorm/SwiGLU to FP8

Generate a candidate with `production.amd.build_config --fused-fp8-producers`.
It is **off by default** and valid only for the AMD FP8 projection paths.
The producers emit FNUZ row codes and column-major FP32 scales directly from
the BF16-rounded output held in registers. This eliminates a separate row
quantizer launch/read at those sites. BF16 outputs remain available for
backward and independently quantized transposed operands; reusing transposed
row codes for weight gradients would change the numerical recipe.

The AMD preflight now requires `fp8_producers` evidence for this flag: compare
norm/SwiGLU outputs, gradients and exact FP8 codes/scales with the separate
implementation across zero, small, ordinary and large inputs and ragged row
counts. Those native checks have not run here. Producer backward retains the
existing BF16 derivative kernels. CPU tests cover sidecar routing and gradients,
but cannot establish Triton compilation, register pressure or gfx942 execution.

## Actual-shape GEMM tile measurement

Prepare shapes on CPU:

```bash
python -m production.amd.autotune --config /external/config.json \
  --microbatch 16 --output /external/gemm-shapes.json
```

On an already available MI325X, the human explicitly adds `--run` and chooses a
fresh output to benchmark six tile candidates for the actual packed QKV,
attention output, gate/up and down projections, including forward, input
gradient and long-reduction weight gradient. Each weight-gradient shape also
tries split-K counts 1, 2, 4 and 8. Forward and input gradients stay at one
split. Correctness uses independent dequantized FP32 matmul and the existing
2.5% bound. A candidate that fails that bound or runs out of memory is not
selected. Measurements record synchronized timing samples, correctness, the
winning split-K count, source hashes and runtime/GPU identity.

The CPU plan lists those candidates and does not choose a tile or a split.
Without `CI1_AMD_GEMM_TUNING`, local GEMMs keep the config split and the
default tile. Set that variable to a schema-2 table in a **fresh process** and
the local Triton GEMM uses the measured split and tile for each recorded
shape. Forward and input-gradient rows must record split 1. A measured table
is rejected after identity, checksum, or split-K changes. Missing shapes
retain the default tile and the requested split. This tunes the local Triton
GEMMs: TE's own GEMMs are unaffected; AITER's local weight-gradient path can
benefit. Recapture runtime/preflight receipts and rerun full updates before
using a table for training. No MI325X speed has been measured.

## Promotion criteria

Compare unfused/fused producers and untuned/tuned tiles at fixed global batch
2,097,152, with candidate microbatch/accumulation pairs 8/8, 16/4 and (if it
fits) 32/2 on eight GPUs. Use matched data and the same warmup, precision and
monitoring cadence. Include backward, accumulation, clipping, optimizer,
communication, data wait, checkpoint capture and periodic validation. Record
slowest-rank time, p50/p95 update time, peak HBM, host RAM and end-to-end token
rate. Require numerical, dense/QAT quality and kill/restore gates before
promoting either optimization. Greater HBM occupancy alone is not a win.

The asynchronous checkpoint candidate is described in [OPERATIONS.md](OPERATIONS.md).
Measure its CPU-copy pause, background serialization/hash contention, upload
queue age and disk floor alongside throughput. Disable it with the builder's
`--synchronous-checkpoint` option if measured full-cadence performance is worse.
