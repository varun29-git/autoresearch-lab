# MI325X ternary FP8 QAT port: prepared for native testing

The H100 production path keeps FP32 latent projection weights. Once per optimizer
update, its versioned weight cache quantizes each 128×128 block using
`scale = max(mean(abs(weight)), 1e-6)` and round-to-nearest-even of
`weight / scale`, clamped to `−1, 0, +1`. The code tensor is FP8 E4M3 FN; its
FP32 scales are separate. Forward and input gradient GEMMs consume those
scaled codes. The weight gradient is an FP32 surrogate from independently
quantized transposed activations and output gradients; it updates the FP32
latent weight. This is an identity STE, including positions whose code is zero.

The AMD port preserves that rule and uses gfx942's native FP8 E4M3 FNUZ
format. Both formats represent `−1, 0, +1` exactly. One Triton CTA per
128×128 weight block reads the latent weight once and emits FNUZ codes and
FP32 scales directly. No BF16 or integer intermediate is written to HBM.
The existing AITER, Triton and TE block-scaled FP8 GEMM candidates consume
the codes. Transposed codes and scales are copied once per weight version;
the H100 implementation already avoids a combined quantize/transpose store
because it left unwritten lanes on its tested Triton build. The AMD cache
reuses both orientations across gradient accumulation. Wgrad remains FP32
output with the existing independent activation/gradient quantizers and
split-K reduction. The tied vocabulary head remains the selected BF16 path.

This is source code and CPU contract evidence, not MI325X acceptance. Native
Triton code generation, FNUZ values, TE/AITER interfaces, complete-update
rate, memory fit, optimizer trajectories and quality are unmeasured here.

## Prepare one phase-compatible config

Use the actual dense checkpoint's `tokens_seen` as `--phase-start-tokens`.
Keep its global batch, world size, model geometry, FP8 candidate, tokenizer
and ordered data stream. The builder requires a positive, batch-aligned phase
start and a warmup plus decay that fits the remaining manifest budget.
Its QAT optimizer is the documented starting candidate, not a measured winner:
Muon 0.0085, embedding AdamW 0.003, norm 0.00015 and Q/K norm 0.00005,
with the dense spectral rule, momentum, decay and clipping retained.

```bash
python3 -m production.amd.build_config \
  --manifest /external/ci1/manifest.json \
  --output /external/ci1/qat-config.json \
  --world-size 8 --candidate fp8_te --weight-mode ternary \
  --global-batch-tokens 2097152 --phase-start-tokens <DENSE_TOKENS_SEEN> \
  --warmup-tokens <QAT_WARMUP_TOKENS> --decay-tokens <QAT_DECAY_TOKENS>
```

Choose the warmup and decay from the intended QAT phase length using
[QAT_ALLOCATION.md](QAT_ALLOCATION.md), rounded to full updates. The inherited
configuration uses immediate full ternarization and a phase-relative WSD
cycle. The transition utility checks parent/child model and data contracts,
checkpoint update boundary, carried optimizer state, stream continuity and
token cursor. The configured phase start must match the parent checkpoint.

## Native qualification sequence

1. Build the pinned ROCm runtime, run the QAT preflight on an MI325X, and
   retain the model/source/runtime-bound receipt outside Git. It checks
   `−1/0/+1` codes, mean-absolute scales, zero and nonfinite blocks, both
   matrix orientations, forward, dgrad, FP32 STE wgrad including tiny
   gradients, cache invalidation, attention, reduced-model numerics and Muon.
2. Create a disposable QAT child with `python3 -m terminus.transition` using
   `--parent-run`, `--config`, `--manifest`, `--data-root`, `--output-dir`,
   `--world-size 8`, `--device cuda:0` and a reviewed `--data-policy`. Stage
   the child model separately; dense staging and preflight receipts do not
   certify a ternary child. Keep the dense checkpoint and immutable source.
3. Run a bounded full-architecture QAT update and a complete-update benchmark
   from that child's transition checkpoint. The AMD train entry point accepts
   `--resume` with `--benchmark-updates` for QAT. Compare `fp8_te`, `fp8`
   and `fp8_triton` under matched data and model state, including peak HBM,
   quantizer time, all-rank update time, scheduled validation/checkpoint cost,
   and absence of fallback GEMMs. Choose a candidate from measured results.
4. On eight ranks, compare parent/child token cursor, optimizer moments and
   rates, clipping, finite activations/logits and losses. Perform an exact
   within-QAT checkpoint/restore and a failed-worker recovery drill.
   Compare QAT held-out loss against both the dense parent and matched dense
   continuation. Do not infer quality from a successful kernel preflight.

The human starts and stops all paid GPU work. Keep manifests, data, configs,
receipts, reports, checkpoint payloads and generated outputs outside Git.
