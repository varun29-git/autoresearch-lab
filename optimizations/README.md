# Stack throughput optimizations (staged for GPU validation)

Target: the trial-004 stack — 25 × 2304 (1.5B), Peri-LN branch output norms +
learnable value residual (`configs/terminus_25l_2304d_64k_h100_stack.json`).
The layer count is fixed at 25; nothing here changes the model's depth.

**Status: measured on the H100 NVL, 2026-10-08 (1.5B stack config, batch 2,
synthetic steps, 64×4096-token updates). All parity gates passed — every fused
op matches the eager math on GPU — but the first-cut implementations are
SLOWER than the existing tuned path:**

| variant | tok/s | vs baseline |
|---|---|---|
| baseline (committed stack config) | 36,048 | 1.000× |
| fused (boundary + value mix + fused head loss + batched NS) | 32,643 | 0.906× |
| fused + CUDA graphs | 31,179 | 0.865× |
| fused + FP8 head | 31,096 | 0.863× |

Baseline 36K tok/s ≈ 27.7 µs/token matches the quoted ~26 µs measurement, so
the harness is faithful; the regressions are real implementation gaps, not
numerics: the tiled head loss recomputes logits in backward (4 head-GEMM
passes vs the baseline's 3) and runs a Python tile loop per microstep; the
boundary kernels are unswept first cuts next to the repo's tuned norm kernels;
the batched Muon stacks pay extra memory traffic at only ~100 matrices. The
FP8 head's numerics are fine (0.03% relative loss delta) but its reference
path pays quantize+dequantize without the DeepGEMM FP8 GEMM swap. Report:
`~/terminus-architecture-control/benchmark-20261008-stack-h100.json` (kept out
of Git per repo rules). Next step before any wiring: profile the fused step
(`--event-profile`/torch profiler) and tune or redesign the slow pieces; the
physics headroom (147K tok/s ceiling) is unchanged.

Nothing here is wired into `terminus/`; wiring a certified optimization into
the training harness is a campaign-version change, per AGENTS.md.

## Next H100 pass: isolate the regression

`optimizations/h100_stack_benchmark.py` now scopes each candidate patch to one
full training benchmark, so the existing combined result can be decomposed.
Its default queue is `baseline,output_norm,block_boundary,value_mix,head_loss,
batched_muon,graphs,fused`. The `fused` row reproduces the previous combined
bundle; `fused_graphs` and `fused_fp8_head` remain selectable separately. The
report keeps individual step times, allocated/reserved memory, runtime
metadata, and variant order. Optional `--event-profile` records phase times;
optional `--profile-dir` writes operator traces. Event profiling requires a
variant list without CUDA graphs. For example, after the owner starts the H100
validation session:

```
python -m optimizations.h100_stack_benchmark \
  --variants baseline,output_norm,block_boundary,value_mix,head_loss,batched_muon,fused \
  --event-profile --output /external/stack-ablation.json
```

Keep `--output` and `--profile-dir` outside Git. This is a synthetic-step
performance probe; it has **not** been rerun on H100 after the ablation change.
Compare each component against a matched baseline, then use the GPU trace to
redesign only a measured bottleneck. Do not wire the slower combined path into
training. Follow the numerical and real-token gates in the HPO planning branch
before adopting a faster candidate.

## Why: the physics accounting

The H100 NVL's FP8 tensor-core floor for this model is ~5.9 µs/token
(9.79 GFLOPs at 1,671 TFLOP/s). Measured: ~26 µs/token. So ~77% of wall time is
not tensor-core work. Where it physically goes:

1. **Serial dependency chains** — norm → quantize → GEMM → cast → add → norm,
   ~250+ serialized kernel boundaries per microstep at 25 layers × 2 branches.
   Fix: fuse chains into single kernels (this folder's `stack_fusion.py`) and,
   longer term, into GEMM prologues/epilogues.
2. **Redundant FP8 quantization** — the same hidden state re-quantized by every
   consumer GEMM. Fix: quantize once per tensor per microstep
   (`activation_cache.py`; weights are already cached this way).
3. **Small-kernel launch gaps** — thousands of launches/microstep at ~3–5 µs
   dead time when serialized. Fix: CUDA-graph the whole microstep
   (`terminus/cuda_graph.py` exists; `gpu_check.py` validates capture on the
   stack; wiring into the FP8 training path is the harness change).
4. **Tile quantization waste** — K-dim splits in wgrad underfill SMs. Already
   addressed by split-K wgrad (`fp8_wgrad_splits`, DeepGEMM kernels).
5. **The loss path** — 65K-vocab cross entropy materializes per-chunk logits
   and softmax gradients; head GEMMs run BF16. Fix: `fused_head_loss.py`.

Perfect execution of today's math is ~147K tok/s (GEMMs at 80% of FP8 peak);
the 69.4K tok/s needed for 1.5T-in-6000h sits at 47% of that — inside physics.
The gap is engineering, and engineering is the fixable part.

## The stack's two additions, fused

The stack already combines both architecturally in `terminus/model.py`
(`branch_norm="peri"`, `value_residual="learned"`; tests green). What was not
fused is the *kernel* level — each addition ran as separate pointwise ops.
`stack_fusion.py` fixes that:

- `out_norm_residual` — Peri-LN branch output norm + residual add in one
  kernel; backward retains only the branch tensor and the inverse RMS.
- `value_mix` — V′ = λ₁V₁ + λ₂Vₙ in one kernel, with dV₁/dVₙ/dλ in one pass.
- `block_boundary` — **idea 3**: OutNorm(branch) + residual add + the next
  InNorm in one pass, custom backward retaining x′ (the residual stream, alive
  anyway), the branch, and both inverse-RMS vectors. Must be benchmarked as a
  full block on the H100 — forcing norm into a GEMM epilogue could hurt
  tensor-core efficiency, which is why this stays a standalone kernel first.

## The numbered ideas

| # | Idea | File | Notes |
|---|------|------|-------|
| 1 | FP32 owner per projection, FP8 replicas everywhere | `owner_optimizer.py` | Reduce-grad-to-owner, owner applies Muon once, broadcasts both FP8 layouts. ~25% less projection ring volume than FP32 all-reduce; >10 GiB/GPU replicated-state saving at 8 ranks (`ring_volume_bytes`, `replicated_state_bytes`). Needs real multi-rank NCCL validation and checkpoint/parity sign-off before use. |
| 2 | Output head as one training operation | `fused_head_loss.py` | Vocab-tiled exact FP32 log-sum-exp; no materialized logits; dgrad/wgrad in FP32 accumulators; `fp8_head=True` runs head tiles as FP8 E4M3 with FP32 softmax accumulation. FP8 head changes numerics — must pass a dense-and-QAT quality comparison before adoption. |
| 3 | Peri-LN memory cost at the block boundary | `stack_fusion.py` (`block_boundary`) | See above. |
| 4 | Faster shape at equal params | — | **Excluded.** We are not reducing the number of layers; the 20-layer 2560-wide variant is not implemented. Geometry stays a documented lever only. |
| 5 | Optimizer update + FP8 cache production as one pipeline | `update_fp8_pipeline.py` | `produce_weight_operands`: one scale pass → both FP8 layouts (ternary and dense modes), exactly the bundle idea 1 broadcasts. `batched_muon_update`: equal-shaped matrices share batched Newton–Schulz GEMMs (pinned `muon` implementation accepts stacked inputs). Complements idea 1; not an independent percentage. |

Plus `activation_cache.py` from leak 2 (quantize-once per activation per
microstep, keyed by tensor identity, cleared at microstep boundaries).

## GPU validation plan (the "test at the end")

1. `python -m optimizations.gpu_check` on the H100 — parity gates (fused vs
   eager stack math; fused vs streaming head loss; batched vs sequential
   Newton–Schulz; owner-vs-pinned-Muon at world_size=1; CUDA-graph capture of a
   stack microstep) plus timings. Any `[FAIL]` blocks wiring.
2. Multi-rank owner-optimizer check under `torchrun` (NCCL reduce-to-owner +
   replica broadcast, checkpoint gather/restore, global clipping parity against
   the DDP baseline over a few hundred steps).
3. FP8 head earns adoption only through the dense-and-QAT quality comparison
   (numerics change, per idea 2's own caveat).
4. Only then: wire certified pieces into `terminus/` behind config flags, as a
   new campaign version.
