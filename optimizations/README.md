# Stack throughput optimizations (staged for GPU validation)

Target: the trial-004 stack — 25 × 2304 (1.5B), Peri-LN branch output norms +
learnable value residual (`configs/terminus_25l_2304d_64k_h100_stack.json`).
The layer count is fixed at 25; nothing here changes the model's depth.

**Status: CPU parity tests pass (`tests/test_optimizations_*.py`, 39 tests).
No throughput or GPU-numerics claim in this folder is verified yet — that is
what `python -m optimizations.gpu_check` on the H100 is for. Nothing here is
wired into `terminus/`; wiring a certified optimization into the training
harness is a campaign-version change, per AGENTS.md.**

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
