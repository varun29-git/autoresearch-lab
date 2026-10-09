# Unfinished speed experiments for the five-hour dense campaign

Base candidate: `1e8c0a4` (E3 packed learned Q/K norm+RoPE, E4 fused value mix, E9 batched Muon), 25 layers and 1,500,456,496 parameters. The three paired 20-update runs averaged 40,996 tokens/s for the candidate and 36,723 for the incumbent. An archived line from the 80-update soak reports 41,075 tokens/s and 55.5 GB peak allocated; its full `block.json` was not copied locally. Use the complete paired receipts as the auditable baseline until a new complete sustained receipt is saved.

At 524,288 valid tokens/update, 41,075 tokens/s means 12.76 s/update. The no-overhead minimum 1.5T rate requires 7.55 s/update, about **5.21 s less**; 2T requires 5.66 s/update, about **7.10 s less**. At 32 accumulation microsteps those gaps are roughly 163 and 222 ms/microstep. Production overhead and QAT make the required dense headroom larger. Small pointwise fusions alone cannot close this.

The previous instrumented diagnostic suggested the following **nonadditive** gaps, per 16,384-token microstep. Instrumentation includes launch gaps and did not match raw block time, so remeasure before committing to a long kernel project:

| Component | Observed | Optimistic service floor | Apparent gap |
| --- | ---: | ---: | ---: |
| DeepGEMM weight gradients | 67 ms | 26.5 ms | 40.5 ms |
| Forward + input-gradient GEMMs | 100 ms | 53 ms | 47 ms |
| Hybrid attention | 48 ms | 17.8 ms | 30.2 ms |
| FP8 conversion and layouts | 41 ms | ~20 ms | ~21 ms |
| Exact vocabulary head/loss | 31 ms | 17.8 ms | 13.2 ms |

These floors omit some real costs, and the gaps cannot simply be added. Attack whole-update time and use isolated timings only to locate the cause.

## Complete inventory from the original atlas

`P0` means start here if a short diagnostic confirms the exposed cost. `P1` is the next frontier or a dependency for P0. `P2` needs a measured trigger. A rejected implementation is evidence about that implementation, not a proof that the mathematical direction is exhausted.

| ID | State after the first campaign | Next falsifiable experiment | Priority |
| --- | --- | --- | --- |
| S0 cost map | Done, but diagnostic timings disagree with raw block time | One brief exact-shape event/CUDA trace for the current candidate; reconcile clocks, power, GEMM shapes and instrumented overhead | P0, brief |
| E1 tensor ownership/rematerialization | Only mb2/4/8 sweep done | Storage-ID lifetime map for V₁ and packed MLP inputs; free or rematerialize only if it enables a faster tile/grouped-wgrad schedule under 74 GB | P1 |
| E2 residual + Peri-LN fusion | Implemented and rejected when gain reduction erased the saving | Revisit only with a new reduction/producer-consumer design and a whole-update upper bound | P2 |
| E3 packed learned Q/K + RoPE | Implemented and kept | Further QKV gradient/layout work only if new profile shows exposed time; preserve shared gain reductions | P2 |
| E4 value residual | First fused value-mix kernel kept | Fuse V₀/Vl mixing into attention V-tile loads, including exact dV₀ fan-in and coefficient gradients; compare with materialized mix | P1 |
| E5 FP8 operand lifetimes | Quantizer switch only, no operand graph | Trace X/W/dY codes, scales, transpose orientations and lifetimes; generate/reuse exact consumer layouts and eliminate duplicate conversion | P0 |
| E6 shape-specific GEMM | Only split-K=4/8 toggles tried; both regressed | Tune DeepGEMM/CUTLASS/Triton dispatch for actual M/N/K of forward, dgrad and especially wgrad; isolate tile waste and launch/scale cost | P0 |
| E7 grouped/deferred wgrad | Untested | Bound memory for 2–4 accumulation microsteps; issue one larger exact wgrad from preserved FP8 operands while dgrad runs normally; compare gradients and update | P0 |
| E8 exact vocabulary loss | Loss chunk 2048/8192 toggles only | Measure head GEMM vs logits/softmax traffic; prototype exact tiled online-LSE loss with tied-head and embedding gradients | P1 |
| E9 Muon/Adam | Implemented; one-GPU speed was a wash | Keep parity evidence; measure benefit only at the production-local accumulation profile, not as a claimed single-GPU win | P2/later |
| E10 compile/graphs | Naive `cuda_graphs=true` regressed | Selective compilation or graph capture only if a fresh trace shows host gaps; preserve changing weights/data/LR and finite checks | P2 |
| E11 data/checkpoint | Data `next()` ~0.7 ms/update; checkpoint not measured | Defer input tuning; later measure production checkpoint and asynchronous archival costs against paid GPU-hours | P2/later |
| E12 hybrid attention | Untested | Count issued vs valid tiles for 20 local-window and 5 global layers; specialize exact mask/FA3 schedule and compare forward **and backward** | P0 |
| E13 distributed transfer | Analytical scenario only; contains stale anchors | Correct from sustained receipt; later test NCCL payload, bucket readiness, rank-sharded input and actual 32-GPU efficiency | P1 model, later hardware |
| E14 dependency-aware overlap | Untested | Trace resource and gradient readiness; overlap independent microbatch, wgrad or optimizer groups only with correct final clip/update ordering | P1 |
| E15 joint tensor schedule | Untested | Search small legal combinations of saved activations, FP8 layouts, grouped wgrad and microbatch under measured memory limit; GPU-test only best few | P1 |
| E16 persistent producer/consumer fusion | Untested | Prototype one exact gate/up → SwiGLU → FP8 pack → down or GEMM → norm/residual path after byte/time ceiling; require full-update win | P1 |
| N1 FP8 head/attention | Untested numerical branch | Only after exact-path ceiling: matched gradients and matched-token held-out quality before claiming any speed | P1/high risk |
| N2 activation compression | Untested numerical branch | Apply only to measured large, long-lived activations; compare recomputation, error and matched-token quality | P2/high risk |
| N3 ternary QAT representation | Deferred by user | Later separate QAT campaign; this five-hour GPU pass remains dense-only | Later |

## New mathematical directions are welcome

The table is not an exhaustive idea list. For each new idea, first write a concrete algebraic identity or dependency rearrangement, the saved bytes/FLOPs/launches, the unavoidable new work, and a falsifier. Promising families include:

- Batch independent microbatches as a **dependency graph** so one batch's attention/SFU work can overlap another's GEMM or weight-gradient work without stale parameters or changed token accounting.
- Form grouped weight gradients from the **same already-quantized FP8 blocks** across accumulation microsteps. Concatenating and requantizing under a different scale rule is a numerical change; preserve block boundaries or label the trial `N`.
- Turn the 20 local attention layers into exact block-sparse/window tile schedules that avoid masked tiles, and fuse value-residual mixing at V-tile consumption where the backward fan-in can be proved.
- Stream the tied vocabulary projection through online log-sum-exp and exact FP32 gradient accumulation, or develop a WGMMA/TMA pipeline that joins producer epilogues to their real consumers without inventing partial RMSNorm reductions.
- Search the actual Hopper tile and warp-group configurations for each fixed model shape, with a roofline/occupancy bound and a full-update A/B. A library default is a starting candidate, not a law of physics.

A novel numerical or optimizer idea is allowed as a separately labeled research branch only if the 25-layer model, unique-token budget and matched-token quality remain credible. Never count a precision shortcut, skipped gradient, changed objective or untrained branch as a speed win. If the best exact ideas cannot plausibly recover the required 5.21 s/update, escalate the numerical branch early enough to test it; do not spend five hours polishing 1% changes.
