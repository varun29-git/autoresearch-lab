# 500M-token matched ternary QAT probe — baseline vs trial-004 (stack)

**Owner-authorized follow-on measurement** (2026-10-07) beyond the completed campaign's 12 GPU-hour cap; runs used the protected probe's semantics at 5× length: fixed blockwise mean-abs ternary quantizer with STE, FP8 E4M3 activations unchanged, continuation data stream (no repeats, cursor carried), optimizer moments carried, all LRs ×0.5, WSD warmup 10% / decay 20% of the 500M phase. Parents: baseline full `948663d` and trial-004 full `8bd5523`, each at exactly 2,500,067,328 tokens.

## Results

| | baseline | trial-004 (stack) |
| --- | --- | --- |
| Dense final (2.5B) | 2.68445 | 2.65363 |
| **Immediate conversion loss** (zero updates) | **7.78** | **9.15** |
| QAT +56M / +252M / +449M | 3.030 / 2.911 / 2.868 | 3.034 / 2.894 / 2.853 |
| **QAT final (500M)** | 2.82651 | **2.81095** |
| QAT penalty vs dense | +0.142 | +0.157 |
| **Ternary endpoint comparison** | — | **−0.55% vs baseline** |

## Stability through conversion + QAT (the mechanism's test)

| layer-24 signal | baseline QAT | stack QAT |
| --- | --- | --- |
| residual RMS | 36.2 → 65.6 (peak 76.1) — migration continues | 5.5 → 11.1 (peak 12.2) — bounded |
| attention output RMS | 8.6 → 13.9 — amplifying | 1.75 → 1.96 — flat |
| FFN output RMS | 12.0 → 21.1 — growing | 5.8 → 6.0 — flat |

Zero incidents in both probes; clipping minimal in both (5 and 7 sampled updates).

## Reading

- The stack's dense advantage **survives ternary conversion**: 2.6536→2.8110 ends ahead of baseline's 2.6845→2.8265. The margin narrows (+1.15% dense → +0.55% ternary) because the stack's one-shot shock is larger.
- **Why the larger shock (mechanism):** Peri-LN's output norms let raw branch outputs carry large magnitudes (the norm rescales them into the residual), so o_proj/down_proj latent weights sit in a higher-dynamic-range regime that one-shot blockwise mean-abs ternarization clips harder. During QAT the STE adapts and the bounded residual stream wins the recovery.
- The baseline's QAT phase reproduced the pilot's documented failure pattern (late-layer amplification, residual re-growth); the stack's QAT phase was flat on every one of those channels — the architectural reason to prefer it for the production ternary stage.
- Caveats: 500M probe tokens is short of the 2B+ the pilot used; per ParetoQ (2502.02631) ternary QAT saturates only after ~30B tokens, so both endpoints improve with a longer QAT, and the penalty ranking could narrow further. BitCPM4's recipe (LR re-warmup, QAT ≥40% of tokens) is the stage-two reference.

## Artifacts

Blob prefix `architecture/qat500-probes-20261007/`: qat500-baseline (receipt `4a0a9211…`), qat500-trial-004 (receipt `e29b1150…`). Each archived with SHA-256 receipt, verified by independent download on the VM, and re-verified per-file from the controller. VM deallocated after verification.
