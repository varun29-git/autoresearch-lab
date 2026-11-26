# Trial diagnostic review: trial-004 (stack: Peri-LN + value residual) / screen / chain 20261007T001343Z — completed, wins the full slot

## Outcome and evidence coverage

- Candidate commit `8bd5523` (branch trial/stack/004; design note notes/trial-004-stack.md), 200,329,264 params. Screen: 500,170,752 tokens on the identical stream/schedule as all trials. Run inside chain 20261007T001343Z (one boot; trial-004 screen archived with SHA-256 receipt while the baseline full trained — archive concurrency verified in the driver tests; controller-side hash re-verification of all chain archives happens at chain end).
- Read: full events/stability records via the VM; incidents: none.

## Quality and learning (matched exposure)

| tokens | baseline | trial-001 (peri) | trial-006 (vres) | trial-004 (stack) | Δ vs baseline | Δ vs 006 |
| --- | --- | --- | --- | --- | --- | --- |
| 98.3M | 4.6046 | 4.5915 | 4.5020 | 4.1116 | −10.7% | −8.7% |
| 196.6M | 3.5708 | 3.5770 | 3.4991 | 3.4344 | −3.83% | −1.85% |
| 294.9M | 3.3377 | 3.3270 | 3.2847 | 3.2598 | −2.33% | −0.76% |
| 393.2M | 3.1897 | 3.1813 | 3.1468 | 3.1299 | −1.87% | −0.54% |
| screen eval | 3.13222 | 3.13205 | 3.09097 | **3.08531** | **−1.50%** | −0.18% |

The stack beats BOTH parents at every matched point. The early margin is huge (Peri-LN's fast-stable-start effect compounds with the value path) and compresses to −0.18% vs trial-006 by the screen eval — honest reading: the stack's edge over value-residual alone is small at 0.5B; the full horizon decides whether it persists.

## Numerical and functional behavior

- Peri-LN's mechanism survives composition: layer-24 residual RMS 6.9 → 12.0, max 15.4 (trial-001: 12.4/16.8; baseline: 86.4/91.9). Residual migration eliminated in the stack.
- Layer-0 attention the most engaged of all trials: output RMS 6.32 (baseline 0.447, trial-006 3.08).
- λ trajectory differs from trial-006's: value_lambda RMS stays ≈0.50 across layers 1/12/24 (trial-006 drifted to 0.62 deep). Interpretation: with branch output norms controlling scale, the model no longer leans harder on V₁ at depth — the two mechanisms interact benignly; the stack gets the quality of both without the λ drift.
- Zero incidents; clipping 25 sampled, last at update 250 (warmup); FP8 input error ≤2.97%.
- **Cost:** 143.5K tok/s vs baseline 153.0K (−6.2%) — the stack carries both parents' overhead (Peri-LN −1.3%, vres −1.6% do not fully add; shared pointwise fusion). This is the stack's main negative: −6.2% throughput for −1.50% screen loss.

## Judgment

| Finding | Sign | Evidence | Follow-up |
| --- | --- | --- | --- |
| Beats both parents at matched exposure | positive | curve table above | full continuation vs baseline |
| Residual migration eliminated (Peri-LN survives composition) | positive | layer_24.residual 12.0 | verify at 2.5B |
| λ drift absent with output norms | positive/uncertain | value_lambda probes | note for the report |
| −6.2% throughput | negative | events tok/s medians | include in Pareto cost axis; stage-three kernel work can reclaim it |

**Decision (predeclared rule: stack full only if it beats both parents at screen — satisfied):** tonight's chain runs baseline full, then trial-004 full. trial-006 remains the Pareto alternative (nearly the same quality, 4.6% faster, fewer moving parts); trial-001 is the stability-only ancestor.
