# Trial diagnostic review: trial-004 (stack) / full / chain 20261007T034649Z50f2566e — completed, wins the campaign

## Outcome and evidence coverage

- trial-004 commit `8bd5523`, baseline commit `948663d`. Both full stages resume their exact screen checkpoints through 2,500,067,328 tokens on the identical English stream and WSD schedule. Baseline full: eval en **2.68445**. trial-004 full: eval en **2.65363** → **−1.15% at matched full horizon** (screen was −1.50%).
- Archives: baseline full (28 files, 3.72GB, receipt `63260c90…`, VM-side verify + controller download/hash check), trial-004 screen (R1, 15 files verified controller-side), trial-004 full (21 files, 3.74GB, receipt `1c991d1b…`, verified both sides). All under Blob prefix `architecture/20261007T034649Z50f2566e/`. VM deallocated after all verifications.
- Operational note: the chain driver stalled once (SSH session hang after baseline's completion); direct intervention launched trial-004's full within minutes and repaired/verified both archives. A design fix is queued for the driver: archive prefixes must be per-(trial, stage) and archives must never target a live-mutating trial dir (the R1/R2 receipt mismatch caught exactly this).

## Quality and learning (matched exposure, full horizon)

| tokens | baseline | trial-004 | Δ |
| --- | --- | --- | --- |
| 0.50B | 3.1041 | 3.0507 | −1.72% |
| 0.98B | 2.9335 | 2.8970 | −1.24% |
| 1.47B | 2.8747 | 2.8359 | −1.35% |
| 1.97B | 2.8344 | 2.7946 | −1.40% |
| 2.26B | 2.7468 | 2.7173 | −1.07% |
| 2.50B (curve) | 2.6677 | 2.6400 | −1.04% |
| final eval | 2.68445 | 2.65363 | **−1.15%** |

The stack leads at every matched point of the 2.5B run; the margin is stable ~−1.1 to −1.4% through the stable and decay phases. Curve area (optimization efficiency) also favors the stack at every checkpoint. No late regression in either run.

## Numerical and functional behavior (full horizon)

- **The headline stability result:** baseline layer-24 residual RMS ended at 177.9 (peak 293.6); trial-004 ended at 12.1 (peak 15.5). The Peri-LN output norms bound the residual stream across the entire 2.5B run — the mechanism transfers from screen to full exactly as the Peri-LN paper's FP16-overflow analysis predicts. Baseline layer-24 branch outputs: FFN 71.8 / attention 18.8 RMS at the end; trial-004: FFN 14.5 / attention 3.15.
- Layer-0 attention engaged throughout (trial-004 output RMS 13.8 final).
- Zero incidents, zero nonfinite probes in both full runs. Clipping confined to warmup (both). QK gains stable. FP8 input error ≤3.0% both.
- Value-residual λ in the stack: stable ≈0.50–0.56 (no drift; contrast trial-006's 0.62 deep-layer drift without output norms).
- Attention entropy (probe-limited): sharper early-layer focus in trial-004 (layer-1 min-entropy 0.17 final) — functional given the quality win, flagged for the finalist long-context checks.
- **Cost axes:** trial-004 trains at 143.5K tok/s vs baseline 152.4K (−5.8%); peak memory 86.9 GiB vs 76.0 GiB (+14%). Quality-per-GPU-hour still favors the stack at 2.5B (−1.15% loss for +6.1% time); at fixed wall-clock budget the trade is closer — recorded honestly in the Pareto table.

## Judgment

| Finding | Sign | Evidence | Follow-up |
| --- | --- | --- | --- |
| −1.15% held-out loss at matched 2.5B | positive | eval_full both trials | stage two: cross-domain + width |
| Residual stream bounded end-to-end | positive | layer_24.residual 12.1 vs 177.9 | removes the pilot's QAT-stage failure mode |
| −5.8% throughput, +11 GiB | negative | events/memory probes | stage-three kernel fusion (out-norms fuse with pointwise) |
| Sharper early-layer attention | uncertain | layer_01 min-entropy 0.17 | finalist long-context/retrieval probes |

**Decision: trial-004 (stack) is the stage-one incumbent and the recommended architecture for the 1.5B μP transfer.** trial-006 (value residual alone) is retained as the Pareto alternative (nearly the quality, +4.6% throughput, fewer moving parts); trial-001 (Peri-LN alone) is the stability-only ancestor. QAT probe pair: skipped — the 12 GPU-hour cap cannot fit it after the full pair (spent ≈11.7h); the quantization-sensitive rationale is documented for stage two.
