# Trial diagnostic review: trial-001 (Peri-LN) / screen / cycle 20261006T131819Z1c8dbdf4 — completed

## Outcome and evidence coverage

- Candidate commit `14d62aa` (branch trial/periln/001, +2 output RMSNorms per layer, learnable channel gains), baseline commit `948663d` (cycle 20261006T114631Zfaa18256). Both screens: 500,170,752 tokens, same stream/schedule. trial-001: 200,329,216 params (+38,400), 0.938 GPU-h, 1.46 VM-on h. Archive verified (`0ee8d44e…`), clean deallocation.
- Read: full events/stability/incidents (none)/eval/preflight/dmon for both trials plus `diagnostic_status_screen.json` comparison block. No missing probes.

## Quality and learning (matched exposure)

| tokens | baseline en val | trial-001 en val |
| --- | --- | --- |
| 98.3M | 4.6046 | 4.5915 |
| 196.6M | 3.5708 | 3.5770 |
| 294.9M | 3.3377 | 3.3270 |
| 393.2M | 3.1897 | 3.1813 |
| 491.5M | 3.1074 | 3.1020 |
| 500.2M (screen eval, 8 blocks) | **3.13222** | **3.13205** |

- Curve: trial-001 leads from ~295M onward by 0.2–0.3%; final screen eval is a dead tie (−0.006%, far inside the ±0.5% indifference band). Verdict: **quality-neutral at screen, slight curve-area edge**.

## Numerical and functional behavior (the mechanism test)

- **Residual migration eliminated**: layer-24 residual RMS 6.9 → 12.4, max 16.8 (baseline: 0.62 → 86.4, max 91.9 and still rising). Depth gradient flattened (layer-0 1.56 / layer-24 12.4 vs baseline 9.7 / 86.4). This is exactly the Peri-LN evidence prediction (2502.02732 Fig. 11) and it neutralizes the pilot's QAT-stage failure mode (residual re-growth + late-layer attention amplification).
- Raw branch outputs now carry scale (layer-0 FFN output RMS 45.7 pre-output-norm vs normed contribution to the residual) — the norm is doing the work, not suppressing branches: layer-0 attention output RMS 4.71 (baseline 0.447) — layer-0 attention is *more* engaged, not less.
- Grad norms smaller late (0.152 vs 0.433 median) — expected: output norms decouple branch magnitude from weight scale, so projection gradients shrink. Clipping: 36/128 sampled (baseline 28/128), last clip at update 360 vs baseline ~120 — slightly longer into the early-stable phase; not alarming, watching.
- Zero incidents; FP8 input error ~2.5% unchanged; QK gains stable.
- Attention entropy: no material change vs baseline (mid-layer concentration pattern persists — Peri-LN was not aimed at it).

## Cost

- 151.1K vs 153.0K tokens/s (−1.3%, inside the ≤3% prediction); same memory profile; +38K params.

## Judgment

| Finding | Sign | Evidence | Follow-up |
| --- | --- | --- | --- |
| Residual migration eliminated | positive | layer_24.residual rms 12.4 vs 86.4 | carry to full; this is the QAT-robustness case |
| Screen quality tie | neutral | eval 3.13205 vs 3.13222; curve edge from 295M | full horizon decides |
| Layer-0 attention more engaged | positive | layer_00 attention_output 4.71 vs 0.447 | consistent with mechanism |
| −1.3% throughput | negative (small) | events.jsonl tok/s | accepted tradeoff |
| Slightly extended early clipping | uncertain | 36 vs 28 sampled clips | monitor in full |

**Decision: trial-001 (Peri-LN) merits a full continuation** — not on screen quality (tie) but because its mechanism removes a measured failure mode that the archived pilot shows *grows into the dense→QAT stage*, with quality neutral-to-ahead and 1.3% speed cost. Next: screen trial-002 (value residual) to pick the single full-run candidate; if trial-002 clearly beats on quality (>0.5%), reconsider stacking 001+002 for the full slot.
