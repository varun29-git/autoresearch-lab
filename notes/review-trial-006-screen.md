# Trial diagnostic review: trial-006 (value residual, ResFormer V1 skip) / screen / cycle 20261006T172553Ze65526d1 — completed, clear win

## Outcome and evidence coverage

- Candidate commit `ee71e32` (branch trial/value-residual/002; mechanism per notes/trial-002-value-residual.md), baseline commit `948663d`. Both screens: 500,170,752 tokens, same stream/schedule. trial-006: 200,290,864 params (+48 λ scalars), 0.917 GPU-h, 1.22 VM-on h. Archive verified (`1b1fe34c…`), clean deallocation.
- Operational note: this cycle survived a control-plane jam (wedged `enablevmaccess` VM extension blocking Azure start LROs) and a power-loss interruption at update 36; the run resumed from the interruption checkpoint with no tokens redone, and completed under the async-start driver. The extension was deleted; starts are non-blocking now.
- Read: full events/stability/eval records. `incidents.jsonl` absent (zero incidents).

## Quality and learning (matched exposure)

| tokens | baseline | trial-001 | trial-006 | Δ vs baseline |
| --- | --- | --- | --- | --- |
| 98.3M | 4.6046 | 4.5915 | 4.5020 | −2.23% |
| 196.6M | 3.5708 | 3.5770 | 3.4991 | −2.01% |
| 294.9M | 3.3377 | 3.3270 | 3.2847 | −1.59% |
| 393.2M | 3.1897 | 3.1813 | 3.1468 | −1.34% |
| 491.5M | 3.1074 | 3.1020 | 3.0664 | −1.32% |
| screen eval | **3.13222** | 3.13205 | **3.09097** | **−1.32%** |

A consistent win at every matched checkpoint — larger early, stabilizing around −1.3%. This exceeds the ±0.5% indifference band at every point. Consistent with ResFormer's equal-loss-at-16%-fewer-params claim (2410.17897): at equal params, expect a quality edge.

## Numerical and functional behavior

- Zero incidents; clipping confined to warmup (23 sampled, last at update 220); grad norms healthy (early 1.07 → late 0.16).
- **Mechanism confirmed operating:** value_lambda RMS at last sample — layer 1: 0.474, layer 6: 0.505, layer 12: 0.498, layer 18: 0.620, layer 24: 0.571 (init 0.5): the V₁ share is growing in deep layers, matching the paper's "skips to final layers help most" (their Table 7). Not inert (the note's falsifier was λ₁→0 everywhere — did not happen).
- Layer-0 attention more engaged than baseline: attention_output RMS 3.08 vs 0.447 at screen end; layer-0 attention/FFN amplitudes are now balanced (3.08 vs 2.99).
- Residual migration **not** addressed by this mechanism (layer-24 residual RMS 75.1) — as predicted; that axis belongs to trial-001.
- Attention entropy: concentration pattern persists, shifted layers (13/15/18 min-entropy 0.57–0.81). The probe is last-query/≤128-keys; not evidence the mechanism failed there, but no improvement claim either.
- Multi-path gradient to layer-0 v_proj healthy (update/weight 0.0066, no outlier); FP8 input error ≤2.9%; throughput 150.6K tok/s (−1.6% vs baseline).

## Judgment

| Finding | Sign | Evidence | Follow-up |
| --- | --- | --- | --- |
| −1.32% held-out loss at matched tokens, consistent across the curve | positive | eval_screen + validation events | full continuation |
| λ grows toward V₁ in deep layers | positive (mechanism active) | value_lambda parameter probes | none |
| Layer-0 branch balance restored | positive | layer_00 probes | none |
| Residual migration persists | negative (expected, out of scope for this mechanism) | layer_24.residual 75.1 | stack with trial-001 |
| Entropy concentration unmoved | uncertain | attention_sample probes | monitor at full |

**Decision:** trial-006 is the quality incumbent. Because trial-001 independently delivered its stability mechanism, screen the stack (trial-007 = Peri-LN + value residual, commit `8bd5523`) next; then the chained full pair tonight: baseline full, then the stack's full if it beats both parents at screen, else trial-006's full. Precedence rule (predeclared): if budget tightens, the full pair outranks the stack screen.
