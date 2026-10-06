# Trial diagnostic review: baseline / screen / cycle 20261006T114631Zfaa18256 — completed

## Outcome and evidence coverage

- Baseline commit `948663d` (campaign `architecture-20261006c`), screen stage, 500,170,752 tokens (1,272 updates), 200,290,816 params, 0.933 GPU-h, 1.47 VM-on h, archive verified (receipt `04733c98…`), clean deallocation. Blob prefix `architecture/20261006T114631Zfaa18256`.
- Read: `run/events.jsonl` (128 train / 6 validation records), `run/stability.jsonl` (129 sampled probe records), `run/incidents.jsonl` (empty), `eval_screen.json`, `diagnostic_status_screen.json`, `gpu_dmon_screen.log`, `preflight.json`. No missing or truncated probes.
- Native preflight passed; checkpoint integrity receipts verified by the worker (archive verification is hash-complete).

## Quality and learning

- English held-out curve: 4.6046 (98.3M) → 3.5708 (196.6M) → 3.3377 (294.9M) → 3.1897 (393.2M) → 3.1074 (491.5M); screen evaluation (8 spaced blocks × 32 seqs): **3.1322** at 500.2M tokens. Still descending at screen end; no late regression. This is the mid-schedule checkpoint of a 2.5B WSD run (no decay yet), so absolute loss is not directly comparable to the archived 10B-token pilot.
- Train loss by phase: early median 4.879 → mid 3.404 → late 3.148; train/validation relationship normal.

## Numerical and functional behavior

- **Zero incidents, zero nonfinite probes.**
- Gradient norms: early median 1.40 (max 16.7, warmup), mid 0.50, late 0.43. Clipping confined to warmup (all clipped sampled updates ≤ ~120). Healthy.
- **Layer-0 attention is ENGAGED** (attention_output RMS 0.037 → 0.447 and rising; layer-0 FFN RMS 9.62 dominates but attention is alive). The pilot's layer-0 gain collapse (→0.0165 at norm_lr 0.01) does **not** reproduce at norm_lr 2e-4. The pilot's own norm-LR hypothesis is supported.
- **Residual migration REPRODUCES:** layer-24 residual RMS 0.62 → 23.6 → 86.4, still rising at 0.5B (pilot: peak 191 at 0.53B, settled 34). Depth gradient is steep (layer-0 9.7, layer-10 33.5, layer-20 55.6, layer-24 86.4); layer-24 FFN output RMS 37.5 (max_abs 42). This is the clearest baseline weakness and directly matches Peri-LN's evidence target (2502.02732 Fig. 11).
- **Attention logits healthy:** learned-gain QK RMSNorm holds sampled logit RMS ≈ 1.5–3.0 across all layers (pilot without QK norm: max 54). QK gains stable (1.0 → 0.957/1.032/1.037 at sampled layers 0/12/24) — no runaway at qk_norm_lr 1e-4.
- **Attention concentration partially reproduces:** min-entropy dips at local mid layers (layer 10: 0.90, layer 11: 0.89, layer 12: 0.63, layer 17: 0.52; pmax up to 0.90). Weaker than the pilot's layers-13/14 pattern, probe-limited (last query, ≤128 keys).
- FP8 input relative RMS error ~2.5% across all projections (pilot: 2.43%) — benign and flat.
- Update/weight ratios small (max 0.0158 at layer-0 o_proj; no alerts of the pilot's 0.26 class).

## Cost

- 152.9–153.0K tokens/s median in all three phases (predicted 150–200K — on the low side as expected from width 768, window 1024, learned QK norm). Peak allocated 76.0 GiB (81.2 GiB peak during warmup probes) of 96 GiB. Screen cycle: 1.47 VM-on h (0.93 GPU-h). Full-stage projection: ~4.2 VM-on h (3.6 train + overhead).
- gpu_dmon: sampled during training (334 records); GPU util saturated (99% spot checks) — no duty-cycle anomaly.

## Judgment before the next experiment

| Finding | Sign | Evidence | Follow-up |
| --- | --- | --- | --- |
| Residual RMS migration at depth (86.4 rising at 0.5B) | negative | stability.jsonl layer_24.residual | **screen trial-001 (Peri-LN)** — its exact evidence target |
| Mid-layer low-entropy concentration (weak) | uncertain | attention_sample min-entropy layers 10–12/17 | screen trial-002 (value residual) if budget allows |
| Layer-0 attention engaged | positive | layer_00 attention_output RMS rising | weakens trial-003's rationale (nothing to rescue); deprioritize 003 |
| Logit scale controlled, QK gains stable | positive | attention_sample logits RMS 1.5–3.0 | none — learned_rms + qk_norm_lr 1e-4 validated |
| FP8 error benign (~2.5%) | positive | fp8_input probes | none |

**Decision:** run trial-001 (Peri-LN) screen next; then trial-002 (value residual). Trial-003 (gated attention) is demoted to reserve: its primary Terminus rationale (branch suppression) did not reproduce, its evidence base is large-scale with a small-scale replication failure, and its FFN trim is a confound. Full continuations only for screen winners vs baseline; QAT probe pair (baseline + Peri-LN) is justified by Peri-LN's quantization-sensitive mechanism if it reaches full and ≥2 VM-h remain.
