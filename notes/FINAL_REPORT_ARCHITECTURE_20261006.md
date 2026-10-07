# Terminus stage-one architecture campaign — final report

**Campaign:** `architecture-20261006c` (+ owner-authorized 500M QAT probe extension, 2026-10-07)
**Dates:** 2026-10-06 → 2026-10-07 · **GPU:** 1× Azure H100 NVL (Central US) · **Spend:** ≈11.7 of 12 protected GPU-hours (campaign) + ≈2.6h (owner-authorized QAT probe)
**Verdict:** the stack (Peri-LN output norms + value residual), commit `8bd5523`, is the recommended stage-one architecture.

---

## 1. Executive summary

Starting from the maintained Terminus baseline (25 layers, dim 768, 6Q/2KV heads, hybrid local(1024)/global attention, learned-gain QK RMSNorm, tied 65K embedding, Muon + WSD, dense FP8), the campaign screened three SOTA-derived mechanisms at 0.5B tokens and ran the decisive matched 2.5B-token comparison. The winning architecture adds two small mechanisms:

1. **Peri-LN**: a learnable-channel-gain RMSNorm on each sublayer's *output* (`y = x + OutNorm(Module(InNorm(x)))`), +38.4K params.
2. **Value residual** (ResFormer-style): each layer ≥1 mixes layer-0's values into its own (`V'n = λ1·V1 + λ2·Vn`, λ init (0.5,0.5)), +48 params.

At matched 2.5B-token exposure on the English proxy: **2.65363 vs 2.68445 held-out loss (−1.15%)**, with the residual stream bounded end-to-end (layer-24 RMS 12.1 vs baseline's 177.9). After conversion to ternary weights with FP8 activations and 500M QAT tokens: **2.81095 vs 2.82651 (−0.55%)** — the advantage survives quantization, and the stack's QAT-phase stability channels stay flat where the baseline's keep drifting.

## 2. Fixed controls (the yardstick, unchanged throughout)

Frozen 65K tokenizer (SHA-256 `c53e0e95…d29`); English-only prepared view of the ModelTunnelv2 10B token set (418 shards, 5.748B train / 29.2M val tokens, scope `modeltunnelv2-en-architecture-v1`, `source_mix={"en":1.0}`); Muon (pinned `f98f1cac`, per-matrix 0.2·√fan_in update-RMS rule) lr 0.003, AdamW 0.003 (embedding), norm 2e-4, qk_norm 1e-4, wd 0.1; WSD schedule (10% warmup, 20% cosine decay); 393,216 tokens/update; seq 4096; parameter envelope 190–210M; FP8 E4M3 blockwise (DeepGEMM) with FA3 attention. Screen = 0.5B tokens (a checkpoint of the 2.5B schedule, not a decayed short run); full = 2.5B; every comparison at matched token exposure.

## 3. Results

### 3.1 Screen stage (0.5B tokens)

| Trial | Commit | Mechanism | en loss | Δ vs baseline | tok/s | Params |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `948663d` | — | 3.13222 | — | 153.0K | 200,290,816 |
| trial-001 | `14d62aa` | Peri-LN | 3.13205 | −0.006% (tie) | 151.1K | 200,329,216 |
| trial-006 | `ee71e32` | value residual | 3.09097 | −1.32% | 150.6K | 200,290,864 |
| trial-004 | `8bd5523` | stack | **3.08531** | −1.50% | 143.5K | 200,329,264 |

### 3.2 Full stage (2.5B tokens, exact screen-checkpoint resumes)

| | baseline | trial-004 (stack) | Δ |
| --- | --- | --- | --- |
| Final held-out en loss | 2.68445 | **2.65363** | **−1.15%** |
| Median throughput | 152.4K tok/s | 143.5K tok/s | −5.8% |
| Peak memory | 76.0 GiB | 86.9 GiB | +14% |
| Incidents / nonfinite | none | none | — |
| Clipping | warmup only | warmup only | — |

Matched-curve detail (validation en loss): the stack led at every one of 26 matched checkpoints; margin −1.0% to −1.4% through stable and decay phases (early: −10.7% at 0.1B).

### 3.3 Ternary QAT probe (500M tokens per arm, fixed quantizer, FP8 activations, continuation stream, LRs ×0.5)

| | baseline | trial-004 (stack) |
| --- | --- | --- |
| Immediate conversion loss | 7.78 | 9.15 |
| QAT final (500M) | 2.82651 | **2.81095** |
| QAT penalty vs dense | +0.142 | +0.157 |
| layer-24 residual RMS during QAT | 36.2 → 65.6 (peak 76.1) | 5.5 → 11.1 (peak 12.2) |
| layer-24 attention output RMS during QAT | 8.6 → 13.9 (amplifying) | 1.75 → 1.96 (flat) |

The stack's dense advantage survives conversion. Its larger one-shot shock is mechanistic (output norms let branch weights sit in a higher-dynamic-range regime that one-shot blockwise mean-abs ternarization clips harder); the STE recovery closes most of it, and the stack's QAT-phase stability is categorically better — the pilot's documented late-layer amplification pattern (3.09→20.76 in the archived 2B QAT) is absent in the stack and present in the baseline.

## 4. Stability findings (all axes, all trials)

- **Residual migration:** the campaign's central measured weakness. Baseline layer-24 residual RMS: 0.62 → 86.4 (0.5B) → 177.9 final, peak 293.6. trial-001 and trial-004: bounded at ~12–15 through 2.5B. Value residual alone does not fix it (trial-006: 75.1 at screen).
- **Layer-0 attention:** alive in every campaign trial (the archived pilot's collapse was a norm-LR artifact: 0.01 then, 2e-4 now). Stack: strongly engaged (output RMS 13.8 at full end).
- **Attention logits:** learned-gain QK RMSNorm held logit RMS ≈1.5–3.0 in all trials; no gain runaway (qk_norm_lr 1e-4). No softcap needed.
- **Entropy concentration:** mid-layer min-entropy dips (probe-limited: last query, ≤128 keys) persist in all trials; not worsened by any candidate; unresolved as a functional question (flagged for finalist retrieval probes).
- **FP8:** input relative RMS error ≤3.0% everywhere; benign.
- **Incidents:** zero nonfinite events in every completed run (screens, fulls, probes).

## 5. Candidate analyses

- **trial-001 (Peri-LN)** — Screen tie, but eliminated residual migration and improved layer-0 engagement; subsumed by the stack. Evidence: Peri-LN (2502.02732, 400M–3.2B × 30B tokens × 5 seeds), OLMo-2 (2501.00656), Gemma-2/3/4 design practice. Notes: `notes/trial-001-periln.md`, `notes/review-trial-001-screen.md`.
- **trial-006 (value residual)** — Clear screen win (−1.32%), cheap (+48 params, −1.6% speed), mechanism confirmed active (λ drift toward V₁ at depth, matching the paper's late-layer-skip finding). Does not bound the residual stream. Retained as the Pareto alternative. Evidence: ResFormer (2410.17897, 82M–1.6B/20B tokens), modded-nanogpt adoption. Notes: `notes/trial-002-value-residual.md`, `notes/review-trial-006-screen.md`.
- **trial-004 (stack)** — beats both parents at every matched point of both horizons; composition benign (λ drift disappears when output norms control scale). **Recommended incumbent.** Notes: `notes/trial-004-stack.md`, `notes/review-trial-004-screen.md`, `notes/review-trial-004-full.md`.
- **trial-003 (Qwen gated attention)** — fully implemented and tested (commit on `trial/gated-attention/003`) but never screened: its target pathology (layer-0 branch suppression) did not reproduce in the baseline, and SinkProbe (2609.08574) failed to reproduce its sink effect at small scale. Shelved with code intact.

## 6. Research trail (all sources accessed 2026-10-06/07)

Full log with per-source verification status: `notes/research-log-20261006.md`. Includes: ParetoQ 2502.02631 (QAT below 3 bits is reconstruction; ~90/10 dense/QAT budget split), Continual-QAT 2502.11895 (earlier transition better; spike is transient), BitNet v2 2504.18415 (Hadamard matters at ≤4-bit activations only — killed that candidate), Spectra 2407.12327 (ternary-from-scratch winners = our norm choices), Scaling-laws-for-precision 2411.04330 (overtrained models quantize worse), KIVI 2402.02750 / KVQuant 2401.18079 (K-path channel outliers ≙ our K-projection ternary fragility), Muon-scalable 2502.16982 (update-RMS rule verified implemented exactly), tensor-programs 2203.03466 (μP transfer boundaries), vocab-law 2407.13623 (65K vocab is a deliberate over-investment at this budget), Gemma 2/3/4 (2408.00118, 2503.19786, 2607.02770), Qwen3 2505.09388, MiniCPM4/BitCPM4 2506.07900.

## 7. Production-risk assessment and 1.5B/μP handoff

1. **Width/depth transfer:** Muon's per-matrix rule auto-rescales LRs across width/FFN/head changes (verified in code and against 2502.16982). Init std does NOT transfer across depth — if the 1.5B shape changes depth, recompute `residual_init_scale ≈ 1/√(2L)`. Weight decay does not μP-transfer — re-tune at 1.5B (WD matters in overtrain regimes). The unused `sqrt_fan_in_ratio` rule in `terminus/optim.py` is the width-transfer study's comparison hook.
2. **Cross-domain (Indic/code/math):** untested by design; the 65K vocab is a deliberate over-investment on an English-only screen (vocab-law: ~15K optimal at this compute). First finalist check in stage two.
3. **Ternary conversion:** the stack is the better ternary citizen post-recovery, but its conversion shock is larger — consider a short warm conversion (gradual strength or brief re-warmup per BitCPM4) in the stage-two QAT recipe; QAT ≥40% of total tokens per BitCPM4's Fig. 5.
4. **Long-context:** 4K loss screens are blind to window/retrieval failure modes; Gemma-4's pp-RoPE on global layers is the queued candidate; run retrieval probes before any window change.
5. **Inference cost:** stack adds two pointwise norms + one add per layer (negligible FLOPs); KV cache bytes/token unchanged (mixed V cached per layer; +2MB rolling V₁ buffer at 4K). Training cost −5.8% tok/s, +11 GiB — stage-three kernel fusion of the output norms reclaims most of it.
6. **Memory watch at 1.5B:** 86.9/96 GiB at 200M — the stack's branch buffers need a memory check at width (gradient checkpointing or fusion) before the 1.5B run.

## 8. Stage-two queue (ranked)

1. Long matched dense→ternary QAT at the 1.5B shape (the production answer).
2. Cross-domain transfer suite (Indic/code/math) on the incumbent.
3. μP width bridge + WD re-tune; memory re-check.
4. Kernel fusion of the output norms (stage three).
5. Deferred candidates: pp-RoPE (long context), gated attention (if scale revives it), Hadamard (only if activations go ≤4-bit), geometry axis (deep-thin vs wide) in the shape search.

## 9. Reproducibility

- Code: branch `autoresearch/architecture-20261006` (campaign record + notes); trial branches `trial/periln/001`, `trial/value-residual/002`, `trial/stack/004`, `trial/gated-attention/003`. Exact trial commits in the tables above.
- Data view: `manifest.en.json` (English scope of the 2026-10-01 packed set) on the VM persistent disk `/srv/terminus-10b/architecture/`.
- Artifacts: Blob container `terminus-autoresearch` — campaign cycles under `architecture/<cycle-id>/`, QAT probes under `architecture/qat500-probes-20261007/`; every archive carries `artifact_receipt.json` (SHA-256 per file), was verified by independent VM-side re-download, and the analysis files were re-verified per-file from the controller. Checkpoints remain in Blob and on the persistent disk.
- Cost: ≈11.7/12 campaign GPU-hours + ≈2.6 probe GPU-hours; VM-on ≈ 16.3h total including the documented operational incidents.

## 10. Operational appendix (honest record)

Four pre-training failures occurred, each reviewed in `notes/` and each closed by a permanent gate: stale self-deallocating finalizer units (masked), missing Azure CLI on the VM (installed), a latent protected-code Muon group-key crash (behavior-preserving fix + regression test; campaign re-versioned b→c), a staged-manifest off-by-one (fixed; local sparse-file replica now validates the full data path offline), a candidate return-type violation of the protected stability-hook contract (fixed; monitor-active training-loop gate), and a candidate config guard that broke native preflight's checkpoint-parity gate (fixed properly — layer-0 uncheckpointed under value residual; parity test added). An Azure control-plane jam from a wedged VM extension was cleared; the controller now starts the VM asynchronously and chains trials in one boot with concurrent per-trial archival (drivers + tests in the controller directory, outside the repo). GPU-hour impact of all failures combined: 0.044h.
