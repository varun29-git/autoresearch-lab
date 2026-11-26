# Terminus stage-one architecture campaign report — architecture-20261006c

**Status: COMPLETE.** All trials finished, all archives verified (VM-side + controller-side SHA-256), H100 deallocated.

Campaign: English-proxy architecture search, 200M envelope (190–210M), dense FP8 pretraining → (later) ternary QAT. Fixed: 65K tokenizer, data stream, Muon/WSD/optimizer policy, 393,216 tokens/update, 0.5B screen / 2.5B full horizons, 12 GPU-hour cap. Editable: `candidate.json`, `terminus/model.py`, new `architecture_*`/`test_architecture_*`/`notes/` files.

## Trials and commits

| Trial | Commit | Mechanism | Screen en loss @0.5B | Δ vs baseline | tok/s | Params | GPU-h (screen) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | `948663d` | unchanged 25L/768/6Q2KV/hd128, win1024, global every 5, learned-gain QK RMSNorm | 3.13222 | — | 153.0K | 200,290,816 | 0.933 |
| trial-001 | `14d62aa` | Peri-LN: learnable-channel-gain RMSNorm on each sublayer output | 3.13205 | −0.006% (tie) | 151.1K (−1.3%) | 200,329,216 | 0.938 |
| trial-006 | `ee71e32` | Value residual: V′ₙ = λ₁V₁ + λ₂Vₙ, λ init (0.5, 0.5), 2 scalars/layer ≥1 | 3.09097 | **−1.32%** | 150.6K (−1.6%) | 200,290,864 | 0.917 |
| trial-004 | `8bd5523` | Stack: Peri-LN + value residual | **3.08531** | **−1.50%** | 143.5K (−6.2%) | 200,329,264 | ~1.0 |

Failed attempts (0 tokens each, all archived and reviewed): trial-002 (tuple return vs protected stability hook), trial-005 (config guard vs native preflight's checkpoint parity), two baseline attempts (protected Muon group-key assertion — fixed as campaign version b/c; data-manifest off-by-one — fixed staging). Failure reviews: `notes/review-baseline-screen-20261006T095206Z.md`, `notes/review-trial-002-screen-failed.md`, `notes/review-trial-005-screen-failed.md`.

## Screen learning curves (English held-out, matched token exposure)

| tokens | baseline | trial-001 | trial-006 | trial-004 |
| --- | --- | --- | --- | --- |
| 98.3M | 4.6046 | 4.5915 | 4.5020 | 4.1116 |
| 196.6M | 3.5708 | 3.5770 | 3.4991 | 3.4344 |
| 294.9M | 3.3377 | 3.3270 | 3.2847 | 3.2598 |
| 393.2M | 3.1897 | 3.1813 | 3.1468 | 3.1299 |
| 491.5M | 3.1074 | 3.1020 | 3.0664 | (screen eval 3.08531 at 500.2M) |
| screen eval @500.2M | 3.13222 | 3.13205 | 3.09097 | 3.08531 |

## Stability findings (screen stage)

- **Residual-stream migration (the pilot's documented failure mode):** baseline layer-24 residual RMS 0.62 → 86.4 and still rising at 0.5B. trial-001: 6.9 → 12.4 (bounded). trial-004: 6.9 → 12.0 (bounded). Value residual alone does not address it (trial-006: 75.1).
- **Layer-0 attention engagement:** baseline 0.447 (weak but alive — the pilot's norm-LR collapse does not reproduce at norm_lr 2e-4); trial-006: 3.08; trial-004: 6.32 (best).
- **Attention logits:** learned-gain QK RMSNorm holds logit RMS ≈ 1.5–3.0 in all trials; QK gains stable (no runaway at qk_norm_lr 1e-4).
- **Entropy concentration:** mid-layer min-entropy dips (0.5–0.9, probe-limited) persist in baseline and trial-006; not worsened by any candidate.
- **FP8:** input relative RMS error ≤3.0% in all trials (benign). Zero nonfinite incidents in every completed trial. Clipping confined to warmup in all trials.
- **Value-residual λ:** drifted toward V₁ at depth in trial-006 (RMS 0.62 at layer 18); stays ≈0.5 in trial-004 (output norms remove the pressure).

## Full-stage results (2.5B tokens, matched schedule and stream)

| | baseline (`948663d`) | trial-004 stack (`8bd5523`) | Δ |
| --- | --- | --- | --- |
| Final English held-out eval | 2.68445 | **2.65363** | **−1.15%** |
| Last curve point (2.50B) | 2.6677 | 2.6400 | −1.04% |
| Layer-24 residual RMS (final) | 177.9 (peak 293.6) | **12.1 (peak 15.5)** | bounded |
| Layer-24 FFN / attn output RMS (final) | 71.8 / 18.8 | 14.5 / 3.15 | controlled |
| Incidents / nonfinite | none / none | none / none | — |
| Clipping | warmup only | warmup only | — |
| Throughput | 152.4K tok/s | 143.5K tok/s | −5.8% |
| Peak memory | 76.0 GiB | 86.9 GiB | +14% |
| Train-span GPU-hours (stage) | 3.65 | 3.88 | — |

Both fulls resumed their exact screen checkpoints; archives verified in Blob (receipts `63260c90…` baseline, `1c991d1b…` trial-004-full, R1 trial-004-screen), each hash-checked after an independent download. The stack's margin held through the stable and decay phases (−1.0 to −1.4% at every matched point) — this is not a screen artifact.

**Pareto shortlist (final):**
1. **trial-004 stack (Peri-LN + value residual) — RECOMMENDED INCUMBENT.** Best quality at both horizons; the only candidate that bounds the residual stream through full training (the pilot's documented QAT-stage failure mode). Cost: −5.8% train throughput, +11 GiB.
2. **trial-006 value residual — Pareto alternative.** −1.32% at screen, only −1.6% throughput cost, fewest moving parts; does not bound the residual stream (75.1 at screen). Choose if stage-three kernel work can't reclaim the stack's speed cost.
3. **trial-001 Peri-LN — stability ancestor.** Screen tie with the full residual fix; subsumed by the stack.
Baseline retained as the control.

## QAT probe

**Skipped — documented reason:** the matched probe pair (baseline + incumbent, 100M tokens each) needed ~0.7 GPU-h beyond a budget that ended at ≈11.7 of 12 GPU-hours; the predeclared precedence rule (the matched full pair outranks the probe) applied. The quantization-sensitive rationale is on file for stage two: the incumbent caps the branch-output amplification the pilot measured during QAT (attention output RMS 3.09→20.76); the stage-two probe protocol is unchanged (`--stage qat_probe`, baseline first, matched continuation stream, FP8 activations, fixed ternary quantizer, half-rate WSD).

## Research trail (controller-side, all sources accessed 2026-10-06)

Full log: `notes/research-log-20261006.md`. Headlines:
- The baseline already matches measured production winners (Gemma-3 hybrid+QK-norm; Spectra's ternary-from-scratch winners: pre-norm + learnable gains + dense embedding; MiniCPM4's FP8 recipe).
- Peri-LN (2502.02732) + OLMo-2 (2501.00656) + Gemma-2/3/4 pre+post norms → trial-001.
- ResFormer (2410.17897, 82M–1.6B/20B tokens; modded-nanogpt adoption) → trial-006.
- Qwen gated attention (2505.06708) → trial-003 (built, never screened: target pathology absent in baseline; SinkProbe 2609.08574 small-scale counter-evidence).
- Killed by evidence without spending GPU: softcap (Gemma-3 superseded it), H-BitLinear Hadamard (BitNet-v2 2504.18415: matters at ≤4-bit activations, ~neutral at 8-bit — we use FP8; their Table 4: fused rotation hurts ternary), SubLN (Spectra 2407.12327: learnable-gain pre-norm beats it), MQA-from-scratch (GQA paper App. A instability), factorized tied embedding (softmax-bottleneck 1711.03953), Mix-LN/DeepNorm (instability ≥1B), differential attention (needs custom kernels — stage three).
- ParetoQ (2502.02631) + Continual-QAT (2502.11895) + Scaling-laws-for-precision (2411.04330): the pilot's QAT latent drift is expected STE behavior; conversion timing/budget guidance recorded for stage two.

## Production-risk assessment (for the 1.5B/μP decision)

Full-stage evidence is now included below (trial-004 vs baseline at 2.5B):
1. **Larger width/depth:** Peri-LN's evidence base extends to 3.2B and its mechanism (bounded branch scale) is the one whose absence already bit the pilot harder at QAT time; value residual verified to 1.6B by its authors. μP notes: Muon's per-matrix √fan_in rule auto-rescales across width/FFN/head changes; init std does NOT transfer across depth (recompute residual_init_scale ≈ 1/√(2L) if depth changes); weight decay must be re-tuned at 1.5B (μP: regularization doesn't transfer; Muon paper: WD matters in overtrain regimes).
2. **Multilingual/code/math transfer:** untested by design (English proxy); the fixed 65K tokenizer's rare-token rows are a screen-stage "vocab tax" shared by all candidates (vocab-law 2407.13623: 65K ≫ 15K optimal at this budget) — finalist check required.
3. **Dense→ternary conversion:** pilot evidence: K-projection worst (ternary error ≤0.83, zeros ≤44.8%); conversion shock 2.42→10.5 recovered over ~200M tokens. Candidates' branch-scale control (Peri-LN) is the hypothesized mitigation; QAT probe or stage-two long QAT must confirm.
4. **Long-context:** screen is 4K, loss-only; window/cadence ablations are perplexity-insensitive (Gemma-3) — retrieval probes needed before any window change; pp-RoPE on global layers (Gemma 4) is the queued long-context candidate.
5. **Inference cost:** stack adds 2 pointwise norms + 1 add per layer (negligible FLOPs); KV cache unchanged (value residual stores mixed V per layer, same bytes/token; +2MB V₁ buffer at 4K). Throughput cost is on TRAINING (−6.2%): stage-three kernel fusion can reclaim most of it (the output norms fuse with existing pointwise kernels).

## Deferred candidate queue (stage two)

1. QAT-probe pair for the incumbent (quantization-sensitive mechanism on file).
2. Hadamard-rotated intermediate states — only if activations ever go ≤4-bit (BitNet-v2 evidence).
3. pp-RoPE (global-layer frequency truncation, p≈0.25) when long-context enters scope.
4. Qwen gated attention (built: commit on trial/gated-attention/003) — if larger-scale behavior revives its rationale.
5. Geometry axis (deep-thin 40L×640/hd64 vs wide 20L×896): MobileLLM-vs-Densing conflict unresolved; belongs to the 1.5B shape search with μP.
6. BitCPM4 recipe note (schedule, not architecture): LR re-warmup at QAT start is critical; QAT ≥40% of tokens ≈ from-scratch QAT.

## Cost ledger

- GPU-hours (protected meter, reconstructed from archived event timestamps; the authoritative on-VM ledger is preserved on the persistent disk): failed attempts 0.044 + baseline screen 0.933 + trial-001 screen 0.938 + trial-006 screen 0.917 + trial-004 screen ~1.05 + baseline full ~3.80 + trial-004 full ~4.03 ≈ **11.7 of 12 GPU-hours**. Cap respected; no QAT probe pair was run for exactly this reason.
- VM-on hours ≈ 13.7 (includes the staging/failure history documented in the reviews; the last full day ran one boot with 8-second inter-trial gaps).
- VM-on hours and the operational incident history (stale finalizer self-deallocation, missing az CLI, Muon group-key crash, manifest off-by-one, extension-wedged start LROs) are documented in the failure reviews; every incident produced a permanent controller/test gate.
- Controller upgrades delivered during the campaign (out-of-repo, repo untouched): async-start cycle driver, chained multi-trial driver with concurrent archiving, controller-side independent download+hash verification, local sparse-replica data-path validation, pre-launch gate suite.
