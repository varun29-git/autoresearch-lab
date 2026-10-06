# Architecture trial: baseline (screen → optional full → optional QAT probe)

Unmodified control for campaign `architecture-20261006`. This note is written before the first training result.

## Evidence and applicability

- The baseline is exactly `candidate.json` at the campaign base commit: 25 layers, dim 768, 6 Q heads / 2 KV heads (head_dim 128), SwiGLU ffn_dim 1920, sliding window 1024 with every 5th layer global, learned-gain per-head-dim QK RMSNorm (`qk_norm_mode="learned_rms"`), SubLN disabled, tied 65,536-token embedding, `residual_init_scale=0.1408` (≈1/√(2·25)), RoPE θ=10,000, dense FP8 (`weight_mode="dense"`, DeepGEMM blockwise, FP8 activations).
- Closest measured precedent: the archived Lean 150M dense-FP8 10B run (`experiments/terminus_lean_150m_10b/analysis/FINAL_RUN_ANALYSIS.md`) and its 2B-token ternary QAT continuation (`experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md`). Differences from that pilot: width 640→768, GQA 5:1→3:1, window 512→1024, QK norm none→learned_rms, norm LR 0.01→2e-4, English-only instead of the 8-language mix. None of these deltas has isolated evidence; the baseline screen is the first measurement of this exact configuration.
- Optimizer is fixed by policy: Muon (sqrt_fan_in rule) lr 0.003, AdamW 0.003 for the tied embedding, norm 2e-4, qk_norm 1e-4, weight decay 0.1, WSD (10% warmup, 20% cosine decay) over 2.5B tokens, 393,216 tokens/update.

## Mathematical specification

Standard pre-norm transformer as implemented in `terminus/model.py` at the base commit; no changes in this trial. Per layer: `h += o_proj(attn(RMSNorm(h)))`, `h += down_proj(swiglu(RMSNorm2(h)))`; attention with RoPE, learned-gain QK RMSNorm per head-dim, FA3 kernel with `window_size=(1023,0)` on local layers and full causal attention on every 5th layer; logits = tied embedding. Parameter count must land in [190M, 210M]; expected ≈ 201M (embedding 50.33M + per-layer ≈ 6.0M × 25 + norms).

## Prediction and experiment

- Throughput: the 150M pilot (window 512, no QK norm, dim 640) sustained ~236K tokens/s per-update median on one H100 NVL. This baseline is wider, doubles the local window, adds two KV heads and learned-gain QK norm; predict 150–200K tokens/s per-update median, i.e. 0.5B screen tokens in 0.7–0.95 h of training. Falsifier for the campaign plan: <100K tokens/s would make the 12 h cap binding and force fewer trials.
- Stability watchlist, from archived evidence: (a) layer-0 attention gain collapse (pilot: gain RMS →0.0165; suspect was norm_lr 0.01, now 2e-4 — predict no collapse, or much slower onset); (b) late-layer residual RMS migration (pilot peak 191, settled 34; predict lower peak with residual_init_scale and lower norm LR, watch layer-24); (c) sampled attention logit max (pilot 54 with no QK norm; predict <30 with learned_rms at qk_norm_lr 1e-4 — watch for gain runaway, the old affine-gain failure mode); (d) FP8 input relative RMS error ~2.4% median (predict similar; >5% would be new); (e) mid-layer attention min-entropy <0.1 readings (pilot layers 13/14; watch layers 12–15).
- English held-out loss: no calibrated prior at this width/data; the pilot reached 2.76 English at 10B tokens on the full mix with decay. This screen ends at 0.5B of a 2.5B schedule (no decay yet), so absolute loss is not comparable to the pilot; only the curve shape and stability matter.
- QAT probe: `skip` unless a candidate with a quantization-sensitive mechanism completes matched full runs; the baseline probe then runs first as the control.

## Results

(To be filled after the screen: commit, trial ID, tokens, GPU-hours, loss vector, stability events.)
