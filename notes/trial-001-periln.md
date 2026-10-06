# Architecture trial: trial-001 — Peri-LN branch output normalization (learnable channel gains)

Written before training. Campaign `architecture-20261006`.

## Evidence and applicability

- **Terminus observation (measured, archived):** the Lean 150M dense pilot showed late-layer residual RMS migrating 0.61 (init) → 191.18 (0.53B tokens) → 34.37 (10B), and during ternary QAT re-growing to 48.28 with last-layer attention output RMS amplifying 3.09 → 20.76 (`experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md` §5–7). The same pilot's layer-0 attention gain collapsed 1.0 → 0.0165 and its branch output RMS fell to 0.076 vs FFN 3.18 — a branch-scale failure in the opposite direction. Both are *branch-output scale* pathologies the current pre-LN baseline leaves unconstrained. Caveat: the pilot ran norm_lr 0.01, no QK norm, dim 640, window 512; the campaign baseline (norm_lr 2e-4, learned_rms QK norm, dim 768, window 1024) may show weaker symptoms — the baseline screen's telemetry decides how much weight to give this mechanism.
- **External evidence (verified full text, accessed 2026-10-06):**
  - Peri-LN, arXiv 2502.02732v3 (ICML 2025): `y = x + Norm_out(Module(Norm_in(x)))` with learnable output γ. 400M/1.5B/3.2B models, 30B tokens, 5 seeds, seq 8192, Llama-3-style: best loss across LR sweeps and sizes; no gradient spikes where Pre-LN spikes; hidden-state variance stays balanced; Pre-LN top-100 activations exceed FP16 max from 0.5B tokens onward while Peri-LN never does (Fig. 11 — directly analogous to our residual migration into a BF16/FP8 pipeline). Fig. 39: freezing output γ removes the benefit — **the learnable channel gain is the mechanism**, not unit-scale forcing.
  - OLMo 2, arXiv 2501.00656v3 (COLM 2025): post-sublayer RMSNorm + QK-norm; grad-spike score 0.108 → 0.069; shipped at 7B/13B/32B.
  - Gemma 2, arXiv 2408.00118v3 §2: normalizes both input and output of each sub-layer at production scale.
  - Independent search trail: queries covered value-residual, entropy-collapse, logit-control, and residual-growth literature (full trail in the campaign research log). Competing mechanisms considered: Mix-LN (2412.13795 — early post-LN, diverged at 1B in third-party runs), DeepNorm (2203.00555 — post-norm, diverged at 1B in Mix-LN's Table 1), LNS/Curse-of-Depth (2502.05795 — zero-param 1/√depth LN scaling; partially redundant, no learnable capacity), LayerScale (2103.17239 — vision-only evidence). Peri-LN has the strongest verified multi-seed LM evidence at our scale class.
- **Why this is not the excluded SubLN repeat:** the campaign excludes the parameter-free forced-unit SubLN/branch-normalization stack (`docs/EXCLUDED_UNIT_NORMALIZATION.md`). The old Terminus SubLN used scalar or no learnable gain and was justified only as a safety clamp. This candidate uses **learnable per-channel gains** with a stated quality mechanism (preserve branch expressivity while bounding branch scale), prior multi-scale evidence that the learnable gain is load-bearing, and a predeclared falsifier. `use_subln` remains `false`; the new norms are added explicitly in editable model code and documented here.

## Mathematical specification

Per layer `l`, replacing the two residual updates of the baseline:

```
a_l  = Attn(RMSNorm_in(x_l))                    # unchanged attention block incl. o_proj
x'_l = x_l + g^A_l ⊙ RMS(a_l) · a_l / RMS(a_l)  # i.e. x'_l = x_l + OutNormA(a_l)
f_l  = FFN(RMSNorm_in2(x'_l))                   # unchanged SwiGLU block incl. down_proj
x_{l+1} = x'_l + OutNormF(f_l)
```

`OutNorm` = RMSNorm over the channel dim (eps 1e-5, FP32 internally, matching `terminus/model.py` RMSNorm) with **learnable channel gain** g ∈ R^768 init 1.0 (optimizer: `norms` group, norm_lr 2e-4, wd 0 — 1-D parameter, classified automatically). Tensors: a_l, f_l ∈ R^{B×T×768} BF16 in, BF16 out.

- Backward: gradients flow through the gain and the normalized branch; the residual identity path is unchanged. Expected effect: branch contribution scale is decoupled from weight scale, so Muon's orthogonalized updates to o_proj/down_proj control *direction* while g controls *magnitude*.
- Init/stability: at init, E‖OutNorm(branch)‖₂² per token ≈ 768·(mean g²) = 768 — larger early branch contribution than the baseline's residual_init_scale=0.1408 (which the output norm largely overrides for magnitude). Prediction derived from Peri-LN Figs. 4/11: residual RMS stays bounded O(1–10) across depth and training instead of the 0.61→191→34 migration.
- Cost: +2×768 params/layer (+38,400 total, inside envelope); 2 extra pointwise RMSNorm kernels/layer (fused_pointwise Triton kernel reused); +≈0.1% FLOPs; activation memory +2 BF16 branch buffers/layer if not fused-away (negligible at 200M); KV cache and attention math unchanged; inference adds two vector ops per layer.
- FP8/QAT interaction: output norms bound the scale of what enters the residual stream → pre-norm inputs to later projections stay better conditioned for 1×128 FP8 activation quantization; branch amplification during QAT (measured 3.09→20.76) is directly capped by OutNormA. Weights themselves are not rotated or rescaled; the dense→ternary mapping is unchanged in form.
- Which parameters receive gradients: the two new gain vectors per layer (norms group). All other parameterization unchanged.

## Prediction and experiment

- Predeclared predictions at matched 0.5B-token exposure vs baseline: (a) English held-out loss equal or lower (Peri-LN's 400M–3.2B evidence says lower; 0.5B tokens may be too early — treat "equal with better stability" as a pass to full); (b) max sampled residual RMS far below baseline if baseline migrates (layer-24 residual RMS stays < ~20 throughout); (c) layer-0 attention branch stays engaged (attention_output RMS same order as other layers; no gain collapse); (d) grad clipping frequency ≤ baseline; (e) throughput cost ≤ +3% step time; (f) FP8 input relative RMS error equal or lower.
- **Falsifiers:** val loss >0.5% worse than baseline at matched tokens; residual RMS migration unchanged; any output-norm gain collapsing to ~0 (would indicate the norm is being routed around, i.e. mechanism not operating); throughput regression >5%.
- Tests before training: CPU reference tests — shapes/dtype, causality (future-token perturbation invariance), gradient flow to gains and projections, forward_cached parity vs forward, loss parity with baseline at init when gains≡... (not identical — different parameterization; instead check loss is finite and sane at init), plus the existing suite.
- QAT probe: `run` **if** this candidate reaches a matched full run — mechanism is quantization-sensitive: it changes branch/residual scale distributions that feed both FP8 activations and (indirectly) the weight distributions that get ternarized. Predicted signal: lower immediate conversion loss than baseline and smaller late-layer attention-output amplification during the probe. Requires baseline probe first (control).

## Results

(To be filled after screen; see `notes/REVIEW_TEMPLATE.md` copies.)
