# Architecture trial: trial-003 — head-specific sigmoid output gate on attention (Qwen-style)

Written before training. Campaign `architecture-20261006`.

## Evidence and applicability

- **Terminus observation (measured, archived):** layer-0 attention branch suppression (gain → 0.0165, output RMS 0.076 vs FFN 3.18) and persistent mid-layer low-entropy attention (layers 13/14) in the Lean 150M pilot; late-layer attention output amplification during QAT (3.09 → 20.76). A learned, head-specific output gate gives the model a *linear-algebra-free* off-switch per head (rather than collapsing a shared norm gain) and bounds what enters o_proj.
- **External evidence (verified full text, accessed 2026-10-06):** "Gated Attention for LLMs", arXiv 2505.06708v1 (Qwen team): `Y' = σ(XW_θ) ⊙ SDPA(X)` with W_θ ∈ R^{d×d}, elementwise (head-specific), gate input = pre-attention normed hidden. 15A2B MoE + 1.7B dense, 3.5T tokens: **−0.2 PPL and +2 MMLU**, near-eliminates loss spikes, tolerates larger LR, first-token attention share 46.7% → 4.8% (sink relief). Mechanism attribution: non-linearity on the low-rank W_VW_O map + query-dependent sparsity. 30 gating variants ablated; the SDPA-output sigmoid gate was the best.
- **Contrary evidence (verified):** SinkProbe, arXiv 2609.08574v2 (Sep 2026): at small scale the gating sink-reduction effect **did not reproduce**; "the training objective produces the sink, not the architecture." So the sink claim is unproven at 200M — our falsifier watches sink/entropy telemetry directly, and the loss claim rests on the Qwen 1.7B result, not the sink story.
- **Not chosen:** differential attention (2410.05258 — needs custom kernels, stage three); softmax1/sink tokens (no published held-out-loss validation); attention logit softcapping (superseded by QK-norm per Gemma-3 2503.19786 §2.1).

## Mathematical specification

Per layer, after SDPA and before o_proj:

```
S    = SDPA(Q, K, V)                       ∈ R^{B×H×T×128}  (unchanged; FA3, local/global as baseline)
g    = σ( InNorm(x) · W_g )                W_g ∈ R^{768×768}, init N(0, 1/√768); ∈ R^{B×T×768}
Y'   = reshape(S) ⊙ g                      elementwise over merged heads, R^{B×T×768}
out  = o_proj(Y')                          unchanged (residual_init_scale 0.1408)
```

W_g is a `BitLinear` (dense FP8 DeepGEMM path like every projection, Muon optimizer group, wd 0.1 — consistent with qkv/o projections). FP8: the gate's input is the same normed hidden the qkv projection consumes (already FP8-quantized there); the gate output σ(·) ∈ (0,1) bounds Y' elementwise by |S| → o_proj's FP8 activation input cannot have outliers beyond S's own range; expected FP8 input relative error on o_proj ≤ baseline.

- **Parameter envelope control:** +768² × 25 = +14.75M would exceed 210M, so ffn_dim 1920 → **1664** (=13×128, DeepGEMM-aligned), −3×768×256×25 = −14.75M. Net ≈ 200.5M. **This trim is a confound**: SwiGLU ratio 2.5 → 2.17. If trial-003 loses narrowly, the FFN trim rather than the gate may be the cause; if it wins despite the trim, the gate is doing real work. Stated here before training.
- Backward: gradients to W_g through σ; to V/O paths modulated by g (a head can be driven to ~0 output without moving norm gains).
- Init/stability: at init g ≈ σ(0) = 0.5 → attention branch output halved at init (like an extra 0.5 on the attention residual contribution; benign given residual_init_scale).
- Cost: +1 FP8 GEMM per layer (768→768): attention GEMM FLOPs per token rise from 2×768×2048 to 2×768×2816 (+37.5% attention-side; ≈+9.8% of layer GEMM FLOPs); plus a sigmoid-mul pointwise. Serial kernel count +2/layer. Expect ~5–10% step-time cost; memory +gate activation buffer. KV cache and decode FLOPs unchanged (gate is per-token position-local; decode adds the same GEMV per token — inference cost rises proportionally to the GEMM, noted for production).
- QAT: W_g ternarizes like other projections (its input is the normed residual stream — the well-behaved FP8 input family per BitNet v2's observation); σ bounding Y' keeps o_proj input controlled during QAT (targets the measured 3.09→20.76 o_proj-input amplification). No quantizer change.

## Prediction and experiment

- Predictions at matched 0.5B exposure: (a) English held-out loss lower than baseline despite the FFN trim (Qwen evidence −0.2 PPL at 1.7B/3.5T; scale-down uncertain — this is the test); (b) fewer min-entropy <0.1 samples and lower first-token max-probability in mid layers IF baseline shows concentration (sink watch, given SinkProbe's counter-result); (c) layer-0 attention_output RMS not collapsed — the gate gives a cleaner off-switch so norm-gain collapse should be unnecessary; (d) grad clipping ≤ baseline; (e) step time ≤ +10%.
- **Falsifiers:** loss worse than baseline (cannot distinguish gate vs FFN trim alone — check per-layer gate statistics: if gates saturate near 0 or 1 uniformly, mechanism inert → reject); entropy/sink telemetry identical AND loss identical → inert, reject; step time >+12%.
- Tests before training: shapes/dtype; causality; gradient flow to W_g; gate statistics sane at init (mean ≈ 0.5); forward_cached parity; protocol envelope check.
- QAT probe: `conditional` — run only if (i) it reaches a matched full run AND (ii) baseline full shows late-layer attention-output amplification again; the gate directly bounds that tensor. Baseline probe first.

## Results

(To be filled after screen.)
