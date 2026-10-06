# Architecture trial: trial-002 — value residual learning (ResFormer-style, learnable)

Written before training. Campaign `architecture-20261006`.

## Evidence and applicability

- **Terminus observation (measured, archived):** the Lean 150M pilot showed persistent low-entropy attention samples (min-entropy <0.1 in 3.6% of readings, concentrated at layers 13/14 of 25) and a layer-0 attention branch whose output gain collapsed to 0.0165 (`FINAL_RUN_ANALYSIS.md`, `COMBINED_DENSE_QAT_SCALING_REVIEW.md` §5–6). The pilot also showed residual-state peaks (layer-24 residual RMS 191) — the ResFormer authors' Fig. 13 analysis ties exactly this cluster (attention sinks, value-state drains, residual peaks) to the missing value pathway.
- **External evidence (verified full text, accessed 2026-10-06):** ResFormer, arXiv 2410.17897v5, Eqns. 5–6, Tables 2/3/7/14, Figs. 2/3/13: `V'_n = λ_{n,1}·V_1 + λ_{n,2}·V_n` with the layer's own attention matrix (no second attention). 82M/180M/320M/468M + 1.6B models, 20B SlimPajama tokens, GPT-2 vocab, SwiGLU, RoPE: matches vanilla validation loss with **16.11% fewer parameters or 20.3% fewer tokens**; Learnable variant +1.7 avg downstream. Ablations: only V_1-sourced skips help; V_2 skips neutral, late-layer sources harmful; skip to *final* layers helps most; Q/K/A residuals hurt (Table 7). Adoption: modded-nanogpt speedrun uses a value-embedding variant (repo record, not paper). License of github.com/Zhou-Zhanchao/ResFormer not verified — equations reimplemented from the paper, no code copied.
- **Contrary/limits:** authors' own runs only (no independent large-scale replication); 20B-token regime vs our 2.5B full horizon; Sparse-ResFormer with large V_1 share on shallow layers hurts. Search trail and competing mechanisms (gated attention 2505.06708, differential attention 2410.05258, sink literature 2309.17453/2508.10925) recorded in the campaign research log; value residual chosen for zero-GEMM cost and direct anti-sink mechanism.

## Mathematical specification

For layer n ≥ 1 (0-indexed), with layer 0 unchanged:

```
V_1  = VProj_0(InNorm(x_0))                    ∈ R^{B×H_kv×T×128}   (computed once, reused)
V'_n = λ_{n,1}·V_1 + λ_{n,2}·V_n               λ ∈ R^2 per layer, init (0.5, 0.5), free scalars
A_n  = Attention(Q_n, K_n, V'_n)               # attention matrix unchanged
```

V carries no RoPE (values are unrotated in this architecture), so cross-layer reuse is position-aligned and causal: position t of V'_n mixes only position-t values. Local layers (window 1024) and global layers share the same V_1; the attention mask/window is untouched. Parameters: 2 scalars/layer × 24 layers = 48 (layer 0 has none), classified as 1-D `norms` group (norm_lr 2e-4, wd 0). NOTE: this LR is much smaller than the projection LRs; λ movement will be slow — the init at 0.5 provides the immediate effect and the screen tests the mechanism more than its learned tuning.

- Backward: gradients flow to V_1's projection from every layer (W^V_1 becomes multi-path → expect larger gradient share for layer-0 v_proj; watch it), to V_n, and to λ.
- Init/stability: at init λ=(0.5,0.5), V' is an average of two same-distribution random projections → variance unchanged in expectation vs baseline (Var(V') = 0.25+0.25 = 0.5 of sum variance; V_1, V_n independent at init → Var(V') = 0.5·(σ²+σ²)·... = σ²·(0.25+0.25)·2 = σ² — equal). Attention logits unchanged; no residual-stream change.
- Cost: +48 params; memory +1 V slab (B×2×4096×128 BF16 = 2MB at microbatch); no extra GEMMs/kernels — V' forms by one fused multiply-add before attention; throughput impact <1%. Inference: KV cache stores V' per layer (unchanged size); plus one rolling V_1 buffer (2MB at 4K context) — KV bytes/token per layer unchanged. Prefill/decode FLOPs unchanged except the V' add.
- FP8/QAT interaction: attention V path is BF16 (FA3), not FP8-quantized; unchanged. QAT watch item: W^V_1 ternary error now propagates to all layers (multi-path weight) — monitor v_proj ternary error at conversion; no mechanism change to the quantizer.
- Cached inference: model keeps a rolling V_1 buffer (capacity 4096); each layer mixes the tail aligned to its own cache window (local 1024 / global 4096), then caches V'_n as usual. Layer-0 cache behavior unchanged.

## Prediction and experiment

- Predeclared predictions at matched 0.5B exposure vs baseline: (a) English held-out loss lower or equal; ResFormer's equality-at-16%-fewer-params implies a real but small gain at equal params; at 0.5B tokens expect ≤1% — treat "equal + healthier attention telemetry" as pass-to-full; (b) fewer/less extreme min-entropy <0.1 samples in mid layers (the sink/concentration signal) IF the baseline screen reproduces the pilot's concentration; (c) layer-0 attention branch better engaged (higher attention_output RMS than baseline if baseline shows suppression), via the value-path gradient rather than norm gain; (d) throughput within 2% of baseline.
- **Falsifiers:** learned λ_{n,1} → ~0 across all layers (mechanism unused); val loss >0.5% worse at matched tokens; entropy/concentration telemetry unchanged AND loss unchanged (mechanism inert → reject); layer-0 v_proj gradient norm exploding (multi-path pathology).
- Tests before training: shapes/dtype, causality (position-t changes must not affect <t through the V_1 path), gradient flow to λ and both v projections, forward_cached parity vs full forward (window-boundary alignment of the V_1 buffer), identical output to baseline when λ=(0,1).
- QAT probe: `skip` unless full-stage telemetry shows the V_1 multi-path made v_proj_0 ternary error an outlier — the V path is not the measured conversion bottleneck (K is).

## Results

(To be filled after screen.)
