# Muon scaling for stack 004: decision and justified improvements

Reviewed 2026-10-10. This is research and a proposed dense-training recipe, not a trained result or an implemented optimizer. Target: the existing 25-layer Peri-LN/value-residual architecture, widths 768/1536/2304, fixed head dimension 128 and GQA ratio 3, with the completed dense speed implementation retained. QAT and production parallelization are later work. No GPU campaign was launched for this review.

**Decision:** retain the paper's width exponent. Improve the complete parameterization, decay/horizon accounting, and calibration of how updates affect this architecture. The best narrowly scoped experimental extension is fixed-radius retraction on attention-output and FFN-down matrices while retaining the paper's spectral update scale. It is mathematically motivated but has no Terminus training result. Published Hyperball is a useful comparator, not an automatic add-on to Muon µP.

**Implementation follow-up:** the user selected the direct `eta × sqrt(d_out/d_in)` convention, without Moonshot scaling or hidden base-width matching constants. It is implemented in `8153a78` on `hpo/muon-spectral`, based on the completed speed branch `8a62293`; see the [implementation and configuration note](https://github.com/varun29-git/terminus-autoresearch/blob/8153a78/docs/MUON_SPECTRAL_IMPLEMENTATION.md). New Muon configurations on that branch default to `spectral_unclamped`. Forty selected local optimizer/training tests passed; native H100 checks remain unrun. This implements hidden-matrix scaling only; the tied parameterization, decay experiments, retraction and Hyperball discussed below remain proposals. The base-width matching option below explains the paper's relative convention and is not an extra multiplier in the chosen implementation.

## 1. What is already well supported

The screenshot is Appendix G of Qiu et al., [Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](https://arxiv.org/abs/2512.05620v2). Sections 3, B.4, D.1, G and H, and both public code branches were inspected.

Let a logical projection have shape `m × n` = `fan_out × fan_in`, and let `Q` be the raw Newton–Schulz output, before shape scaling. The current Terminus combination is

`ΔW = −η × 0.2√max(m,n) × Q`.

The paper motivates a coefficient proportional to `√(m/n)`. Its [released trainer](https://github.com/charliezchen/scaling-matrix-preconditioning/blob/5a02d4f3a3532cf9b77ea6435cd1d5db278abf54/train.py#L256) implements **ratios relative to each matrix's base shape**, rather than requiring identical numerical constants for every layer:

`a_j(d) = a_j(base) × √[(m_j(d)/m_j(base)) / (n_j(d)/n_j(base))]`.

For our proportionally widened hidden projections, the ratio is one. Set `a_j(base)` to the current combined coefficient for each logical matrix if the first comparison should preserve the base-width optimizer. The corresponding correction to the existing width-growing coefficient is `√[fan_in(base)/fan_in(d)]`: 1, 0.7071, 0.5774 at widths 768, 1536, 2304. FFN-down has base fan-in 1920; most other projections have base fan-in 768. A single literal base fan-in applied indiscriminately to all roles is wrong. Q/K/V and gate/up are logical matrices even when packed into one tensor.

This supersedes treating the repository's extra `fan_in_base/fan_in` rule as the leading µP candidate. That rule remains historical experiment code, not the selected derivation. Do not silently change old optimizer/checkpoint semantics.

The exact RMS operator identity explains the exponent:

`RMS(ΔW x) ≤ √(n/m) ||ΔW||₂ RMS(x)`.

For `||Q||₂ = Θ(1)`, choosing `a = Θ(√(m/n))` gives width-independent control of output changes. At fixed aspect ratio, multiplying by another positive power of width loses this uniform bound; multiplying by a negative power forces the bound toward zero. The paper supplies the gradient/activation alignment argument needed to connect this bound with feature learning. A norm bound alone does not prove that every useful direction learns or that the finite-width best LR is exactly invariant.

This convention is not obscure:

| Inspected implementation | Actual option/formula |
| --- | --- |
| [PyTorch Muon](https://github.com/pytorch/pytorch/blob/2cb7870324d01b4b70100bac3bbff9ccd25304ed/torch/optim/_muon.py#L74) | `spectral_unclamped` = `√(m/n)`; `match_rms_adamw` = `0.2√max(m,n)` |
| [NVIDIA Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/a93fc65d5b468c4b5b646a0f96108b4031c32b40/emerging_optimizers/orthogonalized_optimizers/muon.py#L190) | `unit_rms_norm` = `√(m/n)`; its option named `spectral` instead returns `√max(m,n)`, with a separate extra scale |
| [Moonlight](https://github.com/MoonshotAI/Moonlight/blob/c2ad5b20c605086526a179d36901bfc41b52b44b/examples/toy_train.py#L142) | `0.2√max(m,n)` |
| [Qiu et al. larger-model trainer](https://github.com/charliezchen/scaling-matrix-preconditioning/blob/a5be0b1eb4d4c6ee885bddaf02beac0198d164a1/train.py#L293) | Per-role base shapes and relative input/output scaling; separate Adam embedding/readout groups |

At one fixed architecture, scalar shape factors can be absorbed into sufficiently flexible per-matrix LRs. The main value is affordable transfer between widths, not an automatic speedup of each training step.

## 2. Complete the tied parameterization before adding optimizer mechanisms

[Tensor Programs V](https://arxiv.org/abs/2203.03466), Table 8 and Appendix B, explicitly supports tied embeddings. A suitable starting arm for our auxiliary AdamW path is:

* `d0=768`, `r=d/d0`.
* Initialize the shared matrix `E` with variance `σ_E²/d0` at every width.
* Input lookup uses `E[token]`; output logits use `α_out × h Eᵀ/r`.
* Keep the shared matrix's AdamW base LR width-constant in these stored coordinates; tune that LR and, if necessary, the fixed output multiplier separately from hidden Muon LR.
* Keep vector/scalar base rates width-constant initially, retaining the known conservative norm and Q/K-gain rates. Check epsilon sensitivity when their gradients shrink.

At `r=1`, `σ_E=α_out=1` matches our current embedding initialization/readout. At other widths this changes the training parameterization and starts from initialization. Do not apply it to a trained checkpoint as if it were a lossless migration. The fixed head dimension means the usual attention `1/√128` factor has no width dependence in this family.

This is a principled starting combination of the tied AdamW rule and hidden Muon rule, not a theorem for every Peri-LN, GQA, value-residual and FP8 interaction. The readout multiplier must affect training loss, returned logits and cached inference consistently. It can be fused into existing pointwise computation; there is no reason to introduce another large matrix multiplication.

## 3. A stack-specific mathematical opportunity: normalized branch updates

In the selected model, each attention-output projection and FFN-down projection is immediately followed by Peri-LN. For a branch vector `z=Wx`, gain vector `g`, and `r_z=√(mean(z²)+ε)`, its output is

`y = diag(g) z/r_z`.

The exact first derivative is

`δy = diag(g)/r_z × [δz − z mean(z δz)/r_z²]`.

When `ε=0`, this projects out the component of `δz` parallel to `z`. A pure radial change `δz=c z` has zero first-order effect. Conversely, for the same matrix step, the angular effect is amplified when `r_z` is small. Thus weight/update RMS and a global spectral bound do not fully describe how fast a Peri-LN branch learns.

**Useful calibration improvement:** on occasional small samples of real activations, measure the change after normalization, and its contribution relative to the receiving residual stream. Use persistent drift to justify a local O/down LR adjustment. Do not force every layer to have equal changes, divide by a near-zero signal, or add an untested every-step adaptive controller. Q/K sensitivity should be assessed after Q/K normalization and through attention logits. The first-layer V source receives gradients from multiple later layers; its update must be judged after that aggregation, not split into independently optimized uses.

### Candidate: fixed-radius retraction while retaining spectral step scaling

For **only** `attn.o_proj.weight` and `ffn.down_proj.weight`, initially consider

`R_j = ||W_j(initial)||_F`

`W_trial = W − a_j(t) Q`

`W_next = R_j × W_trial / ||W_trial||_F`.

Keep the paper's coefficient `a_j(t)` and the existing Muon direction. Do not normalize `Q` by its Frobenius norm in this candidate. Use zero explicit weight decay on these radius-constrained matrices; decay inside the normalized trial point would instead alter the effective step coefficient. Other roles keep the reference recipe.

There is a specific mathematical justification, rather than just a norm heuristic. At `ε=0`, for any positive scalar `c`,

`PeriLN(c W_trial x) = PeriLN(W_trial x)`.

The retraction therefore leaves this step's branch function unchanged in ideal arithmetic. Every nonzero weight matrix has a functionally equivalent representative at radius `R_j`. Also, differentiating the retraction at `||W||_F=R` gives

`δW_retracted = δW − W〈W,δW〉/R²`.

It removes only a radial component that the following ideal normalization cannot observe. The proposed benefit is more predictable **future** effective learning rates, by preventing parameter radius drift. It does not create an immediate loss improvement by rescaling the current branch.

This is a **candidate**, not a demonstrated improvement or a novelty claim. Actual epsilon and FP8 rounding break exact invariance; verify that discrepancy before interpreting training. Fixing the radius can also remove beneficial implicit annealing, so give this arm its own LR neighborhood. Retain the ambient Muon momentum convention initially, as an explicit algorithm choice; retraction does not prove equivalence of future optimizer states to the unconstrained run. Store each logical matrix's radius if the candidate is implemented. The extra reductions are a cost to measure and fuse where useful, not assumed free.

Start with O/down because the invariance is direct in our computation graph. Do not blanket-constrain the tied embedding, value projections, or SwiGLU gate. In particular, changing one V scale changes its relative contribution to the learned value mixture; changing gate scale changes the nonhomogeneous SiLU. Those scales affect the function. Q/K are additional possibilities under their normalization, but are not needed for the first test.

## 4. Hyperball is evidence for this direction, but is a different step convention

Wen et al., [Hyperball Optimization](https://arxiv.org/abs/2606.16899v1), prescribe

`W_next = R × normalize_F(W − η_ang R normalize_F(Q))`.

The paper reports Qwen3-style experiments up to 1.2B, including QK-Norm, and 130M models trained up to 128B tokens. That latter range approaches 1,000 tokens/parameter and is more relevant to our long horizon than only 20-token/parameter experiments. It reports 20–30% **token-equivalent speedup relative to a fitted AdamW baseline** in the 1.2B experiments, while Muon with weight decay achieves roughly 10% there. Do not interpret this as another guaranteed 20–30% on top of Muon, or as measured H100 throughput. The paper also reports that Hyperball can start slightly worse and overtake during decay.

[NVIDIA's MuonHyperball](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/a93fc65d5b468c4b5b646a0f96108b4031c32b40/emerging_optimizers/orthogonalized_optimizers/muon_hyperball.py) implements the wrapper and explicitly labels it experimental. Its constructor requires a common supplied radius; our logical matrices have different radii, so it is not a drop-in replacement for the existing batched optimizer.

**Why not simply combine the formulas:** `normalize_F(sQ)=normalize_F(Q)` for every positive scalar `s`. Hyperball cancels any preceding scalar Moonshot or µP shape multiplier. Moreover,

`||η_ang R Q/||Q||_F||₂ = η_ang R/√stable_rank(Q)`.

Even a constant angular LR therefore has a spectral step that depends on effective rank and radius. Its empirical transfer result does not establish the same asymptotic width law as the paper in the screenshot. Use it as a separately tuned comparator. The selective retraction candidate above deliberately preserves the spectral coefficient to isolate radius control.

## 5. Treat decay and duration separately from width

Qiu et al. combine µP with inverse-width **independent per-step decay** in their compute-optimal experiments. Their 1.4× compute multiplier excludes optimizer operations, their large-model runs use 20 tokens/parameter, and their models have separate input/output matrices and no learned normalization gains. Their horizon grows with model size. This does not prove that inverse-width decay is optimal at a fixed token horizon, or for our ~1,000–1,333 tokens/parameter target.

Our current shrinkage per step is `1 − LR_group × WD_group`, which reduces to `1 − η_base(t) λ_base` under the existing shape compensation. Keep actual shrinkage explicit when changing the Muon coefficient.

An interpretable tuning coordinate is the **decay-only half-life in tokens**, `H`. For an update processing `B` valid tokens,

`λ_step = 1 − 2^(−B/H)`.

For a varying schedule, use the corresponding integrated decay rate over the tokens in that update. This is an exact composition identity for decay alone, not a theorem about the best `H` or total training dynamics. It allows batch changes to be tested without accidentally changing decay timescales. AdamW-group WD must be converted using the actual LR; do not divide by a zero warmup LR.

Likewise, `β(B)=exp(−B/τ)` preserves a nominal EMA memory length in tokens. It does not preserve gradient noise, Nesterov dynamics, or optimal momentum, and is a diagnostic comparison rather than a mandated production beta.

[Power Lines](https://arxiv.org/abs/2505.13738v2) finds that optimal decay timescales depend on tokens/parameter in AdamW. Its fitted exponents are not Muon constants. For our next search, tune width at fixed batch/horizon, then check a modest LR/decay neighborhood on a longer completed schedule. Longer checkpoints need not have the same best LR as five-minute runs. Do not spend the entire research budget fitting one universal exponent to a few early points.

## 6. Which other proposals deserve experiments

| Proposal | Evidence inspected | Decision for this stack |
| --- | --- | --- |
| [Hyperball](https://arxiv.org/abs/2606.16899v1) | 1.2B Qwen3-style, QK-Norm, and long 130M horizon; public optimizer implementation | Most relevant published comparator for radius control; separate angular-LR convention |
| [NorMuon](https://arxiv.org/abs/2510.05491v1) | 1.1B/5.4B, 50B tokens, 4K context, 2M-token batch; reported 11.31% step-efficiency improvement over Muon at 1.1B | Credible later direction if neuron update imbalance is observed. Their packed-QKV orthogonalization and scaling differ from ours; post-orthogonalization row normalization changes the spectrum |
| [MuonEq](https://arxiv.org/abs/2603.28254v2) | Pre-NS row/column equilibration; 130M–1B C4 runs up to 21B tokens; Ascend hardware | Later controlled direction if the NS input spectrum is problematic. It changes the direction; NPU timing and baseline LR choices do not establish H100 savings |
| [AF-Muon](https://arxiv.org/abs/2610.01395v1) | New tied-table optimizer; 124M–1B, studies up to ~10B tokens; released 2026-10-01 | Relevant but less mature. Keep auxiliary AdamW for the first coherent parameterization; replacing both tied-table and scalar optimizers would confound the comparison |
| [Muon++](https://arxiv.org/abs/2601.01306v2) | Projection intended to keep weight spectral norms fixed, contingent on singular-gap/step conditions | Do not adopt as default; it restricts useful directions and requires spectral-vector estimates. Its guarantee is not proof of better loss |
| [Spectral Scaling Laws of Muon](https://arxiv.org/abs/2606.04058v2) | Momentum-spectrum measurements in 77M–2.8B models, with layer-dependent finite-NS behavior | Supports inspecting actual NS responses; does not justify globally increasing NS iterations or declaring every small singular direction useful |

A simple counterexample explains the reservation about Muon++. Let `W=diag(2,1)`, `A=[[0,−1],[1,0]]`, `D=AW`, and `L(W)=−〈D,W〉`. The feasible path `W(t)=rotation(t)W` keeps spectral norm 2 and has loss derivative `−||D||_F²=−5` at zero. Muon++'s stronger two-sided projection with `P=diag(0,1)` gives `P(−D)P=0`, discarding that descent direction. This does not invalidate its constrained-update proof; it demonstrates why enforcing its sufficient conditions can be more restrictive than remaining on the spectral sphere itself. There is no basis to call it automatically better for our model.

The pinned five-step NS polynomial also does not make all singular values one. For normalized singular inputs `10^-4` and `10^-6`, its scalar response is about 0.0485 and 0.000485. Blindly amplifying these modes or normalizing by a tiny estimated singular value can amplify noise. Retain NS5 initially, sample actual updates later, and only test extra iterations or a different polynomial if loss-to-cost evidence supports it.

## 7. Minimal experiment order and deliverable

1. Establish a coherent paper-based dense arm with the tied readout, logical base shapes, calibrated auxiliary rates, and explicit decay convention. Reuse the current recipe as control. Calibrate at 200M and check LR neighbors at larger widths before asserting transfer.
2. Give the O/down-only retraction candidate one matched comparison with its own small LR neighborhood. Continue a promising candidate far enough to see the declared decay phase; the Hyperball literature makes early-only rejection particularly misleading here.
3. If radius control is useful, compare the published Hyperball convention on the same selected roles. Other direction-changing optimizers wait for a measured reason and remaining budget. This is a ranked menu, not a commitment to run every row.

Measure held-out loss at matched valid tokens and total GPU-hours, actual changes to normalized branches/logits, and optimizer overhead. Width transfer, absolute quality, and throughput are distinct outcomes. Better token efficiency does not increase the number of literal tokens that fit a fixed GPU-hour budget unless throughput also increases.

The first implementation decision can use the paper-based arm without any of the speculative extensions. The evidence does not justify claiming that a new formula beats it today. The added value of this review is a precise, architecture-specific challenger and a clearer interpretation of the tuning variables.

## Validation performed for this review

Read the selected model and batched optimizer, the referenced paper sections, both author-code branches, and the pinned framework sources above. Locally checked with NumPy: the RMS operator bound on square/tall/wide matrices; the full epsilon-aware RMSNorm derivative by finite differences; exact post-normalization invariance under positive radial retraction at epsilon zero; the retraction's tangent derivative; the Muon++ counterexample; composability of decay in token units; and scalar responses of the pinned NS polynomial. These are algebraic sanity checks, not LLM training experiments. No claim of improved Terminus loss, width transfer, FP8 parity, or GPU speed has been measured by this review.
