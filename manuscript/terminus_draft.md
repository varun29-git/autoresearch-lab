# Terminus: A Peri-Normalized, Value-Residual Transformer for Dense-to-Ternary Training

**Working manuscript, 12 October 2026.** This draft describes the implemented
architecture and the research-stage training method. Production training,
multilingual evaluation, a packed inference artifact, and final cost
accounting have not been completed. Bracketed items below are manuscript
slots to fill from final, independently checked evidence; they are not
results.

## Abstract

We describe Terminus, a decoder-only transformer designed for dense FP8
pretraining followed by ternary-weight quantization-aware training (QAT).
Its residual blocks combine learned output normalization on each attention
and feed-forward branch with a learned cross-layer value residual. The
training implementation uses grouped-query attention, a mixture of local
and global attention layers, a tied embedding and classifier, and logical
matrix updates with spectral Muon scaling. We specify the architecture,
quantization rule, optimization groups, and the experimental controls used
to select them. **[Insert final model size and training-token count;
multilingual, downstream, and inference results; measured model-file size;
and end-to-end GPU-hour accounting after the production run.]**

## 1. Motivation and scope

Low-bit inference is attractive only if the compressed model retains useful
quality and the full training and deployment costs are accounted for.
Ternary weights reduce the information required by the projection matrices,
but the embedding, normalization parameters, scale metadata, and inference
runtime remain part of the model. We therefore treat dense learning,
dense-to-ternary conversion, quality after QAT, and actual packed storage as
separate measurements.

Terminus follows a two-stage training plan. It first learns dense weights
using FP8 projection arithmetic while retaining higher-precision master
weights. It then continues training with ternary projection weights in the
forward path and FP8 activations. The selected architecture addresses two
specific concerns: drift in the residual stream and the limited reuse of
low-layer attention values at greater depth. The component mechanisms are
related to Peri-LN [1] and ResFormer-style value residuals [2]. Our
contribution is their concrete composition, its quantization-aware training
path, and a measured account of its behavior at the final scale. The final
claim must be supported by the results in Sections 6–8.

## 2. Architecture

The planned production geometry has 25 blocks, width $D=2304$, feed-forward
width $F=5760$, 18 query heads, six key/value heads, and head dimension 128.
The 65,536-entry token embedding is tied to the output classifier. Context
length is 4,096 tokens. Four out of each five attention layers use a causal
1,024-token local window; every fifth layer uses full causal attention. The
configuration contains 1,500,456,496 trainable parameters. These are model
definition facts, not evidence of a completed production training run.

Let $R_\gamma(z)=\gamma\odot z/\sqrt{\operatorname{mean}(z^2)+\epsilon}$
denote RMS normalization over the last dimension, with learned gain
$\gamma$ and $\epsilon=10^{-5}$. For block $\ell$, write its input as
$x_\ell$. The attention and feed-forward residuals are

$$
u_\ell=R_{\gamma^{\rm in,a}_\ell}(x_\ell),\qquad
a_\ell=x_\ell+R_{\gamma^{\rm out,a}_\ell}
  \!\left(W^O_\ell\,\operatorname{Attn}_\ell(u_\ell)\right),
$$

$$
h_\ell=R_{\gamma^{\rm in,f}_\ell}(a_\ell),\qquad
x_{\ell+1}=a_\ell+R_{\gamma^{\rm out,f}_\ell}
  \!\left(W^D_\ell[\operatorname{SiLU}(W^G_\ell h_\ell)
                \odot(W^U_\ell h_\ell)]\right).
$$

The output normalizers are the Peri-LN component: they normalize each
branch's contribution immediately before addition to the residual stream.
The input and output gains are learned independently. The residual
projection weights receive the fixed initialization multiplier recorded in
the released configuration. The exact parameterization and epsilon should
accompany the model checkpoint.

Attention computes packed Q/K/V projections. Each query and key head has
its own learned-gain RMS normalization followed by rotary position
embedding; values bypass Q/K normalization. If $v_\ell$ is the projected
value tensor before attention, then

$$
\tilde v_0=v_0,\qquad
\tilde v_\ell=\lambda_{\ell,0}v_0+\lambda_{\ell,1}v_\ell,
\quad \ell\ge1,
$$

with both coefficients initialized to $0.5$ and learned without a simplex
constraint. The same position's layer-zero value reaches subsequent layers;
the local or global attention mask is applied after this mix. This path must
retain gradients to the first value projection. During cached inference, a
rolling buffer supplies the layer-zero values for newly decoded positions.

## 3. Dense training and optimization

Dense-stage projection operands use block-scaled FP8 E4M3 arithmetic through
DeepGEMM, including the selected full-FP8 projection backward path. The
residual stream uses BF16. Attention uses FlashAttention-3 where supported;
the output classifier uses a streaming cross-entropy computation while the
tied table remains an auxiliary AdamW parameter. FP32 master weights and
optimizer state are retained during training. FP8 arithmetic therefore does
not imply an FP8-only training checkpoint or an FP8 output head.

The packed Q/K/V matrix is partitioned into logical Q, K, and V slices for
Muon, and the packed gate/up matrix is partitioned into gate and up slices.
The attention-output and feed-forward down projections are separate logical
matrices. Let $U_\ell$ be the five-step Newton–Schulz direction for one
logical matrix with $d_{\rm out}$ rows and $d_{\rm in}$ columns. The selected
spectral update rule is

$$
W_{t+1}=(1-\eta_t\lambda)W_t
        -\eta_t\sqrt{d_{\rm out}/d_{\rm in}}\,U_t.
$$

The coefficient is applied to the direction for each logical matrix,
including packed slices; it is not the width-growing Moonshot
$0.2\sqrt{\max(d_{\rm in},d_{\rm out})}$ multiplier. The embedding/readout,
normalization gains, Q/K gains, and learned value-mix coefficients use
separately grouped auxiliary AdamW updates. Their learning rates and decay
are recorded as part of the final training recipe, not inferred from the
Muon matrix rule. In particular, tying the input embedding to the output
classifier prevents us from asserting full $\mu$P transfer for that table.

The final paper should distinguish three experiments: architecture
selection under the frozen earlier optimizer, later dense hyperparameter
research under this spectral Muon rule, and final production pretraining.
These are different optimizer contracts and their loss curves cannot be
merged into one ablation table.

## 4. Ternary continuation

During QAT the projection matrices retain master weights $W$. For each
128-by-128 weight block $B$, the current quantizer computes
$s_B=\max(\operatorname{mean}_{i,j}|W_{B,ij}|,10^{-6})$ and uses

$$
Q_B(W)=s_B\,\operatorname{clip}
  \left(\operatorname{round}(W_B/s_B),-1,1\right).
$$

The forward projection uses $Q_B(W)$ with FP8 activations; the training
implementation uses a straight-through gradient path to the master weight.
The tied embedding/readout and learned gains are not part of this ternary
projection set. The final QAT duration, transition schedule, and any change
in validation loss at conversion must be reported from the completed run.
An immediate conversion shock and its subsequent recovery are distinct
measurements.

## 5. Data and reproducibility

The architecture and hyperparameter screens used a fixed 65,536-token
tokenizer and a prepared English training/held-out view. Production data
mixing, deduplication, language proportions, and total distinct token
exposure must be documented from the final manifests. A proposed Indic data
release is separate from the model; its claimed token count, provenance,
licenses, quality filters, and train/evaluation overlap checks require a
dataset-specific audit before appearing as a result in this paper.

The final artifact record should identify the exact code commit, tokenizer
hash, data-manifest hashes, checkpoint lineage, dense and QAT token counts,
optimizer schedule, GPU type/topology, and paid GPU-hours. Aggregate token
counts must exclude padding, duplicated or replayed examples presented as
new exposure. The intended production resource envelope is 32 H100-class
GPUs and 6,000 aggregate GPU-hours across dense training and QAT; achieving
a token target within that envelope remains a measurement, not a premise.

## 6. Experimental design — results to insert

**Architecture.** Compare the fixed baseline, Peri-LN alone, value residual
alone, and their combination under identical optimizer, data order, token
exposure, and held-out scorer. Report loss trajectories, end-point loss,
training throughput, memory, clipping, and per-layer residual/attention
statistics. Dense and ternary-QAT comparisons must each have matched
continuation streams and clear checkpoint lineage. **[Insert externally
verified aggregate architecture results and a figure.]**

**Width and optimization.** Compare equal-token training at the smaller
research widths and production width. Report the exact Muon and AdamW group
rates, matrix and embedding decay, global batch, warmup, and validation
sets. A winner at one width does not establish a tied-embedding LR scaling
law. **[Insert width-transfer curves and any production-scale diagnostics.]**

**Final model quality.** Use held-out cross-entropy by language and domain,
then report downstream tasks with task name, language, prompt format,
evaluation harness version, and contamination checks. Include dense/QAT
quality deltas and failure modes, not only aggregate scores. **[Insert final
benchmark tables after checkpoint evaluation.]**

**Systems.** Measure valid tokens per second over complete updates on the
actual 32-GPU topology, including communication, validation, checkpointing,
and dense/QAT allocation. Compare measured throughput with a shape-specific
compute and memory model; do not substitute hardware peak FLOPs for a
training measurement. **[Insert matched baselines and full-run GPU-hour
accounting.]**

## 7. Storage and inference accounting

The present 1.5B configuration contains 1,349,222,400 projection-weight
parameters eligible for ternary storage, 150,994,944 tied embedding/readout
parameters, and 239,152 other parameters. At a simple two-bit code per
ternary projection parameter, the projections alone require 337,305,600
bytes. Keeping the tied embedding/readout in BF16 requires a further
301,989,888 bytes; the other BF16 parameters add 478,304 bytes. Thus this
representation requires **at least 639,773,792 bytes before block scales,
file headers, alignment, and runtime state**. A fixed-rate base-3 encoding
at $\log_2 3$ bits per projection weight would still be about 570 MB with
the BF16 tied table; entropy coding could be smaller if the learned symbol
frequencies allow it, but no such artifact has been measured.

These are arithmetic bounds, not a measured exported model. A roughly
400 MB model-file claim would require a separately implemented and
quality-tested compression of the tied table and an exact measurement of
all scale/metadata bytes. Runtime RAM, including activation and KV caches,
must be reported separately from on-disk weight size. **[Insert packed
artifact format, exported-file bytes, load-time RAM, and inference
throughput after implementation.]**

## 8. Limitations and conclusions — finalize after training

The current research evidence is strongest for the architecture at proxy
scale and for a dense optimizer starting point. It cannot establish final
multilingual capability, long-horizon stability, full-scale QAT quality,
32-GPU throughput, or a compact inference footprint. The completed paper
should state which conclusions are supported by the production checkpoint
and which remain architecture or systems hypotheses. **[Write conclusion
after the final tables are populated.]**

## References to complete and verify before submission

1. Peri-LN, arXiv:2502.02732.
2. ResFormer, arXiv:2410.17897.
3. Scalable Muon, arXiv:2502.16982; cite the exact optimizer rule used here.
4. FlashAttention-3, arXiv:2407.08608.
5. DeepGEMM, pinned implementation and license in the training environment.
6. Ternary-weight/QAT comparator papers and benchmark sources: add after
   choosing the final comparison set and verifying each cited result.
