# Learning-rate references for the selected production recipe

Primary sources inspected 2026-10-28. This is an engineering comparison of
published recipes, not a measured CI-1 stability result or a new LR selection.
Downloaded papers/pages remain outside Git.

The selected dense peak rates are spectral Muon 0.017, tied embedding/head
AdamW 0.003, ordinary norms/value coefficients 0.0003, and Q/K gains 0.0001.
The supplied QAT candidate halves Muon and norm rates and retains embedding
AdamW 0.003. Optimizer groups and implementation are defined by
[recipe.py](../../recipe.py), [optim.py](../../../terminus/optim.py), and the
[spectral update equation](../../../docs/MUON_SPECTRAL_IMPLEMENTATION.md).

## What MiniCPM actually reports

[MiniCPM, §3.3 and §6.2](https://arxiv.org/html/2404.06395v2) reports an
optimal **base LR around 0.01**, searched at smaller scales and checked at
approximately 2.1B parameters. Its stable pretraining stage uses a maximum
base LR of 0.01 with a roughly 3.93M-token batch. This is direct precedent
for tuning smaller models and transferring rates to larger ones.

Appendix A.1, Table 7 specifies that **each two-dimensional tensor's LR is
multiplied by `d_base / d_model`**. The
[released MiniCPM-2B configuration](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/blob/main/config.json)
has `d_base=256` and `d_model=2304`, giving:

```text
two-dimensional tensor LR = 0.01 × 256 / 2304 = 0.00111111
CI-1 embedding LR / that coefficient = 0.003 / 0.00111111 = 2.7
```

This is a comparison of coefficients. MiniCPM also multiplies embedding
outputs by 12, scales logits by `256/2304`, and uses different initialization
and residual scaling. CI-1 uses Peri-LN, learned Q/K gains/value residual and
spectral Muon. Copying its base LR or its width factor alone would change the
selected model/optimizer semantics.

[MiniCPM4, §2.3.1](https://arxiv.org/html/2506.07900v2) continues the small-model
search and muP approach, compares it with StepLaw, and evaluates transfer
using both language-model loss and ScalingBench. The paper supports the
transfer method but does not disclose a numeric final 8B pretraining peak LR
that can be used as a direct CI-1 comparator. Its fastText classifier LR and
post-training/QAT references must not be substituted for that missing value.

## Muon references with disclosed coefficients

[Moonlight, §2.2 and §3.2 Table 2](https://arxiv.org/html/2502.16982v1) uses
Muon for hidden matrices and AdamW for auxiliary parameters, including
embedding/head and RMSNorm. It scales the raw orthogonalized matrix update
by `0.2 × sqrt(max(rows, columns))`, enabling a shared base LR.

| Moonlight scaling model, excluding embeddings | Width | Shared peak LR | Global tokens/update at 8K context |
| --- | ---: | ---: | ---: |
| 1.1B | 2304 | 0.0008561 | 1,572,864 |
| 1.5B | 2560 | 0.0008305 | 2,097,152 |

The first row matches CI-1's width; the second matches the selected global
batch. They differ in depth, architecture, precision, data and schedule.
CI-1's embedding/head LR is approximately **3.5–3.6 times** these shared
auxiliary rates. Moonlight's final 3B-active/16B-total MoE uses a lower peak
of 0.00042 over a much longer 5.7T-token program (§3.3); it is less directly
matched to CI-1 than the scaling models.

For a square Q/O matrix of width 2304, equating the coefficients on the
same raw orthogonalized update gives:

```text
CI-1:     coefficient = 0.017
Moonlight convention: coefficient = shared_LR × 0.2 × sqrt(2304)
equivalent shared_LR = 0.017 / 9.6 = 0.00177083
ratio to Moonlight's width-2304 coefficient = 0.00177083 / 0.0008561 ≈ 2.07
```

This only equates the square-matrix update coefficient. It does not equate
optimizer state, decay, gradients, precision or complete trajectories.
Rectangular logical K/V, gate/up and down matrices have their own shape
factors; the same numerical conversion must not be assumed for all groups.

[Kimi K2, §2.1 and §2.5](https://arxiv.org/html/2507.20534v1) uses RMS-matched
MuonClip, peak LR 0.0002, 500 warmup steps and a 67M-token batch for its
15.5T-token main pretraining. Its MoE scale, MLA and QK-Clip make that a
stability reference rather than a suitable numeric LR target for CI-1.
The published 0.0002 cannot be compared directly with spectral Muon 0.017.

## What can be concluded from Gemma

The inspected [Gemma 1](https://arxiv.org/html/2403.08295v4),
[Gemma 2](https://arxiv.org/html/2408.00118v3),
[Gemma 3](https://arxiv.org/html/2503.19786v1), and
[Gemma 4](https://arxiv.org/html/2607.02770v1) technical reports do not disclose
a usable numeric pretraining peak LR or embedding-group LR. Gemma 4's
official [model card](https://ai.google.dev/gemma/docs/core/model_card_4)
links to that report; no missing LR is supplied by the card.

Gemma is valuable evidence for architecture and training methods. Gemma 3
uses knowledge distillation and Gemma 4 describes a similar pretraining
recipe. Public fine-tuning examples, optimizer-library defaults and community
reproductions do not establish Google's original pretraining LR. No claim
that CI-1's LR is above or below Gemma's is supported by these sources.

## Implication for training readiness

The earlier SmolLM-only comparison was too narrow for a general assessment.
MiniCPM and Moonlight provide stronger evidence that successful recipes can
use coefficients around 0.001 under their respective parameterizations.
The selected CI-1 rates remain **aggressive transfer candidates**, especially
embedding/head AdamW 0.003; the disclosed references do not establish that
they are too high for this architecture.

Keep the existing HPO/width-bridge result as the first candidate. The archived
700M bridge favored embedding LR 0.003 over 0.00424 and 0.006; that comparison
does not show that lower rates would lose over the production horizon.
Inspect actual embedding/head update-to-weight ratios, norm/QK gains,
clipping and per-domain held-out learning through the selected warmup and
plateau milestone. Add a focused comparison only if evidence warrants one,
under [BATCH_STABILITY.md](../BATCH_STABILITY.md), rather than restarting full
HPO or copying another model's numeric LR.

No peak rate, batch, warmup, architecture or optimizer behavior is changed by
this reference review. Native MI325X fit, numerics and learning remain unrun.
