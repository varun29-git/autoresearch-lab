# Batch and phase stability plan for eight MI325X GPUs

Reviewed 2026-10-26. This is a test plan for the selected 24-layer, width-2304
model and spectral Muon. No MI325X full update or QAT run has been measured.
The human controls paid GPU launches. The executable dense candidates are in
[profiles.py](profiles.py); AMD ternary training remains unavailable until its
native implementation and acceptance gates pass.

## What the smaller-model HPO is meant to transfer

The dense HPO is **not discarded or scheduled to be repeated at 1.446B**.
It tunes the base rates on tractable models, then carries them through the
selected width-transfer rules. In particular, spectral Muon updates each
logical matrix with `η × sqrt(d_out / d_in)` times its NS5 direction. The
Q/O, K/V, gate/up and down aspect ratios are the same at widths 768, 1536
and 2304, so this factor has the same value across the width family; see
[the derivation](../../docs/MUON_SPECTRAL_IMPLEMENTATION.md). The tied
embedding has its own AdamW rate, supported by the 700M width bridge.

This is the intended way to use the HPO rates on the larger model. The
393,216-token HPO **global batch** is a separate training choice: no width
formula makes 2,097,152 tokens/update equivalent to it. Keep that batch via
`batch_bridge_mb12` if its full-model MI325X speed is acceptable, or qualify
a larger global batch by a bounded learning comparison. Monitoring the
production model checks the transfer; it is not a plan to retune every rate
at full scale.

## Keep three batch quantities separate

`microbatch` is sequences processed simultaneously **on each GPU**. One
microstep processes `8 × microbatch × 4096` global tokens. `accumulation` is
the number of microsteps before one optimizer update. Their product is the
**global batch**, in tokens per optimizer update. Raising microbatch at a
fixed global batch changes activation memory, GEMM shapes and launch count;
raising the global batch changes the number of optimizer updates per token.
The memory-optimal, speed-optimal and learning-optimal choices can differ.

| Dense profile | Sequences/GPU/microstep | Microsteps/update | Global tokens/update | Updates per 1B tokens | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `batch_bridge` | 4 | 3 | 393,216 | 2,543 | HPO batch transfer control; slow execution possible |
| `batch_bridge_mb12` | 12 | 1 | 393,216 | 2,543 | Same HPO global batch in one larger eight-GPU microstep; fit/speed unmeasured |
| `batch_1m_mb16` | 16 | 2 | 1,048,576 | 954 | Intermediate learning/speed candidate |
| `balanced` | 16 | 4 | 2,097,152 | 477 | Initial eight-GPU baseline |
| `batch_3m_mb16` | 16 | 6 | 3,145,728 | 318 | Fixed-global-batch control for microbatch 24 |
| `batch_3m_mb24` | 24 | 4 | 3,145,728 | 318 | Conditional larger-microbatch speed trial |
| `large_batch` | 16 | 8 | 4,194,304 | 238 | High-batch quality stress test, conditional on earlier results |

At 2,097,152 tokens/update, `mb8`, `balanced` and `mb32` are another
fixed-global-batch execution comparison with accumulation 8, 4 and 2. A
microbatch-24 trial cannot use that 2M batch with constant-size microsteps.
`batch_bridge`/`batch_bridge_mb12` pair keeps the HPO global batch and
optimizer-update frequency unchanged while using three or one microstep.
The latter is a concrete way to exploit more HBM **without** discarding the
HPO batch choice. It still needs MI325X fit/speed and full-width quality
checks; the HPO itself used a smaller model, an English-first data scope and
a shorter warmup. The
3,145,728-token pair tests microbatch 24 without changing the global batch *within
the pair*. Moving from 2M to 3.146M still needs a learning comparison.

The HPO rates were chosen on a **393,216-token** update. The dense production
rates in [recipe.py](../recipe.py) are Muon 0.017, tied embedding AdamW 0.003,
ordinary norms 0.0003 and Q/K norms 0.0001, with matrix decay 0.02 and
embedding decay zero. The width-transfer rule motivates using these rates
without a full-scale HPO rerun. It does not make them batch-invariant, and
no learning-rate multiplication by batch ratio is authorized by this plan.

Holding rates fixed does **not** hold training dynamics fixed. Muon momentum
0.95 has a roughly 13.5-update half-life: about 5.3M tokens at the HPO batch,
28.3M at 2M and 42.5M at 3.146M. At a constant peak Muon LR, projection
decay alone multiplies a weight by `1 - 0.017 × 0.02` each update. Its
illustrative decay-only retention over 1B tokens is about 0.42, 0.85 and
0.90 at those three batches respectively; the actual WSD schedule changes
these values. Fewer optimizer updates, different momentum memory measured in
tokens, clipping incidence, and FP8 block quantization can all change the
trajectory. This arithmetic explains why throughput alone cannot select the
global batch.

## Evidence and its limits

| Source | Useful observation | Why it does not select CI-1's batch |
| --- | --- | --- |
| [Archived HPO program](../../hpo/program.md) | 393,216 tokens/update at the 200M screen and width bridges | Different width, English-first data and much shorter horizon |
| [SmolLM2 1.7B config](SOURCES.md) [S20] | 2,097,152 tokens/update | AdamW, 2048 context, different architecture and data |
| [MiniCPM](https://arxiv.org/html/2404.06395v2), §3.2, Table 2 | Fixed-GPU experiments emphasize loss per tokens; its 1.2B run grows batch from 2M to 4M | Different model, optimizer and empirically chosen schedule; no Muon rule transfers |
| [Muon scaling study](https://arxiv.org/html/2502.16982v1), Table 2 | A 1.5B study lists 256 examples at 8192 context, about 2.1M tokens/update | Different Muon implementation, LR tuning and compute/data regime |
| [Gradient noise scale](https://arxiv.org/html/1812.06162v1), §§2–3 | The useful batch can change during training and is related to a speed/data-efficiency tradeoff | Approximate predictor, not a measured CI-1 gradient-noise scale or a proof for nonlinear Muon |
| [Direct Muon batch-transfer sweep](SOURCES.md) [S30], §§3.2 and E.3 | In its language-model sweep, fixed learning rates and Muon momentum with matrix decay scaled by batch gave the best common rule among 216 candidates; the best rule changed with target batch and task | 12-layer, width-768, BF16, 1024-context, 1.7B-token experiment with different AdamW groups, NS iterations and schedule; it cannot select CI-1's rates or erase its sample-efficiency gap |
| [Non-Euclidean noise scale](SOURCES.md) [S31] and [Muon critical-batch analysis](SOURCES.md) [S33] | Muon-specific batch diagnostics are possible; useful batch depends on measured noise and effective rank | The published adaptive method has not solved stateful Muon momentum plus mixed AdamW; neither source gives CI-1 an absolute useful-batch threshold |
| [Adam/RMSprop batch SDE](SOURCES.md) [S32], §5 | Derives a coordinated square-root LR and moment rule under its assumptions | Not a batch-invariance result for spectral Muon and CI-1's mixed optimizer |
| [ParetoQ](https://arxiv.org/html/2502.02631v1), §2.1; [Continual QAT](https://arxiv.org/html/2502.11895v1), §§4–5; [MiniCPM4/BitCPM4](https://arxiv.org/html/2506.07900v2), §3.3 | Successful ternary transfers use enough QAT updates and may benefit from carried optimizer state and phase-specific LR schedules | Their tested best dense/QAT splits disagree; none validates CI-1's 60/40 split or QAT rates |
| [Local Lean 150M review](../../experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md) | Conversion caused a large initial loss spike; 2B QAT recovered much of it but remained worse than the dense parent on every tracked source | One smaller model with different optimizer/data; no matched dense continuation |

The older H100 25-layer memory observations (55.5 GB at microbatch 4 and
87.4 GB at 8) extrapolate to about 151/215/279 GB at microbatch 16/24/32.
These are **screening numbers**, not MI325X predictions. The selected model,
attention window, FP8 backend and allocator differ. A microbatch 24 or 32
must earn at least 15% free *usable* HBM through a warm optimizer update,
validation and checkpoint; filling nominal 256 GB is not the goal.

## Can the HPO rates reach the intended 2M batch without full retuning?

**Yes, as a testable transfer, not as an exact trajectory match.** Use the
width-transferred rates in `recipe.py` unchanged for the first 2M candidate.
This preserves the HPO result as the optimization anchor and keeps the
selected optimizer semantics. The new MuonW–AdamW batch study [S30] makes
this more plausible than a blind linear or square-root LR increase: in its
language-model sweep, the best common rule kept both learning rates and Muon
momentum fixed. Its result does **not** establish that our 2M candidate meets
our quality guardrail; its best large-batch loss still rose relative to its
small-batch reference at the same 1.7B-token budget.

The smallest defensible qualification is a **batch comparison**, not another
full hyperparameter search. Use the same full-size initialization, ordered
data stream, 1,006,632,960-token warmup, long-horizon WSD schedule and held-out
sets for `batch_bridge_mb12` (393,216) and `balanced` (2,097,152). If the
393k one-microstep profile does not fit, use `batch_bridge` at the same global
batch. Screen at 4,026,531,840 tokens, then extend a credible pair to
8,053,063,680 tokens. The pre-decay bridge detects early learning and
stability differences; it does not prove equal final quality over hundreds
of billions of tokens. Compare matched tokens **and** matched allocated
GPU-hours, and enforce the per-scope guardrail in `CONFIG_SELECTION.md`.

If unchanged 2M fails that comparison, test **one focused correction** before
abandoning the batch: a *separate, explicitly reviewed* 2M matrix-decay
candidate `0.02 × (2,097,152 / 393,216) = 0.1066667`, with the same peak
learning rates, momentum, Adam betas, embedding decay and schedule. This
counteracts the nominal loss of decay updates per token and is motivated by
the language-model result in [S30], but its 1.7B-token setting and 80% LR
cooldown differ from our long WSD plateau. It is **not** the selected recipe,
an automatic config rewrite or a guaranteed improvement. It changes an
optimizer value, so compare it against the selected 0.02 baseline before
promotion. If it does not close the loss gap, compare `batch_1m_mb16` at the
unchanged rates; retain the 393k batch if that yields the best accepted
learning/throughput tradeoff. A later small LR multiplier trial is justified
only by measured failure, with separate Muon and AdamW diagnostics.

Do not use `5⅓ ×` Muon LR, `sqrt(5⅓) ×` every LR, or token-memory transport
of Muon momentum/Adam betas as an automatic fix. The Adam square-root rule
[S32] does not cover Muon. In [S30], transporting Muon momentum or Adam's
first moment hurt its language-model transfer, while its small benefit from
transporting the second moment is not enough to justify changing CI-1's
already-short `β₂=0.95` memory without a controlled trial. A Muon-aware
noise-scale measurement [S31] could help diagnose a bad batch, but it does
not replace the full-model comparison for this stateful mixed optimizer.

## Interpret the supplied QAT candidate before scaling it

The supplied rates are Muon **0.0085**, tied embedding AdamW **0.003**,
ordinary norms **0.00015**, Q/K norms **0.00005**, momentum 0.95, Adam betas
(0.9, 0.95), matrix decay 0.02 and embedding decay zero. They preserve the
spectral Muon algorithm and halve the dense Muon/norm rates while keeping the
embedding rate. This is the user-specified **QAT candidate**, not an HPO
result present in `hpo/` or a native AMD-qualified winner. The repository's
QAT template intentionally leaves production rates unfilled.

The supplied `phase_start_tokens=500,170,752` is exactly 1,272 updates of
the old 393,216-token HPO batch. It is **238.5** updates at the 2M production
batch, so no full dense checkpoint can switch at that exact token under that
batch. It happens to be 159 updates at 3.146M, but 500M dense tokens is still
an HPO-scale endpoint, not a month-long dense/QAT allocation. The converter
requires `phase_start_tokens` to equal the *actual* parent checkpoint's
`tokens_seen`; leave it unset until the real phase boundary is selected.

The supplied 50,017,075-token QAT warmup lasts only about **24 updates at
2M** or **16 at 3.146M**. The 100,034,150-token decay lasts about **48 or 32**
updates. The local Lean QAT run clipped 116 updates during recovery after
conversion, all in its first 122 updates at a much smaller batch. Those
numbers do not establish a threshold for CI-1, but they make a 16–24-update
re-warmup a poor untested long-run default. Keep the supplied *peak rates* as
the candidate; derive long-run phase lengths from the actual QAT horizon.

For the initial 60/40 planning policy, use a complete-update boundary from
[phase_plan.py](phase_plan.py), preferably with the same global batch in both
phases. Start with a separate QAT WSD cycle: re-warm over
`min(1,006,632,960 tokens, 5% of Tq)`, rounded to full global updates;
cosine decay over the final 20% of `Tq` to 0.1× peak. At a 400B-token QAT
phase and 2M batch, that is roughly 480 re-warmup updates and 38,147 decay
updates. These are **starting schedule choices** from
[QAT_ALLOCATION.md](QAT_ALLOCATION.md), not proven optima. Compare the 80/20
alternative under equal total allocated GPU-hours if recovery evidence
justifies it. Preserve a completed dense checkpoint before conversion.

The transition can now rebase the **microstep cursor** when the child QAT
microbatch changes but global tokens per update remain fixed. It preserves
tokens seen, update count, optimizer moments and rank RNG and records the old
and new cursor. A global-batch change at the switch still fails closed:
otherwise optimizer and data-cursor semantics would change. Test the actual
eight-rank branch and exact resume before relying on this CPU-tested path.

## Bounded decision sequence

1. **CPU and native arithmetic gate.** Validate source, tokenizer, ordered
   token stream and exact update accounting. Pass native MI325X FP8/attention
   preflight and one full 1.446B update before timing. Qualify the QAT backend
   separately; its current AMD implementation is incomplete.
2. **Fixed-global-batch execution.** At 2M, compare `mb8`, `balanced` and
   `mb32` only as memory permits. At 3.146M, compare `batch_3m_mb16` and
   `batch_3m_mb24`. Alternate warm full-update timing runs; measure slowest
   rank, data wait, all-reduce, optimizer, allocated/reserved/device HBM,
   checkpoint and validation peaks. A larger microbatch needs at least 3%
   credible complete-update throughput gain, no repeated p95 regression and
   the same numerical/quality gates. Full-size headroom remains at least 15%.
3. **Global-batch learning.** First compare the two 393k execution profiles,
   then compare 393k and 2M at matched tokens; add 1M as an intermediate
   candidate if the reserve permits or the 2M quality result is poor. Use
   the same seed, data prefix, model, fixed dense peak rates, token-based WSD
   milestones and 1,024 held-out sequences per scope key. Screen at the
   existing 4,026,531,840-token plateau bridge; extend credible finalists to
   a common later milestone before selecting a long-run batch. Test 3.146M
   only if its microbatch execution or update overhead makes it attractive;
   4M is a conditional stress comparison. Record **both** loss at matched
   tokens and loss at matched allocated GPU-hours. The 0.02-nat per-scope
   upper paired-bootstrap guardrail in [CONFIG_SELECTION.md](CONFIG_SELECTION.md)
   is an initial acceptance rule, not a theorem about final quality. If the
   interval is wide, extend the run or repeat a seed; do not weaken the rule
   after seeing a result.
4. **Stability at every bridge.** Record nonfinite events, preclip gradient
   norm and clipped-update fraction, Muon logical-matrix update/weight RMS,
   embedding and norm movement, Q/K gains, learned value residual, residual
   and attention-logit RMS, FP8 scales/saturation, and held-out language,
   code, math and Indic losses. Watch warmup and the later plateau separately.
   Do not classify a single expected ternary conversion spike as divergence;
   require finite gradients and a measured recovery curve.
5. **QAT transfer and final objective.** From one verified dense checkpoint,
   run QAT and same-data dense-continuation controls with identical frozen
   held-outs. Check immediate conversion loss, re-warmup, plateau, decay and
   downstream tasks. Compare final **ternary** quality at equal token exposure
   and equal allocated GPU-hours. Select the batch and phase split on that
   evidence, not on HBM occupancy or dense tokens/s alone.

Freeze the chosen global batch, phase token budgets, schedules, data mixture,
source and optimizer settings in external run artifacts before a long launch.
The existing exact-resume contract deliberately rejects changing the batch
or horizon mid-run. A later adaptive-batch policy would require a separately
reviewed transition with the same stream and numerical gates; it is not
enabled by these profiles.
