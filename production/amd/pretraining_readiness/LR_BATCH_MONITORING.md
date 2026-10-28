# Lower-LR controls and monitoring for the H100 qualification run

Engineering proposal, 2026-10-28. The user explicitly requests lower learning
rates informed by strong published Muon runs, tested at the intended global
batch, plus a detailed monitoring contract. This refines the
[focused Azure plan](AZURE_H100_BRACKETS.md). It replaces that plan's immediate
two approximately 2B-token trials with the bounded matrix below. The available
allocation remains **up to 50 hours on one H100**; no new paid run has started.

All trials use the actual **1,446,478,126-parameter production model**, 4096
context, dense FP8 projections, BF16 residual/head and FP32 master weights,
gradients and optimizer state. Preserve spectral Muon, NS5/Nesterov and the
production architecture. Native fit, numerics and the qualification launcher
are prerequisites. These are reviewed experiment specifications, not changes
to production defaults or generated executable run configurations.

## 1. Why challenge the current rates, and how to compare published values

The completed 200M/700M work supports the existing optimizer and rates in its
tested regime. It does not require that a lower rate lose at production width,
2M global batch or a complete short WSD horizon. Test this uncertainty directly;
do not either discard the old evidence or exempt its winner from comparison.

The closest disclosed numerical reference is
[Moonlight, section 2.2 and Table 2](https://arxiv.org/html/2502.16982v1):
its width-2304 scaling model uses shared peak LR **0.0008561**, 1,572,864
tokens/update and 28.54B tokens. The width-2560/1.5B non-embedding model uses
**0.0008305**, 2,097,152 tokens/update and 38.91B tokens. Context is 8192;
architecture, initialization, data and precision differ from CI-1.

CI-1 applies `eta * sqrt(out/in)` to each logical NS5 matrix direction.
Moonlight applies `shared_lr * 0.2 * sqrt(max(out,in))`. Applying the latter
rule to **CI-1's shapes**, purely to compare coefficients, gives:

| CI-1 logical matrix | Spectral factor | Spectral eta matching Moonlight's 0.0008561 coefficient | Current 0.017 coefficient / that coefficient |
| --- | ---: | ---: | ---: |
| Q and O, 2304 x 2304 | 1 | 0.00821856 | 2.07 |
| K and V, 768 x 2304 | 0.57735 | 0.01423496 | 1.19 |
| Gate/up, 5760 x 2304 | 1.58114 | 0.00821856 | 2.07 |
| Down, 2304 x 5760 | 0.63246 | 0.02054640 | 0.83 |

This table does not claim Moonlight uses our GQA shapes. It equates the scalar
on the same hypothetical raw matrix direction, not actual gradients, moments,
decay or entire training trajectories. **No single spectral eta reproduces
Moonlight's scaling on every rectangular matrix.** Fused QKV is three logical
matrices; fused gate/up is two. Never compute these factors from packed row
counts. Reducing the global Muon LR may help Q/O while weakening useful down
updates; collect logical-role diagnostics to distinguish that case.

Our tied embedding/head AdamW 0.003 is about **3.50 times** Moonlight's shared
rate. This motivates testing 0.00085; it does not establish that 0.003 is wrong.
Tied versus separate tables, initialization, output scaling and Adam groups
matter. Norm/value and Q/K gains have their own rates and must not inherit
the embedding rate by accident.

[Kimi K2, section 2.5](https://arxiv.org/html/2507.20534v1) uses 0.0002 with
RMS-matched MuonClip, a 67M-token batch and a 15.5T-token main horizon. It is a
source for attention-logit monitoring, not the primary numeric target for a
1.446B dense model. Moonlight's Appendix D emphasizes norm-gain control; CI-1
uses zero decay on norm groups, so copying only its LR does not copy its full
stability recipe. Adding norm decay or QK-Clip would be a separate algorithm
review, not an unrecorded rescue during a candidate run.

## 2. Mandatory initial matrix: separate the two main LR effects

Use **2,097,152 global tokens/update in all four runs**. On one H100,
microbatch 4 x 4096 x accumulation 128 is a fit candidate; on eight MI325X,
microbatch 16 x 4096 x accumulation 4 has the same global batch. Measure fit
and keep the chosen H100 microbatch fixed across these LR comparisons.

| ID | Spectral Muon peak | Tied embedding/head AdamW peak | Question |
| --- | ---: | ---: | --- |
| R0 | 0.017 | 0.003 | Existing production starting recipe: the control |
| R1 | 0.017 | 0.00085 | Does the lower published-rate region help the tied table? |
| R2 | 0.0082 | 0.003 | Does a lower Muon coefficient help with the old AdamW rate? |
| R3 | 0.0082 | 0.00085 | Do the lower choices work together? |

These are dense-training candidates, not a QAT switch. 0.0082 is rounded from
the square-matrix conversion, not a claim of whole-model Moonlight equivalence.
All table rates are already for **production width 2304**; do not apply the
proxy-to-production width multiplier a second time when generating configs.
The crossed design matters: running only R0 and R3 could not identify whether
Muon, AdamW or their interaction explains the result. The earlier 200M tests
do not answer this particular production-width/global-batch comparison.

For all four, explicitly hold: norm/Peri/final/value LR **0.0003**, Q/K LR
**0.0001**, matrix decay **0.02**, embedding/norm/QK/value decay **zero**,
momentum **0.95**, Adam betas **(0.9, 0.95)**, epsilon **1e-8**, clip **1.0**.
No role LR overrides, graphs, compilation, FP8 head or backend changes.
Fixed auxiliary settings are controls, not newly tuned optima.

Use the same initialization weights as well as seed, representative data
prefix/mixture, tokenizer, masking, loss normalization and evaluator. Verify
their identities. Warmup/decay are in **global valid training tokens**, not
microsteps or per-device tokens. Log all optimizer-group LRs explicitly.
Do not apply square-root or linear LR scaling merely because there are eight
GPUs or a larger batch. A GPU-count change with fixed global batch is a
distributed implementation change, not another batch-size change.

### LR/decay coupling is part of the interpretation

Muon updates have both a direction term and shrinkage `(1 - eta_t * wd)`.
R2 versus R0 changes both because eta changes while wd is fixed. It isolates
the configured Muon LR coordinate, **not pure step length independent of
regularization**. Log `eta_t * wd`, its cumulative log-retention and actual
parameter deltas. A follow-up with lower eta 0.0082 and wd
`0.02 * 0.017 / 0.0082 = 0.0414634146` preserves the control's shrinkage factor
at each update of the same schedule. Compare this separately if needed.
Embedding decay is zero, so that particular coupling is absent in R1.

## 3. Schedule and follow-ups within the cap

Each initial trial uses **1,006,632,960 tokens**, a complete short WSD:
100,663,296 warmup, 704,643,072 stable, 201,326,592 cosine decay to 10% of peak.
At 2M this is 480 updates: **48 warmup, 336 stable, 96 decay**. Every candidate
gets the same full schedule; stop early for a hard failure, not to declare a
healthy candidate worse before its cooldown. Capture pre-decay and final
validation separately. Short-annealed performance is a proxy criterion, not
proof of long-plateau ranking or a production continuation checkpoint.

**Fifth trial, if it fits:** run the leading credible complete recipe at
**1,048,576 global tokens/update**, preserving LRs, data and token schedule.
It has 960 updates (96/672/192). Compare against the existing 2M result.
This tests sample efficiency at these rates; it does not establish independently
tuned optimal rates for both batches. Final AMD throughput still determines
whether any 2M quality/time tradeoff is worthwhile.

**Sixth slot, if it fits:** choose exactly one question from the observed
uncertainty, recording the choice before starting that run:

- If lower embedding LR helps, test **0.002** with the selected Muon LR as
  the strict historical width-transfer alternative/intermediate. The other
  settings match the selected parent recipe, but start the trial fresh.
- If lower Muon helps and shrinkage explains a plausible part of the result,
  use the **0.0414634146 matched-shrinkage** control above at 2M.
- If 1M materially beats 2M, a direct decay-transport control from the **new
  1M comparison** is `0.02 * 2 = 0.04` at 2M with the same selected LRs.
  The earlier **0.1066667** proposal instead references the **old 393,216**
  batch; it answers a different transfer question. Neither is a universal
  decay value, and any changed eta must be included when comparing shrinkage.
- If norm/QK/value drift dominates, halve only those rates to **0.00015 /
  0.00005** on the leading complete recipe. Do not attribute that intervention
  to lower Muon or embedding LR.
- If candidates are close, spend the slot on another seed for the paired
  comparison only when both necessary runs fit; one extra seed for just the
  apparent winner measures its variability but does not confirm pairwise
  seed robustness. Otherwise report a tie rather than force a winner.

These are alternatives, not a hidden additional sweep. Adding lower rates
changes the previous budget: four 2B trials would already take **55.92 hours
at 40k tokens/s**, before setup. The initial horizons are therefore 1B. Longer
confirmation remains later work or must replace some planned trials; it is
not obtained by calling a cooled-down 1B run a prefix of a 2B/1T schedule.

Reserve **eight H100 hours** for setup, native checks, special reference/replay
work and contingency, leaving at most 42 for ordinary trial windows. Ordinary
trial rates must include routine evaluation, logging and checkpoint overhead.

| Trials of approximately 1B each | Aggregate tokens | At 30k tokens/s | At 40k | At 50k |
| --- | ---: | ---: | ---: | ---: |
| Four LR controls | 4.0265B | 37.28 h | 27.96 h | 22.37 h |
| Plus the 1M batch trial | 5.0332B | 46.60 h | 34.95 h | 27.96 h |
| Plus one more 1B diagnostic | 6.0398B | 55.92 h | 41.94 h | 33.55 h |

These are scenarios, not current Azure measurements. With the eight-hour
reserve, approximately **33.3k** is needed for five trials and **40k** for six.
At 30k, do the four LR controls within the cap and keep the batch comparison
explicitly outstanding. Sum actual per-trial rates if the 1M batch is slower.
Native failures or slower preparation can reduce the available trial count.
The human controls paid campaigns; a longer run requires a new budget decision.

## 4. What to record and why

Set `log_interval=1` during qualification. The existing logger's loss and clip
coefficient describe the **last complete update**, not a mean over every
update since the previous log. Sparse logs cannot establish a clipping rate.
Measure logging overhead rather than silently reducing coverage.

| Signal | Required coverage/cadence | Interpretation |
| --- | --- | --- |
| Token-mean train loss, valid-target count, all group LRs, phase, tokens, updates, elapsed allocation | Every complete update | Detect normalization, counting, schedule and learning failures; compare loss versus tokens and wall time |
| Frozen held-out NLL by source/language/domain | Quick fixed slice about every 100M tokens; full fixed sets at pre-decay and final endpoints | Detect poor generalization or a damaged critical domain despite an improving aggregate |
| Global and per-role gradient norms before/after clipping; clip coefficient and rolling frequency | Global every update; per-role diagnostic cadence | Locate the group driving clipping; sustained clipping is an alert, not proof of an excessive LR |
| Actual direction update, decay contribution and total parameter delta; weight RMS; update/weight RMS; update-direction cosine | Per logical Q/K/V/O/gate/up/down and each Adam group, about every 20 updates and at schedule boundaries | Detect excessive movement, ineffective updates, oscillation or decay dominance; do not infer update size from LR or gradient norm alone |
| Embedding/head row statistics and logits | Frequency-stratified row cohorts plus full-table summaries at diagnostic points | Tied head gradients can update rows absent from the input; inspect rare/frequent rows without assuming unused input rows must have zero gradient |
| Norm/Peri/final gains, Q/K gains and value-residual coefficients | Every diagnostic point, all layers | Detect scale drift hidden behind global gradient norms; near-zero parameters need absolute deltas as well as ratios |
| Residual, attention-output and FFN RMS; Q/K norms, scaled attention logits, attention entropy/probability concentration | Sample rotating examples, query positions and layers; emphasize boundaries and any anomaly | Detect growing activations or attention collapse; separate local/global masks and fixed sample support |
| Actual native FP8 scales, clipping/saturation and zero/underflow fractions for weights/activations/dgrad; wgrad/reference error | Representative native operands at startup, warmup exit, pre-decay and on anomalies | Detect precision failure or cache staleness that can resemble an LR problem |
| Parameters, gradients and optimizer moments finite; required gradients present | Every update for global gradients; fused or chunked full-state checks before committing an accepted update/checkpoint | Prevent a sampled healthy tensor from hiding corrupted state; allow only documented intentionally absent gradients |
| Full-update time distribution, data wait, peak driver/allocator memory, clocks, power, temperature, hardware errors | Every update/time sample; repeat memory measurement at validation/checkpoint cadence | Distinguish slow learning from slow execution; require at least 15% usable HBM headroom |
| Checkpoint age, free disk, backup queue/age, content hashes and restore counters | Every checkpoint and periodic watchdog check | Prevent loss of progress and a repeat of the historical disk-full failures |

Cheap scalars are continuous; richer diagnostics are sampled with their
coverage stated. Lower the diagnostic overhead by fusing/reducing summaries,
not by pretending samples are full coverage. Do not add PyTorch allocated and
reserved bytes: reserved already includes allocated storage. Check driver free
memory and non-PyTorch workspaces as well as allocator peaks.

### Important details about interpreting these signals

- **Clipping is not a Muon update-size cap.** NS normalization makes the
  direction approximately invariant to a uniform positive rescaling of its
  input, subject to finite precision and state history. A small clipped raw
  gradient can still yield an LR-sized matrix update. Clipping remains useful,
  but neither `clip=1` nor zero clipping proves LR safety. Inspect actual deltas.
- **Attention samples are not global extrema.** The current monitor takes
  one query and at most 128 keys. Its entropy is over that subset, not the
  full attention support. Do not compare subset entropy against a full-window
  threshold or copy Kimi's logit threshold of 100 without matching units,
  query/key scaling and its different attention architecture.
- **Gradient norms across batches need context.** A larger averaged batch
  naturally changes gradient noise; compare each recipe's trend and held-out
  learning, not only an absolute norm against the 393k history.
- **Ratios need denominators.** Norms/value coefficients can start near zero.
  Always retain absolute deltas and a documented denominator floor; a huge
  ratio caused by a near-zero weight is not automatically divergence.
- **Decay must be visible.** Record direction-only and total update where the
  harness supports it, so decay does not masquerade as useful learning. The
  sum of `log(1 - eta_t * wd)` records nominal decay-only retention, not the
  actual norm trajectory under gradient updates.
- **Finite loss can hide failed learning.** All-zero/tiny cached gradients,
  stale FP8 weights, an omitted optimizer group or a bad target mask can leave
  plausible loss curves. Use complete-update reference controls and data audits.
- **FP8 boundary occupancy is not necessarily clipping.** Dynamic scales can
  place a block maximum at the format boundary by construction. Distinguish
  pre-quantization overflow/clamping from codes equal to the largest finite
  value, and distinguish legitimate zero inputs/padding from lost nonzero
  gradients. Compare against the native format's independently tested reference.
- **NS5 is approximate.** Check the pinned five-step algorithm and its actual
  deltas against a reference; do not require exact SVD-level orthogonality and
  then interpret its expected approximation error as an LR failure.

## 5. Reference checks and operational decision rules

At initialization, warmup exit and pre-decay, use isolated same-state/same-data
short update probes against BF16 and independent kernel/optimizer references.
Preserve moments, RNG, schedule and data cursor; discard probe branches from
the main token count. Compare gradients and actual deltas, not whole-weight
similarity. BF16 is a precision control rather than an exact oracle for FP8
rounding. Freeze tolerances using independent controls before candidate results.
These probes do not substitute for a longer matched-precision learning study.

**Immediate stop:** nonfinite loss/gradient/master state, invalid data or
normalization, missing required gradients, stale/incompatible identities,
checkpoint corruption, uncorrectable hardware error, exhausted storage safety
margin or allocation cap. Preserve the last valid checkpoint and failing batch
identity. Do not silently skip a bad batch, change LR or reset moments.

**Review alarms:** after warmup, severe clipping (coefficient below 0.1 on
more than 20 of the last 100 updates); a sustained role/logit/FP8 statistic
above three times its established comparable range; or held-out deterioration
beyond the declared per-domain margin at consecutive checks. These are
engineering starting alarms, not universal boundaries between good and bad
training. Early short runs may not establish a stable reference range; do not
manufacture one by widening thresholds after a failure. A slow low-LR run
with healthy numerics is a learning comparison, not a numerical failure.

For selection, use token-mean NLL on fixed representative data, report each
critical domain and the pre-decay/final endpoints, and pair evaluations by
document or an explicitly justified block unit. Choose practical superiority
and noninferiority margins before viewing trial results. Bootstrap uncertainty
on held-out documents addresses evaluation sampling, **not training-seed
uncertainty or selection bias** from repeatedly picking the best candidate.
Reserve a separate final held-out set and use another seed when the apparent
advantage is close; if the budget cannot resolve the comparison, report a tie.
Do not count a tiny one-seed NLL difference as a proven production optimum.

The four matrix points are controls, not a bracket that proves an interior
optimum. If the lowest tested rate wins, an additional lower value (for
example embedding LR 0.0005) remains a possible later experiment; it is not
silently added to the six-slot budget. Likewise a failure after the 48-update
warmup establishes a problem with that **LR/warmup combination**. It does not
prove the peak LR would fail under the longer intended production ramp.
A longer-warmup control must be explicitly priced and compared before making
that stronger attribution.

A promising result needs improving held-outs, useful post-warmup learning,
healthy role-level behavior and passed numerics/recovery checks. A good final
cooldown loss with deteriorating pre-decay behavior is a reason to investigate
long-plateau suitability. No 1B experiment proves stability over the funded
dense/QAT horizon. Longer monitoring and late-QAT qualification remain required.

## 6. Current implementation versus required additions

The existing trainer logs global loss, pre-clip gradient norm, clip coefficient,
min/max group LR and allocator memory. It rejects nonfinite global loss and
nonfinite gradient norm. `terminus/stability.py` samples activations, attention,
gradients and parameter deltas. Its coverage is **rank zero, first microbatch**,
with at most 256 sampled parameter values per tensor. Its FP8 hook recomputes
a quantization diagnostic, not the actual native cached operands. Missing
gradients are skipped by that monitor. It is not an all-rank watchdog.

Before paid trials, the qualification work must add explicit per-group LR
records, rotating/full-role summaries, native-operand probes, full-state finite
and required-gradient checks, matched-state controls, safe live backup/replay,
and budget enforcement. Keep telemetry observational: no training RNG draws,
parameter changes or cache changes outside isolated reference branches. The
current launch validators also lock the selected recipe; generate candidate
configs outside Git through a separate reviewed bounded qualification mode.
Do not remove production or archived-HPO locks globally to run this matrix.
For rolling-update direction cosine, record the compared updates and sign
convention; it is a trend diagnostic, not a requirement that momentum always
follow the current negative gradient. Audit actual valid-target counts against
the loss denominator, including document masking and partial tails, before
interpreting an LR comparison. The nominal batch is not a substitute for this
count check.

This document implements no GPU instrumentation or training run. On AMD, repeat
native FNUZ checks and eight-rank comparisons, keep the selected global batch
fixed when changing accumulation, and measure cadence-inclusive throughput.
H100 results do not establish AMD performance or QAT readiness.

Supporting sources: [selected recipe](../../recipe.py),
[optimizer grouping](../../../terminus/optim.py),
[logical Muon](../../../terminus/logical_muon.py),
[current telemetry](../../../terminus/stability.py),
[trainer](../../../terminus/train.py), [LR evidence](LR_REFERENCES.md), and
[Muon batch-transfer study, sections 3.2/E.3](https://arxiv.org/pdf/2610.08975v2).
