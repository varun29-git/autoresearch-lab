# Adaptive production horizon within 5,760 GPU-hours

Engineering policy and implementation specification, 2026-11-01. Read with
[QUALITY_GATES.md](QUALITY_GATES.md). This is the proposed production policy;
the current trainer does **not** yet implement live horizon changes. No paid
work is launched or authorized by this document. Generated forecasts,
contracts, receipts and checkpoints belong outside Git.

## 1. Fix the compute budget and learning contract; measure the token outcome

The objective is the best validated final model within the allocation, with a
recoverable dense model and a qualified ternary artifact if QAT passes. A
1T/1.5T/2T token count is a forecast, not a requirement that overrides quality.
An extra token of duplicated or poor text is not equivalent to an extra useful
training token. Count valid next-token targets and report unique-corpus
exposure separately from total exposure.

For one continuously allocated eight-GPU node:

```
5,760 GPU-hours / 8 = 720 node-hours = 30 days
charged GPU-hours = sum(allocated GPUs * allocation wall-hours)
```

Charge AMD bring-up, qualification, retries, idle allocation, conversion,
validation, checkpointing and export to that same account. The separate Azure
H100 allowance remains its own account; record actual provider billing units.
The existing H100 ledger is capped at one 50-hour campaign and is not an
eight-GPU production budget controller.

Freeze before production:

- Reviewed 1.446B architecture, tokenizer, objective, attention masks,
  initialization, optimizer definitions and precision recipe.
- Qualified peak rates, one-time width-transfer rule, global tokens/update,
  clipping, momentum/betas, weight decay and an **absolute production warmup**
  in tokens/updates. Do not derive warmup afresh from each forecast.
- Source eligibility, quality filters, per-language/domain coverage, maximum
  repetition, contamination exclusions and immutable consumed data history.
- Validation protocols, critical-domain regression margins, incident policy,
  checkpoint/recovery contract, budget ceiling and finishing obligations.

Keep adjustable within qualified bounds:

- Future WSD plateau length and therefore the final token count.
- Microbatch and accumulation at the same global batch, using only measured
  configurations and a checkpoint transition that proves cursor equivalence.
- Future source availability and routing, subject to the frozen quality policy
  and proof that all consumed examples remain unchanged.
- A **prospective** dense/QAT allocation before phase commitment, constrained
  by measured QAT recovery and a funded ending. It is not a continuously moving
  percentage imposed retrospectively on the whole run.

The five fixed 5B-token proxy comparisons remain fixed-horizon experiments.
Elastic production does not turn those trials into unequal comparisons or
discard the completed HPO/width-transfer evidence.

## 2. Use WSD to keep the plateau extendable

MiniCPM's [WSD analysis, sections 4–5](https://arxiv.org/html/2404.06395v2)
explicitly motivates continuous training without a predefined final token
count. It supports this scheduling structure; it does not establish CI-1's
optimal LR, cooldown length, data mix or QAT fraction.

```
dense: fixed warmup -> extendable stable LR -> committed dense cooldown
                                                       |
                                            verified dense fallback
                                                       |
QAT:                           qualified rewarm/recovery -> stable -> cooldown
                                                                          |
                                                             evaluate/export
```

1. Warm up once using the qualified production warmup. Keep peak LR and global
   batch fixed while estimating sustainable speed.
2. Remain on the stable portion while time and approved data permit. Better
   speed buys additional useful updates here. Slower speed reduces the
   uncommitted plateau.
3. Before leaving the plateau, reserve all remaining finishing work at
   conservative **phase-specific** rates. Commit the phase endpoint and decay
   length at a durable complete-update checkpoint.
4. Execute the committed cooldown. Preserve that dense artifact before QAT.
   Once decay starts, do not silently move its endpoint, rewarm it, or raise LR
   because a later throughput sample improves. Extra time can fund evaluation
   or a separately declared branch; it need not all become training tokens.
5. Qualify the QAT implementation and a recovery trajectory early enough to
   know its cost. Use an actual compatible dense parent and a same-data dense
   continuation control. At the production switch, fund the qualified QAT
   warmup, recovery, stable portion and cooldown together.
6. QAT can also have an extendable plateau, **after** its recovery requirement
   is funded and **before** its final cooldown is committed. Dense and QAT
   cooldowns must not compete for the same remaining hours.

The existing 10% dense cooldown, 20% QAT cooldown and 60/40 token allocation
are candidates, not measured optima. The 80/20 challenger needs quality
evidence, not just a faster token forecast. Select phase-length rules and
minimum useful durations from the bounded qualification work; then solve
their costs jointly. A fixed short HPO cooldown is not a trillion-token
finishing schedule. Equally, automatically using 10% of an unknown trillion
token horizon for warmup would change the selected learning experiment.

WSD does not make horizon length irrelevant: optimizer moments, cumulative
decay and validation behaviour still require monitoring over the long
plateau. A short cooled proxy winner alone cannot prove that trajectory.

## 3. Forecast from useful work, with explicit units

Use cluster tokens/second with node-seconds, or per-GPU tokens/second with
GPU-seconds. Never combine cluster rate with aggregate GPU-hours.

Keep two measurements:

- **Update throughput:** valid global targets / complete-update time,
  including backward, clipping, optimizer, reductions and data waits.
- **Sustained phase throughput:** valid targets / elapsed phase wall time,
  including normal validation, checkpoint and monitoring cadence. Measure
  synchronized completion across all ranks. Do not extrapolate from the
  fastest rank, an isolated GEMM, an early cold update or a forward pass.

Use sustained rates for the budget. Do not also subtract the same cadence as
a second overhead factor. Separately reserve genuinely unmeasured future
work, final export/evaluation and an outage buffer. Track samples by runtime,
batch, precision, phase, data path and monitoring cadence; invalidate a rate
when those conditions materially change.

Initial operating policy to qualify:

- Obtain full-state steady updates and a representative cadence window before
  setting the first funded horizon. Include optimizer-state allocation,
  validation/checkpoint peaks and real provider-to-S3 traffic.
- Once available, retain recent short and longer windows (for example 30
  minutes and 6 hours), their distributions and slowest-rank behaviour. Until
  enough history exists, mark the estimate provisional and retain a larger
  buffer; do not invent confidence intervals from a few correlated updates.
- Reforecast at checkpoints roughly every 6 node-hours and after a sustained
  rate loss, changed data/backend or an outage. A single slow checkpoint
  should not change the learning schedule. Closer to the phase commitment
  boundary, shorten the review interval so it cannot consume the reserve.
- Freeze the conservative rate-selection rule before using it to spend the
  buffer. Log central and downside forecasts; make spending decisions using
  the downside case. Historical variability does not bound future outages.

### Remaining-work calculation

For a general checkpoint, price only **unfinished** work:

```
remaining node-seconds = remaining GPU-hours * 3600 / 8
required seconds = sum(remaining tokens in phase i / measured cluster rate i)
                   + uncounted fixed work + contingency
```

Known dense/QAT tokens already completed are not charged again, and tokens
lost on recovery are not credited twice. Separate lifetime attempted compute
from committed model progress. An unfinished mandatory phase cannot be made
to fit by silently dropping its quality minimum.

Before QAT begins, an illustrative ratio-constrained calculation is:

```
D = committed dense tokens so far
s = remaining node-seconds AFTER distinct nontraining reserves
Rd, Rq = conservative sustained cluster rates
q = qualified/candidate QAT fraction of final tokens

T_final = (s + D/Rd) / ((1-q)/Rd + q/Rq)
remaining_dense = (1-q)*T_final - D
remaining_QAT = q*T_final
```

This equation assumes no QAT tokens have been consumed, no cooldown has been
committed, no data ceiling is binding and the minimum remaining phase lengths
fit. If `remaining_dense` is negative, the proposed fraction is already
infeasible; do not clamp it to zero and call the forecast valid. Round phase
boundaries to complete global updates, then recompute all costs and minima.
If QAT has begun, use the remaining-work sum with its fixed dense history.

Apply a separate data limit: the next approved source exposures must actually
exist within the repetition caps. Compute capacity cannot manufacture useful
corpus. A faster GPU may therefore create spare time rather than permission
to feed unreviewed low-quality data.

For sensitivity only, a dense-only run at 60k/80k/100k tokens/s/GPU with an
**all-in** 90% productive fraction would process about 1.120T/1.493T/1.866T.
Those are not measurements, do not include an additional qualification
reserve, and do not price a slower QAT phase. They illustrate why the final
token target should respond to measurements rather than to an H100 anchor.

[`phase_plan.py`](../phase_plan.py) already performs initial two-phase
allocation arithmetic. It does not implement the remaining-work calculation,
live schedule changes, data extension or a production budget controller.

## 4. Protect quality when the horizon changes

### Coordinate the curriculum across dense and QAT

Dense and QAT are two training/weight-representation phases. They need
coordinated learning-rate and data schedules, but they do not require exactly
two data mixtures. Define curriculum segments by the learning milestone and
funded remaining work, rather than a fixed percentage of an assumed 1T/2T
endpoint. The following is the structure to qualify, not a newly selected
set of mixture weights:

| Segment | Data policy | Adaptation rule |
| --- | --- | --- |
| Dense warmup and stable training | Broad approved mixture covering every target language, math and code from the start | Keep the absolute warmup fixed; extend or shorten only future stable training within approved exposure/repetition limits |
| Dense finishing/cooldown | Retain the broad mix as the control; a cleaner finishing mix requires a quality comparison and domain coverage floors | Fund and commit its token budget, mixture and decay together before entering it |
| QAT conversion and rewarm/recovery | Initially continue the distribution used immediately before conversion, with the next unconsumed examples | Protect the qualified recovery budget; isolate quantization from a simultaneous data-distribution change |
| QAT stable training | Continue learning under ternary weights with a qualified mixture retaining all target capabilities | Allow additional useful tokens here when time and approved data permit; do not treat QAT as only repairing the conversion loss |
| QAT final cooldown | Qualified finishing mixture with explicit language/domain coverage | Protect and freeze this ending before it begins; preserve final evaluation/export time |

Initially compare conversion using the same mixture and data continuation as
the matched dense control. Any subsequent QAT mixture change is a separate
candidate. A high-quality-only final mixture is not automatically better:
excessive narrowing can forget capabilities. All training segments use
training data; reserved finishing pools are distinct from held-out evaluation
sets and must obey the same contamination exclusions.

Prepare a minimum funded curriculum plus optional extensions. The minimum
must include useful exposure for every target capability and both qualified
endings. Keep approved finishing-data capacity available before spending the
remaining inventory on an extension; select actual amounts using measured
rates and qualified phase minima. Additional data is eligible only within
source-specific repetition caps. Do not put all advanced math, code or Indic
coverage in a final optional tranche.

At every phase decision, forecast dense and QAT jointly. Additional dense
tokens can increase the QAT obligation under a chosen token-ratio policy, so
they are not free merely because dense runs faster. Before switching, freeze
the completed dense history and fund the QAT continuation. Once in QAT, any
further extension belongs to the qualified QAT plateau.

Track both global progress and phase-local progress, plus cumulative exposure
and absolute source cursors. Regenerating curriculum percentages after a
throughput update must never reset those cursors or change which data earlier
tokens represented. The current `phase_source_mixes` entries have exact token
budgets summing to `total_train_tokens`; they are not live budget-aware rules.
The future-event/continuation protocol in section 5 must generate only the
unconsumed suffix and preserve the consumed-stream proof.

### Quality conditions on every extension

- Keep every critical language/domain represented from the start and in the
  lowest funded horizon. Do not defer all math/code/Indic quality work to a
  nominal final tranche that disappears if throughput is lower.
- Route each additional tranche through an approved data inventory. Record
  unique documents/tokens, effective repetitions and actual per-source
  exposure. Prevent accidental re-reading caused by manifest regeneration.
- Test a high-quality finishing mixture against the unchanged mixture at
  matched tokens/compute. Cleaner text is promising; excessively narrow
  synthetic or English-heavy annealing can still damage other capabilities.
  Do not simultaneously change quantizer, LR and data mixture in the initial
  QAT recovery comparison.
- Evaluate per-domain learning and generated capabilities at fixed token
  milestones; include equal-GPU-hour comparisons for optimization choices.
  Require a practical improvement margin, not a tiny favourable aggregate.
- If validation improvement stalls, first inspect source exhaustion,
  repetition, clipping, FP8 range behaviour and objective weighting. Do not
  automatically increase batch/LR or count more low-quality tokens as success.
- Preserve both a qualified dense fallback and the best validated eligible
  final artifact. A last checkpoint is not automatically the best checkpoint.
  Keep a final holdout out of checkpoint/model selection.

## 5. Required implementation: explicit future schedule events

The current [`terminus/train.py`](../../../terminus/train.py) derives its
learning horizon from `manifest.config.usable_train_tokens` and computes WSD
decay relative to that endpoint. The resume contract binds the training/data
configuration. Editing total tokens or a manifest under the same run is not
an exact resume.

[`terminus/transition.py`](../../../terminus/transition.py) implements an
explicit dense-to-ternary child with stream-continuation checks. It is not a
general dense-to-dense horizon extension API. Preserve those protections.

Implement a reviewed production scheduler/continuation protocol before
promising unattended elastic training:

1. Separate the available data envelope from the current learning endpoint.
   Represent phase, fixed warmup, plateau status, committed decay start/end,
   minimum finishing work and schedule-event sequence in checkpoint state.
2. Allow a same-mode child or prospective schedule event at a complete update.
   Prove that all past LR values and all consumed data are unchanged. Carry
   weights, moments, parameter-group settings, RNG, quantizer/cache rules,
   global token/update counters and the exact next data cursor.
3. Bind every event to its parent checkpoint/hash, source/runtime identity,
   budget ledger, rate evidence, remaining data and finishing forecast. All
   ranks must agree before consuming the next update. Commit atomically and
   make replay idempotent; an incomplete event cannot become a new history.
4. Permit only future plateau changes before a phase's decay commitment.
   Reject attempts to restart warmup, rewrite a consumed mixture, move a
   committed decay or bypass a mismatched resume contract.
5. Count all allocated GPUs and wall time across retries and restarts. Resume
   must not reset the budget clock. Re-evaluate required finishing work after
   recovery. The controller may refuse more plateau work; the human retains
   control of paid launches/stops.
6. Export state, exact schedules, source exposures and quality receipts into
   an external final packet. A forecast must never be accepted as a native
   capability or quality receipt.

Required CPU/integration cases before native testing:

- Two different future horizons produce identical LR/data history throughout
  their common warmup and plateau; endpoint rounding preserves the budget.
- A slow forecast preserves finishing minima; a fast forecast cannot exceed
  the approved data envelope or spend contingency twice.
- Correct GPU-hours versus node-hours and no double charging of already
  completed work/cadence; expired or nonfinite rate evidence fails closed.
- Same-mode event/resume preserves optimizer/RNG/cursor and all prior source
  offsets; repeated or partially written events cannot skip/duplicate data.
- Extension after decay commitment, a missing QAT minimum, an exhausted
  source or a conflicting rank event is rejected.
- Eight-rank event agreement, kill/restart, provider-network backup and final
  checkpoint/export recovery pass on the intended node.

Normal variability can be budgeted. An arbitrarily long outage cannot be
reconciled with both an unchanged schedule and an unchanged deadline. Define
an explicit emergency outcome: retain the last verified artifact, record any
unfinished quality obligation, and do not label a truncated phase a successful
production finish.

## 6. Execution order

1. Close the offline data, evaluation and implementation gates in
   [QUALITY_GATES.md](QUALITY_GATES.md); finish preparation while GPUs are not
   billed. Implement and CPU-test the prospective schedule/ledger protocol.
2. Complete the existing bounded 5B proxy comparison and single transferred
   full-size confirmation from the [Kimi handoff](../../kimi_h100_5b/README.md).
   Use actual billed time; a forecast is not permission to exceed 50 hours.
3. On MI325X, establish native numerics, full-size fit, one/eight-rank update
   equivalence, sustained dense/QAT rates, recovery and task-quality evidence.
   The [five-hour pilot](FIVE_HOUR_PILOT.md) is a bounded first decision stage,
   not assurance that all QAT/long-horizon gates finish in five hours.
4. Lock the qualified learning/data policy and conservative finishing
   obligations; generate the first external production contract. Approve
   further tests only for unresolved measured risks, with explicit costs.
5. Train the extendable plateau, update forecasts, and commit each ending
   while it remains comfortably funded. Report model quality and actual
   tokens/compute; do not optimize for an unsupported round token number.
