# MI325X pretraining readiness

**Status on 2026-10-28: the month-long dense-to-ternary run is not cleared.**
Start with the [complete qualification and tuning plan](FULL_QUALIFICATION_PLAN.md).
The user now requests qualification as if no prior HP tuning existed. All
production rates, global batch, warmup and phase allocation are candidates to
measure on the actual 1.446B model; earlier HPO confers no automatic promotion.
The human launches and stops paid GPU work.

The new plan recommends **eight ranks for every selection-bearing HP trial**,
individual GPUs for arithmetic/fit/debugging, and a **960 GPU-hour total
qualification envelope inside the 5,760-hour allocation**. It includes a
joint Muon/AdamW screen, norm/decay tests, mandatory 1M-versus-2M batch/recipe
comparisons, longer finalists, seed confirmation, cooldown probes, QAT/control
trials, fault recovery and a measured final token budget. It supersedes the
older unchanged-rate-first and 320/480-hour qualification policies.

This is a plan, not newly implemented training machinery. The current AMD
validator locks the old optimizer values; explicit bounded qualification and
dense-fork tooling, native QAT, live durable backup, generated task evaluation
and all-rank safety/acceptance work remain in the plan's engineering backlog.
Native throughput, memory fit, learning and recovery remain unverified.

Ternary QAT does not require ternary arithmetic units. The existing H100
training path represents effective projection weights as FP8 codes `-1`, `0`
or `+1` with block scales while keeping FP32 latent weights and optimizer
state. MI325X FP8 arithmetic can in principle use the same approach with its
E4M3 FNUZ format. The current AMD model/backend and launcher explicitly
reject ternary mode; this is a software implementation and native
qualification gap.

For historical LR evidence, read [LR_REFERENCES.md](LR_REFERENCES.md). It records
the original width rules, the 0.003-versus-0.002 AdamW distinction, external
recipes and the short HPO horizon. These are background for the new search.
The [older batch plan](../BATCH_STABILITY.md) remains useful for exact batch
arithmetic and source evidence; its prior experiment order is superseded.

Read this page before generating the final run configuration or starting a
training campaign. Review the corrected silent failures and remaining launch
blockers in [TRAINING_RISKS.md](TRAINING_RISKS.md). The detailed evidence and test order are in
[the full plan](FULL_QUALIFICATION_PLAN.md), supported by the historical
[batch evidence](../BATCH_STABILITY.md),
[SPEEDUP_TRIAGE.md](SPEEDUP_TRIAGE.md),
[LAUNCH_CHECKLIST.md](../LAUNCH_CHECKLIST.md), and
[ACCEPTANCE.md](../ACCEPTANCE.md).

## Decisions to resolve on the actual eight-GPU node

- [ ] **Microbatch and HBM:** Benchmark complete, warm updates at a fixed
  global batch. Start with the 2,097,152-token `balanced` profile
  (microbatch 16, accumulation 4); compare `mb8` (accumulation 8) and
  conditionally `mb32` (accumulation 2). Microbatch remains a measured
  variable until eight-GPU full-update fit and speed checks finish; all three
  profiles retain the same global tokens per update.
  At 3,145,728 tokens/update, compare `batch_3m_mb16` against
  `batch_3m_mb24`. Require at least 15% of usable HBM free through optimizer,
  validation and checkpoint peaks. Microbatch 16 is a starting point, not a
  measured optimum.
- [ ] **Global batch and learning:** Follow the full plan's joint LR grid,
  auxiliary tests and mandatory two-recipe comparison at both 1,048,576 and
  2,097,152 tokens/update. Continue complete finalists to 12.0796B tokens,
  confirm a second seed and investigate cooldown behavior within the measured
  budget. The initial 480-update warmup at 2M is provisional; test a longer
  ramp if indicated. The 393k batch is an optional diagnostic; 3M/4M require
  a measured reason to add their learning comparisons. Compare quality at
  matched tokens and at matched allocated GPU-hours.
- [ ] **Dense-to-QAT transfer:** Qualify FP8 ternary-weight quantization and
  matrix operations on MI325X, plus full-model conversion, before relying on
  a QAT phase. Treat the supplied historical QAT peak rates as background.
  Follow the full plan's QAT rate
  multipliers relative to the newly selected dense recipe, with an intact
  dense parent and a same-data dense continuation control.
- [ ] **Switch and schedules:** Set `phase_start_tokens` to the actual
  complete-update dense checkpoint. The supplied 500,170,752-token HPO
  endpoint is half an update at the 2M batch. Its 50M warmup and 100M decay
  span only about 24 and 48 updates at that batch. Derive **every** production
  phase's warmup and decay from its funded token horizon and selected batch;
  the QAT HPO schedule is one example. Keep global tokens/update fixed
  across the switch initially. A CPU-tested transition can rebase the
  microstep cursor if QAT needs another microbatch; native eight-rank
  conversion and exact resume still need verification.
- [ ] **Final go/no-go:** Freeze the selected batch, dense/QAT token budgets,
  data mixture, schedules, runtime identity, recovery plan and acceptance
  evidence. Forecast both phases using measured complete-update rates within
  the **5,760 GPU-hour total**. A 60/40 dense/QAT token split is a planning
  default, not a proven optimum.

Record measured configs, data manifests, preflight reports, comparison
receipts, evaluation results and checkpoints **outside Git**. Until these
items are complete, the long run is not cleared. Existing tools support
bounded native bring-up; the new HP/QAT campaign first needs the explicit
qualification machinery listed in the full plan.
