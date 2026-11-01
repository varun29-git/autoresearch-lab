# MI325X pretraining readiness

**Status on 2026-11-01: the month-long dense-to-ternary run is not cleared.**
**Start with the [quality gates](QUALITY_GATES.md) and
[adaptive compute plan](ADAPTIVE_COMPUTE_PLAN.md).** Fix the 5,760 GPU-hour
budget and qualified learning/data policy; let measured dense/QAT throughput
determine the useful token horizon. Extend only the uncommitted WSD plateau
while protecting cooldown, QAT recovery and final evaluation. This policy
requires explicit future schedule/data transitions; editing a live manifest
does not implement it. Current production builders still generate fixed
horizons. Close the offline data/evaluation gates before long paid training.

**Focused learning qualification:** reuse the completed HPO and use the available single
Azure H100, capped at 50 hours, for [lower-LR controls at fixed 2M batch,
with explicit monitoring and a transferred-winner check](LR_BATCH_MONITORING.md).
Run the LR controls and historical anchor on the **194M width-scaled proxy**,
then transfer the winner and perform a short full-size confirmation. Do not
run every LR candidate at 1.446B. The user sets **5B tokens per proxy**.
Five proxy controls plus a short full-size confirmation fit near 50 hours
only at the forecast rates; the additional proxy batch comparison is deferred
by default. Preserve the 50-hour cap and charge all overhead.
For execution preparation, use the
[Kimi handoff folder](../../kimi_h100_5b/README.md), including its paste-ready
prompt, frozen campaign contract, implemented qualification tools and CPU
forecast tool. The source audit below distinguishes implemented H100 tools
from remaining AMD production integration.
The 12 completed 200M
and six 700M trials were verified against their external event-file hashes.
The broad fresh HPO search remains deferred. One GPU can test the intended 2M
global batch through accumulation; eight GPUs do not multiply that batch again.
The [complete qualification plan](FULL_QUALIFICATION_PLAN.md) retains the
broader controls and optional deeper search.
For a subsequent bounded AMD stage, see the [five-hour pilot](FIVE_HOUR_PILOT.md):
40 GPU-hours maximum, with explicit prerequisites and remaining qualification
gaps. Its recommended assurance-first option combines direct correctness tests,
one longer dense trajectory and a real recovery drill. It does not complete
all remaining production gates. It is a bounded decision stage.
The user's latest direction is to use the existing tuning and Muon literature
to narrow the missing transfer checks. Production rates, global batch, warmup
and phase allocation still need the corresponding evidence; completed HPO is
useful evidence, not automatic clearance of the month-long run.
The human launches and stops paid GPU work.

The earlier broad plan budgeted a **960 GPU-hour total qualification envelope
inside the 5,760-hour allocation**. The Azure-first proposal narrows the search
before AMD rental; it does not require spending that entire envelope. The
broader plan describes a joint Muon/AdamW screen, norm/decay tests, batch/recipe
comparisons, longer finalists, seed confirmation, cooldown probes, QAT/control
trials, fault recovery and token-budget accounting. The current focused plan
determines which experiments come first; the larger search is optional.

The single-H100 campaign now has a bounded runner, horizon views, full-state
telemetry, selection/transfer, an allocation ledger and checkpoint/S3 recovery
tooling. These are implemented, not all still backlog items. They are not an
eight-MI325X production runner or acceptance evidence for that node. The AMD
validator still locks the earlier optimizer recipe. Native QAT, production
schedule extension, generated task evaluation and all-rank acceptance remain
open; see the [source audit and quality requirements](QUALITY_GATES.md).
Native AMD throughput, memory fit, learning and recovery remain unverified.

Ternary QAT does not require ternary arithmetic units. The existing H100
training path represents effective projection weights as FP8 codes `-1`, `0`
or `+1` with block scales while keeping FP32 latent weights and optimizer
state. MI325X FP8 arithmetic can in principle use the same approach with its
E4M3 FNUZ format. The current AMD model/backend and launcher explicitly
reject ternary mode; this is a software implementation and native
qualification gap.

For historical LR evidence, read [LR_REFERENCES.md](LR_REFERENCES.md). It records
the original width rules, the 0.003-versus-0.002 AdamW distinction, external
recipes and the short HPO horizon. These support the focused transfer checks.
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
- [ ] **Global batch and learning:** Begin with the proxy LR controls and
  historical anchor in `LR_BATCH_MONITORING.md`, then compare proxy batches
  if the measured budget permits. Reserve a full-size check of the transferred
  winner at the intended global batch. Each initial trial
  has a complete approximately 5B-token WSD with 10% warmup and 20% cooldown;
  this does not select the eventual production schedule. Compare quality at
  matched tokens and matched GPU-hours. Use measured results to price further
  full-size batch, seed and longer-horizon tests from the broad plan; they are
  not all mandatory upfront experiments. A proxy winner alone does not clear
  the production batch, long horizon or phase allocation.
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
- [ ] **Final go/no-go:** Freeze the selected batch, learning/data policy,
  schedule-transition rules, finishing minima, runtime identity, recovery
  plan and acceptance evidence. Forecast both phases within the
  **5,760 GPU-hour total** using measured sustained rates. Keep the future
  token horizon adjustable only through the reviewed protocol, then commit
  each ending before decay. Until that protocol is implemented and tested,
  current generated configs retain a fixed horizon. A 60/40 dense/QAT token
  split is a planning default, not a proven optimum.

Record measured configs, data manifests, preflight reports, comparison
receipts, evaluation results and checkpoints **outside Git**. Until these
items are complete, the long run is not cleared. Existing tools support
bounded bring-up and single-H100 qualification; the remaining AMD work is
specified in [QUALITY_GATES.md](QUALITY_GATES.md) and
[ADAPTIVE_COMPUTE_PLAN.md](ADAPTIVE_COMPUTE_PLAN.md).
