# MI325X pretraining readiness

**Status on 2026-10-26: the month-long dense-to-ternary run is not cleared.**
The repository has CPU-tested configuration candidates and a batch stability
plan. Eight-MI325X throughput, memory fit, learning transfer and native AMD
ternary execution have not been verified. The human launches and stops paid
GPU work.

Ternary QAT does not require ternary arithmetic units. The existing H100
training path represents effective projection weights as FP8 codes `-1`, `0`
or `+1` with block scales while keeping FP32 latent weights and optimizer
state. MI325X FP8 arithmetic can in principle use the same approach with its
E4M3 FNUZ format. The current AMD model/backend and launcher explicitly
reject ternary mode; this is a software implementation and native
qualification gap.

The dense rates intentionally transfer from smaller-model HPO through the
selected spectral Muon width rule and the embedding-rate bridge. A full-scale
HPO rerun is not planned. The open batch decision is separate from that width
transfer: `batch_bridge_mb12` preserves the HPO global batch on eight GPUs
as an optional diagnostic, not a required qualification run.
The [batch-transfer decision](../BATCH_STABILITY.md#can-the-hpo-rates-reach-the-intended-2m-batch-without-full-retuning)
tests the 2M profile with unchanged rates first, then adds a smaller-batch
comparison only if evidence warrants it;
published Muon results support this as a starting protocol but do not certify
the 2M batch for CI-1.

For the learning-rate assessment, read [LR_REFERENCES.md](LR_REFERENCES.md).
It compares MiniCPM's width-scaled LR and Moonlight's RMS-matched Muon rates
with the selected recipe, identifies MiniCPM5's architectural and disclosure
differences, and records that the inspected Gemma reports do
not disclose a usable pretraining peak LR. Embedding/head LR 0.003 remains
an aggressive transfer candidate supported by the archived width bridge;
published recipe coefficients alone do not establish whether it is too high.

The same note records a separate **horizon-transfer gap**: the archived HPO
ranked at roughly 500M tokens after roughly 250M warmup, before any LR decay.
Width transfer and an early production bridge do not qualify trillion-token
learning or the final dense/QAT schedule. Keep HPO as the search center and
bound any longer LR comparison by measured cost and the qualification reserve.

Read this page before generating the final run configuration or starting a
training campaign. Review the corrected silent failures and remaining launch
blockers in [TRAINING_RISKS.md](TRAINING_RISKS.md). The detailed evidence and test order are in
[BATCH_STABILITY.md](../BATCH_STABILITY.md),
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
- [ ] **Global batch and learning:** Run `balanced` (2,097,152) with the
  selected rates first. Both this batch and the 480-update warmup are
  provisional training-readiness choices. Check native stability,
  complete-update speed, warmup and held-out learning through the
  4,026,531,840-token plateau milestone.
  The selected 480-update warmup is shorter in updates than the archived HPO
  warmup (~636) and SmolLM3's 2,000-step warmup; inspect the first plateau
  updates before accepting that schedule.
  If evidence warrants a smaller-batch comparison, test `batch_1m_mb16`
  (1,048,576) at matched tokens **and** allocated GPU-hours on the same model,
  data prefix and frozen held-outs. Use the 393,216-token HPO-batch profile
  only to resolve a remaining question worth its GPU-hour cost. Test
  3,145,728 or 4,194,304 only if speed evidence merits a quality bridge.
  Fixed learning rates do not preserve the number of Muon, momentum and
  weight-decay updates per token.
- [ ] **Dense-to-QAT transfer:** Qualify FP8 ternary-weight quantization and
  matrix operations on MI325X, plus full-model conversion, before relying on
  a QAT phase. Use the supplied QAT
  peak-rate candidate (Muon 0.0085; embedding AdamW 0.003; norms 0.00015;
  Q/K norms 0.00005) as a candidate, not an established AMD winner. Keep an
  intact dense checkpoint and compare QAT with a same-data dense continuation.
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
items are complete, the run is ready for bounded qualification trials only.
