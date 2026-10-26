# MI325X pretraining readiness

**Status on 2026-10-26: the month-long dense-to-ternary run is not cleared.**
The repository has CPU-tested configuration candidates and a batch stability
plan. Eight-MI325X throughput, memory fit, learning transfer and native AMD
ternary execution have not been verified. The human launches and stops paid
GPU work.

The dense rates intentionally transfer from smaller-model HPO through the
selected spectral Muon width rule and the embedding-rate bridge. A full-scale
HPO rerun is not planned. The open batch decision is separate from that width
transfer: `batch_bridge_mb12` preserves the HPO global batch on eight GPUs.

Read this page before generating the final run configuration or starting a
training campaign. The detailed evidence and test order are in
[BATCH_STABILITY.md](../BATCH_STABILITY.md),
[SPEEDUP_TRIAGE.md](SPEEDUP_TRIAGE.md),
[LAUNCH_CHECKLIST.md](../LAUNCH_CHECKLIST.md), and
[ACCEPTANCE.md](../ACCEPTANCE.md).

## Decisions to resolve on the actual eight-GPU node

- [ ] **Microbatch and HBM:** Benchmark complete, warm updates at a fixed
  global batch. Start with the 2,097,152-token `balanced` profile
  (microbatch 16, accumulation 4); compare `mb8` and conditionally `mb32`.
  At 3,145,728 tokens/update, compare `batch_3m_mb16` against
  `batch_3m_mb24`. Require at least 15% of usable HBM free through optimizer,
  validation and checkpoint peaks. Microbatch 16 is a starting point, not a
  measured optimum.
- [ ] **Global batch and learning:** Compare the HPO 393,216-token batch,
  including the one-microstep `batch_bridge_mb12` execution profile,
  `batch_1m_mb16` (1,048,576) and `balanced` (2,097,152) at matched tokens
  **and** allocated GPU-hours on the same model, data prefix and frozen
  held-outs. Test 3,145,728 or 4,194,304 only if speed evidence merits a
  quality bridge. Fixed learning rates do not preserve the number of Muon,
  momentum and weight-decay updates per token.
- [ ] **Dense-to-QAT transfer:** Qualify native MI325X ternary kernels and
  full-model conversion before relying on a QAT phase. Use the supplied QAT
  peak-rate candidate (Muon 0.0085; embedding AdamW 0.003; norms 0.00015;
  Q/K norms 0.00005) as a candidate, not an established AMD winner. Keep an
  intact dense checkpoint and compare QAT with a same-data dense continuation.
- [ ] **Switch and schedules:** Set `phase_start_tokens` to the actual
  complete-update dense checkpoint. The supplied 500,170,752-token HPO
  endpoint is half an update at the 2M batch. Its 50M warmup and 100M decay
  span only about 24 and 48 updates at that batch; derive the long-run QAT
  schedule from the measured QAT horizon. Keep global tokens/update fixed
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
