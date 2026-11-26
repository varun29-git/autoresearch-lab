# Training failure risks and required controls

Engineering review: 2026-10-26. Read before generating the final run bundle.
This page describes code behavior and launch requirements; native measurements
and campaign receipts belong outside Git. The
[2026-11-01 quality audit](QUALITY_GATES.md) distinguishes newly implemented
H100 qualification/recovery tools from remaining AMD production requirements.
Use the focused [5B proxy/transfer plan](LR_BATCH_MONITORING.md), not a fresh
full-width sweep, as the current experiment order. The
[adaptive compute specification](ADAPTIVE_COMPUTE_PLAN.md) defines prospective
horizon changes while protecting finishing work; current source defaults and
fixed-horizon resume contracts remain unchanged.

## Corrected failures

### Unexpected target masks could change gradient normalization

The trainer normalizes loss by the declared global update-token count. It now
compares that denominator with the globally reduced count of valid targets
and rejects a mismatch before applying the optimizer step. This preserves
the intended objective for normal data and legitimate padded final batches.
CPU regressions cover partial/all-masked corruption and unchanged weights and
progress on failure. Native eight-MI325X qualification remains required.

### FP8 telemetry assumed the H100 range

Operand boundary occupancy now derives the maximum finite value from the
actual FP8 dtype, including E4M3FN (448) and E4M3FNUZ (240). CPU tests verify
both formats and E5M2 variants without modifying cached operands. This fixes
an observation gap; it does not establish all-operand native AMD saturation
or quantizer correctness. New source-bound receipts are required before
using the changed code in a previously pinned campaign.

### A valid new corpus could silently break dense-to-QAT continuation

Previously the converter checked the tokenizer, checkpoint, token counter and
batch alignment without proving that the child manifest described the same
preceding stream. A reordered corpus with valid new checksums could pass;
the carried cursor would then skip unseen examples or repeat earlier ones.

Full manifest validation now records ordered shard identities, dtype/token
counts, source schedules, phase boundaries and source offsets in the run and
checkpoint contracts. Conversion compares the consumed prefix for `phase_mix`
and the entire common planned prefix for `continue`. The proof is recorded in
the transition receipt. Batch regrouping, relocation and appending immutable
shards remain possible. Changing previously required shards or routing fails.
These checks do not establish document quality or corpus deduplication.

Legacy parent contracts without stream metadata fail closed for continuation.
Keep their original source/runtime and checkpoints intact; using such a parent
requires an explicit migration that proves its old manifest identity and
stream. Do not edit receipts or select `restart` to bypass a failed proof.

### The FP8 numerical gate could accept completely zeroed small gradients

The shared numerical comparator allowed an absolute error below `2e-5` even
when relative error was large. At token-normalized gradient magnitudes, an
all-zero result could therefore pass. Existing projection cases also used
unit-scale upstream gradients and short reduction axes.

The FP8 gate now requires relative agreement without that absolute exemption,
checks both unit and `2^-24` upstream gradients, and adds a 65,536-row wgrad
reduction at 256-wide geometry. Quantizer checks also require relative
agreement at tiny magnitudes. CPU tests verify rejection of zeroed gradients
and the backward wiring. The native kernels must still pass on MI325X; the
added long reduction is not a substitute for full-width updates.

## Remaining launch blockers

| Risk | Consequence | Required control |
| --- | --- | --- |
| AMD ternary QAT port has no native acceptance result | The prepared QAT path could compile incorrectly or learn poorly on MI325X; MI325X needs no ternary cores | Run the FNUZ `-1/0/+1` kernel and block-scale preflight, then qualify forward/backward, conversion, optimizer-state carry, quality and recovery **before** a mixed campaign |
| FP8/attention/Muon have no native MI325X acceptance receipts | Wrong transposes, FNUZ interpretation, attention boundaries or reductions could damage learning despite finite loss | Run the strengthened kernel preflight, full-size updates and an independent BF16 trajectory comparison on the frozen image |
| Eight-rank numerical equivalence is unmeasured | Missing or differently scaled gradients can look like a speed improvement | Compare the same global examples/update on one and eight ranks: losses, pre-clip gradients, optimizer moments and parameter deltas; allow declared reduction-order differences |
| Microbatch 16 has no full-update HBM measurement | Optimizer, attention or validation workspaces can OOM after forward succeeds | Keep microbatch adjustable at fixed 2M global tokens: compare 8×accumulation 8, 16×4 and conditionally 32×2; measure peaks through backward, optimizer, checkpoint and validation, retaining at least 15% usable HBM headroom |
| Two-million-token batch and 480-update warmup are provisional | Fewer updates per token change optimizer history and regularization; a stable loss does not establish good learning | Keep transferred peak rates first, monitor the warmup-to-plateau boundary and the 4.0265B-token milestone; add the 1M control only if evidence warrants it |
| Final corpus and held-out quality are unverified | Repetition, poor domain balance or contamination can consume the entire budget without useful learning | Verify actual document deduplication, sequence construction, mixture exposure and frozen held-outs; checksum/provenance fields alone cannot prove these properties |
| Durable backup, retention and restore are unfinished | Disk exhaustion stops checkpointing; node loss can erase the only usable state | Establish bounded local retention, verified durable copies and a successful restore drill; see storage requirements below |
| Token horizon is selected before measuring both phase rates | The allocation can end before WSD decay or before adequate QAT recovery | Freeze a funded horizon using complete-update rates plus measured operational costs within the combined 5,760 GPU-hours |

## Schedule and launch mistakes to prevent

- **Preserve the HPO work.** Width transfer and batch transfer are distinct.
  Keep the selected Muon/AdamW/norm peak rates and spectral rule. Do not
  automatically multiply rates or weight decay by the batch ratio. SmolLM3
  supports the plausibility of a similar batch scale; its AdamW recipe does
  not validate this Muon schedule. A mandatory 393k comparison is not added.
- **Freeze the funded schedule before a run intended for continuation.**
  For example, a 10B-token pilot manifest schedules decay near its 10B end.
  A final 1T-token manifest schedules decay near 1T. Replacing the pilot
  manifest/config at 10B changes data and LR history, and exact resume rejects
  the changed contract. If the first 10B tokens should be a production prefix,
  stage and validate the final corpus, start with its 1T manifest/schedule,
  and stop at a chosen update. Otherwise budget a separate qualification run.
  The same rule applies to dense and QAT phases.
- **Bound smoke runs explicitly.** `--purpose smoke` changes warmup and logging;
  it does not impose a trainer update limit. Use a separate small manifest and
  a benchmark/update bound. Never launch the ordinary unbounded command with
  a smoke bundle or treat its checkpoint as the selected long-run schedule.
- **Fit every schedule to its phase and batch.** Use the actual complete-update
  switch point. The supplied HPO endpoint `500170752` is not aligned to a
  2,097,152-token update. Its 50M warmup and
  100M decay are approximately 24 and 48 updates at that batch. Preserve the
  candidate QAT peak rates, then derive phase lengths from the funded QAT
  horizon; keep global tokens/update fixed initially across conversion.

- **Freeze the source and runtime.** These fixes change source identities.
  Regenerate staging, preflight and runtime receipts for the new revision.
  Existing exact-resume contracts must continue using their recorded source;
  changing a checkout during a paid run is not an approved state migration.

The native preflight checks kernels and reduced-model arithmetic before a
paid training trial. Full-size HBM fit, eight-rank reduction equivalence,
learning through warmup/plateau, and checkpoint recovery require bounded
real-data updates on the intended node before the month-long run.

## Storage and recovery capacity

The selected model has 1,295,253,504 Muon parameters and 151,224,622 auxiliary
Adam parameters. FP32 weights plus Muon momentum and Adam first/second moments
imply approximately **12.18 GB per initialized checkpoint**, before serialization
metadata and additional buffers. This is a tensor-count estimate, not a
measured checkpoint file size; gradients are not saved.

The 2,013,265,920-token cadence produces about **497 checkpoints by 1T tokens**,
including the final checkpoint: roughly **6.05 TB** if all are retained.
Corpus shards, validation, temporary checkpoint files, caches and spare space
are additional. A three-checkpoint retention intention does not implement
retention: the existing filesystem maintenance command takes an exclusive
run lock and requires the run to be stopped. Provide a reviewed live backup
and cleanup process or plan explicit stopped maintenance windows.

AWS S3 can hold durable copies, but the current S3 ingestion helper does not
upload checkpoints. Back up only committed payload/receipt pairs, verify the
upload, preserve the source/image/config needed to restore, and download and
resume a real test copy before launch. Test the actual provider-to-AWS network
path from the GPU host: route/firewall and credentials, sustained upload and
download throughput for a full checkpoint, retry behavior, backup lag, restore
time and transfer charges. Size upload workers/retention from those measurements
so backups finish before the local queue fills. Monitor free space throughout.
The trainer's five-minute collective timeout must also accommodate measured
checkpoint/validation stalls; a slow remote volume can abort otherwise healthy
ranks. Prefer a verified local working volume and durable asynchronous copies.

## Existing behavior checked during review

- Global summed losses are scaled by `world_size / global_update_tokens`;
  DDP averaging therefore yields the global token mean. Accumulation uses
  `no_sync` until the final microstep and clips once per complete update.
- Rank striding and microbatch regrouping preserve the ordered global stream.
  Partial final updates count valid targets rather than padding.
- Muon keeps separate logical Q/K/V and gate/up matrices, FP32 master weights
  and moments, with the selected spectral shape rule. Weight caches are
  invalidated after updates; checkpoint loading clears cached optimizer views.
- Checkpoint writes are atomic, have checksums and reject nonfinite model or
  optimizer state. These protections still require durable backups and a
  native interruption/recovery drill.

CPU regression results establish host-side behavior only. Full-width MI325X
numerics, memory, speed, convergence and recovery remain required evidence in
[ACCEPTANCE.md](../ACCEPTANCE.md) and
[LAUNCH_CHECKLIST.md](../LAUNCH_CHECKLIST.md).
