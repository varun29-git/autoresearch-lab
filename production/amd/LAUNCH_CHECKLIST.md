# Eight-MI325X production launch checklist

Reviewed 2026-10-25; updated with [selected configurations](CONFIG_SELECTION.md).
The code is ready to begin native bring-up once an accessible host and test
data are available. The month-long production run is not yet cleared.
No AMD host, full production data snapshot, final run configuration or native
acceptance receipts have been verified in this session. These may exist
externally; absence from Git is intentional and does not prove they are missing.

Use [README.md](README.md) for commands and [ACCEPTANCE.md](ACCEPTANCE.md) for
the numerical and operating gates. Keep configs, manifests, receipts, traces,
checkpoints and evaluation results outside this repository. The human launches
and stops paid GPU campaigns.

## Already implemented or selected

- [x] Selected 24-layer, width-2304, 65,536-vocabulary, 4096-context model;
  1,446,478,126 parameters; Peri-LN, learned Q/K gains and value residual.
- [x] Selected spectral Muon/auxiliary AdamW implementation and starting rates.
- [x] Selected eight-GPU starting profiles: 2,097,152-token global batch,
  microbatch 16 / accumulation 4, TE FP8, roughly 1B-token warmup, final 10%
  decay, fixed evaluation coverage and token-based operational cadence.
  Selection is complete; native qualification remains due.
- [x] Frozen tokenizer artifact and tokenizer implementation.
- [x] Three explicit AMD FP8 projection candidates and a BF16 control;
  AITER/CK attention choices, FNUZ reference and native numerical gates.
- [x] Pinned runtime build recipe, external config generator, immutable data
  staging receipts, runtime identity checks and bounded training launcher.
- [x] Real-data complete-update benchmark, component and RCCL probes, and
  5,760 GPU-hour budget calculator.
- [x] Shared checkpoint/resume, filesystem archive, local status and NLL
  evaluation tools.
- [x] Recorded local regression run: 360 passed, 27 skipped and 183 subtests
  passed. These are historical local checks, not AMD hardware evidence.

## Work before the first controlled GPU training trial

Data preparation and runtime/kernel bring-up can proceed independently.
A representative validated subset is sufficient for the first bounded trial;
the full production corpus and held-outs must be ready before the long run.

1. **Confirm access and the allocation.**
   - [ ] Record SSH host, username, port, working key and host identity.
   - [ ] Confirm one complete eight-GPU node, actual MI325X SKU/capacity,
     allocation duration, and whether billing has already started.
   - [ ] Confirm permissions for the runtime, mounts, GPU devices and logs;
     choose external data, run, cache and temporary directories.
   - [ ] Arrange a persistent session or supervisor so an SSH disconnect does
     not terminate the job. Define who responds to an incident.

2. **Freeze the data specification.**
   - [ ] Inventory actual corpora, licenses/provenance, language/domain mix,
     unique token availability, sampling/repetition and phase order.
   - [ ] Confirm the frozen tokenizer SHA256, special IDs and encoding rules.
     Shards made with a different tokenizer need regeneration.
   - [ ] Declare data filters and duplicate/contamination checks, including
     overlap between training and evaluation. Existing preparation utilities
     do not establish that the full production corpus has passed these checks.
   - [ ] Prepare a representative, separately manifested test dataset before
     hardware bring-up. It must have sufficient tokens for the test updates.

3. **Prepare and validate the production data.**
   - [ ] Produce the final uint16 token shards, 4096-token stream contract,
     tokenizer copy, hashes, provenance and manifest outside Git.
   - [ ] Verify language/domain coverage, actual token counts, padding/EOS and
     mixture exposure. Repeated training tokens do not require duplicate
     stored corpus copies.
   - [ ] Prepare frozen multilingual/domain development sets and separate
     final evaluation sets. Record metrics and tolerances before comparisons.
   - [ ] Benchmark and finish large-scale preprocessing before using the GPU
     month where possible; raw text, tokenization and deduplication workspace
     need their own capacity and time allowance.

4. **Provision and stage storage.**
   - [ ] Size the working volume from actual shards plus checkpoint growth,
     temporary files, runtime/JIT caches and free-space headroom. The earlier
     6 TB suggestion assumes up to 4 TB of unique-token shards and regular
     checkpoint archiving/pruning; it is not a universal requirement.
   - [ ] If using AWS S3, select the bucket/prefix, region, credentials through
     the normal credential provider, access permissions and retention policy.
   - [ ] Verify transfers in both directions and include their cost/time.
     Stage data locally; the training loader uses filesystem memory mapping.
   - [ ] Keep all manifest files accessible to every rank, make the staged
     shard/tokenizer snapshot read-only, and measure actual read/write behavior.

5. **Build the Linux ROCm runtime.**
   - [ ] Build the pinned image or runtime on Linux x86-64; preserve the selected
     ROCm torch/Triton and install the pinned AITER, FA-CK and Muon versions.
     Include TE if testing `fp8_te`.
   - [ ] Resolve compilation, import, API and dependency errors. No native
     build has yet demonstrated that the complete combination works.
   - [ ] Run `pip check` and appropriate tests in the built environment; freeze
     its package set, image digest and source revision.
   - [ ] Put JIT caches outside installed source packages, preserve warmed
     caches, and expose the GPU devices/shared memory to the container.

6. **Inspect the real hardware and host.**
   - [ ] Check all eight GPUs, gfx942, usable HBM, xGMI peer topology, ROCm/driver
     compatibility, GPU error state and absence of unintended partitioning.
   - [ ] Record actual power limits, sustained clocks, temperatures and cooling
     behavior under load. A 1,000 W specification is not a measurement.
   - [ ] Inspect CPU/RAM/NUMA placement and storage/network paths; remove host
     bottlenecks found by measurement.

7. **Generate the bounded test configurations.**
   - [ ] Use `production.amd.prepare` with the real manifest and fresh external
     outputs. `balanced` selects microbatch 16, accumulation 4 and global batch
     2,097,152; `mb4`/`mb8` preserve that global batch at smaller microbatches.
     `batch_3m_mb16` and `batch_3m_mb24` compare microbatches 16/24 at the
     same 3,145,728-token global batch (accumulation 6/4).
   - [ ] For small-data bring-up use `--purpose smoke`; for matched-token
     quality work use the shared long schedule and bounded update counts from
     `selection.json`. Do not resume a smoke checkpoint into production.
   - [ ] Confirm the generated manifest, fixed 1,024 validation sequences per
     scope key, schedule and update bounds. The 4,194,304-token global batch
     is now an explicit `large_batch` comparison, not the initial choice.
   - [ ] Run count-only verification and full data staging validation. Retain
     the staging receipt for the exact source/model/data snapshot.

8. **Pass native numerical preflight.**
   - [ ] Test each candidate being considered on gfx942: FNUZ quantization,
     padded tails, forward/dgrad/FP32 wgrad and weight-cache invalidation.
   - [ ] Check local/global attention forward/backward, GQA, exact window and
     cached offsets; test learned Q/K gains, value residual, pointwise and CE.
   - [ ] Pass the reduced-model composition and spectral-Muon comparisons.
   - [ ] Fix failures without weakening the arithmetic contract or widening
     tolerances to hide a bug. Save the successful preflight and runtime
     receipts after native initialization.

At this point, a bounded real-data training trial is appropriate. A preflight
receipt alone does not clear the long production run.

## Work before the month-long run

9. **Prove full-size training fits and updates correctly.**
   - [ ] Run several complete updates of the actual 1.446B model, first with a
     conservative microbatch and then with candidate larger batches.
   - [ ] Check finite loss/gradients, clipping, learned gains/value coefficients,
     actual optimizer-state creation and parameter updates.
   - [ ] Measure allocated/reserved HBM after caches and optimizer state are
     warm, including checkpoint and validation peaks.

10. **Validate eight-rank correctness.**
    - [ ] Run the RCCL correctness/payload probe and inspect peer traffic.
    - [ ] Compare a matched global update on one and eight ranks using the
      same initial state, token order, schedule and clipping. Adjust local
      accumulation; do not change the global batch for this comparison.
    - [ ] Compare gradients, parameter deltas, optimizer moments and logical
      Q/K/V and gate/up updates under declared numerical tolerances.
    - [ ] Build the native comparison procedure/automation needed for this
      evidence. The existing CPU DDP test does not execute this AMD gate.

11. **Measure and select execution settings.**
    - [ ] Compare the most promising valid FP8 implementation(s), with the BF16
      candidate as a numerical/performance control where useful.
    - [ ] Measure microbatch 8 and 16 at the 2M batch. With measured HBM
      headroom, compare `batch_3m_mb16` and `batch_3m_mb24` at a fixed 3.146M
      batch; test microbatch 32 at 2M only if it fits the same headroom rule.
      Bridge any change in global batch against the 2M quality baseline.
    - [ ] Tune attention selection, wgrad splits/tiles, head chunking and DDP
      buckets only where profiles show material cost. Further producer fusion
      is additional implementation work with its own numerical tests.
    - [ ] Run alternating complete 20-update comparisons and a 40-update soak
      for finalists. Include all-rank times, data waits, real optimizer work,
      memory and monitoring settings; collect profiles separately.
    - [ ] Measure checkpoint and evaluation time separately, then include them
      in the allocation-level forecast. No earlier throughput estimate is a
      measured acceptance result.

12. **Check early training quality.**
    - [ ] Run matched-token candidate/control trajectories and frozen held-out
      evaluations. Check language/domain regressions and clipping/update trends.
    - [ ] Use known-source H100 records if available; otherwise document the
      comparison against the independent arithmetic oracle and AMD BF16 control.
    - [ ] Continue scheduled quality checks during production. A short stable
      run does not establish final model quality.
    - [ ] If claiming generated task accuracy or code pass@1, implement and
      freeze those protocols. The current production evaluator reports NLL.

13. **Complete durable checkpoint operations.**
    - [ ] Measure real checkpoint size/save duration; verify the selected
      2.013B-token cadence and last-three local retention policy against
      acceptable lost work, free space and durable archive capacity.
    - [ ] Implement or configure verified S3 upload and restore. Existing
      `production.checkpoints` accepts filesystem archives; `terminus.s3io`
      handles read-only ingestion, not checkpoint uploads.
    - [ ] Upload committed payloads and receipts, make recovery pointers refer
      only to complete verified backups, and verify restored SHA256 hashes.
      Multipart ETags are not interchangeable with those hashes.
    - [ ] Define safe cleanup. The existing retention tool requires a stopped
      run; either schedule maintenance or implement/test coordination for live
      backup and retention. Never remove the sole recoverable checkpoint.
    - [ ] Preserve the tokenizer, config, manifest, receipts, source/image
      identity and required run metadata with the recovery artifacts.

14. **Exercise recovery and failure handling.**
    - [ ] Test a graceful stop and exact resume, comparing cursor, RNG,
      parameters and optimizer state with uninterrupted execution.
    - [ ] Have the human test abrupt worker loss in a bounded job; verify the
      worker group exits and partial payloads are not marked as latest.
    - [ ] Download and restore a real checkpoint from S3 into a compatible
      environment, then successfully continue training.
    - [ ] Measure restoration time and verify the plan survives loss of the
      working storage copy. Use separate test copies for the drill.

15. **Connect monitoring to an operating response.**
    - [ ] Capture rank failures, stalled updates, nonfinite loss/gradients,
      sustained throughput, clocks/power/temperature and GPU errors.
    - [ ] Monitor free space, newest committed checkpoint, backup lag and
      validation/clipping trends.
    - [ ] Configure a notification/response mechanism and a human stop/recovery
      procedure. `production.status` currently reports local logs/disk only;
      it is not an unattended distributed watchdog or notification service.

16. **Run a production-cadence rehearsal.**
    - [ ] Exercise multiple checkpoint/evaluation/backup cycles on the chosen
      eight-GPU configuration, not just the benchmark's cadence-free window.
    - [ ] Check for memory growth, file-descriptor/data-loader issues, clock
      drift, growing backup queues and repeated runtime identity problems.
    - [ ] Re-run affected gates after any code, arithmetic, tuning or runtime
      change before freezing the campaign.

17. **Freeze the attainable training plan.**
    - [ ] Calculate the token budget from measured eight-rank rates and the
      remaining 5,760 GPU-hour allocation, including tests, cadence and recovery.
    - [ ] Choose the attainable final dense token horizon. The selected policy
      uses 1,006,632,960 warmup tokens and a final 10% cosine decay to 0.1× LR.
      Generate it for the actual manifest. Exact resume binds the schedule;
      later arbitrary horizon changes need an explicitly reviewed migration.
    - [ ] Freeze data mixture/order, candidate, batch/accumulation, monitoring,
      checkpoint policy, source, image and external configuration.
    - [ ] Regenerate/verify final staging, runtime and preflight receipts as
      required by their identities. Confirm the output directory is correct.
    - [ ] The human launches with `production/amd/train.sh` and observes the
      first production updates and operating cadence.

## Separate later work, not a prerequisite for the initial dense run

- [x] **QAT allocation planning:** 60/40 by tokens is the recommended starting
  policy, with 80/20 as a challenger; see [QAT_ALLOCATION.md](QAT_ALLOCATION.md)
  for phase-rate accounting and the 480 GPU-hour combined qualification reserve.
- [ ] **QAT execution:** implement and validate the AMD transition and ternary
  kernels, data-prefix/state carry, recovery quality and throughput. This is
  required before the full mixed campaign, though not before a bounded dense
  bring-up trial. Current AMD FP8 candidates deliberately reject ternary mode.
- [ ] **Multi-node scaling:** repeat network/RCCL, DDP equivalence, staging,
  sustained timing and recovery on each larger topology. Exact resume does
  not silently permit a world-size change.
- [ ] **Release/deployment:** final evaluation, generated-task protocols where
  required, model/data documentation and actual inference/export validation.
  Instruction tuning requires its own data, recipe and compute budget if a
  conversational model is the intended final product.

The main outstanding work is native runtime/kernel validation, real production
data and schedule verification, full-size distributed evidence, and durable
S3 recovery/monitoring integration. CPU checks do not substitute for these.
