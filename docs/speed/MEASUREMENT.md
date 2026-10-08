# Measurement, correctness and promotion contract

This file is a specification for the human-launched native campaign. The offline tools implement provenance, arithmetic, paired speed screening and receipts; they do **not** implement a second training loop or automatically certify numerical/quality parity. Kimi's first native task is to integrate and qualify the small block timer in the winner's trainer. No GPU measurements in this document have been run.

## 1. Separate the reference, candidate and judge

Maintain three identities: immutable winning reference, instrumented reference (S0), and candidate. Keep a trusted checkout of the evaluator/checker outside agent-editable candidate worktrees. The reference configuration is fully materialized, not reconstructed from defaults that may change. Freeze state-dict parameter names/shapes, ties, optimizer groups/state schema, data cursor, tokenizer, training phase, schedule/global-batch and numerical tolerances.

Use `verify-anchor --checkout` before instrumentation. S0 may modify training execution to expose the block boundary; it must not alter equations, masks, rounding rules or update schedule. Compare instrumented versus original on identical batches/checkpoints, including multiple optimizer updates. Commit and tag that measured baseline. All later diffs are reviewed against it and the frozen semantic contract. A candidate's claim that its config is unchanged is insufficient: verify actual parameter coverage and gradients.

The class-E editable surface is the execution code in `terminus/model.py`, the relevant kernel/pointwise/loss/quantization modules, optimizer implementation, trainer scheduling and data loading **implementation**, plus focused tests and candidate execution settings (tiles, buffers, scheduling). `check-config` checks the materialized model fields; new purely operational settings can live in a separate committed execution config rather than extending/changing the protected model semantics. The immutable reference, prepared data, tokenizer, recipe, scorer, cost definition, quality samples and tolerances are outside this surface. Changes to protected measurement/recipe/numerical policy create a new campaign version with renewed baseline measurements. An environment change can be valuable but must be isolated and versioned; do not upgrade libraries incidentally during a candidate.

`campaign.lock.json` records all supplied controls plus hashes of the offline checker files. Keep its SHA-256 and trusted copy outside the candidate checkout. The hash detects drift relative to the recorded lock; it does not prevent an agent replacing both the checker and its hash. The external controller/verifier is the trust boundary. `quality_protocol` names the independently versioned protocol containing the native/error/quality gates; `quality_protocol_sha256` binds its contents, which belong in the external campaign archive.

## 2. Native S0 integration

Use the pinned `terminus.train` implementation and `PrefetchingTokenLoader`, not a new toy trainer. Factor or instrument its existing full-update block so one callback includes:

1. Correct scheduled LR/phase/quantization setup and gradient clearing.
2. Loading the next real, frozen batches, H2D transfer and all accumulation microsteps.
3. Forward, full loss, backward and applicable gradient synchronization.
4. Nonfinite detection, global clipping, all logical Muon and auxiliary Adam updates.
5. FP8 cache invalidation/versioning and committed data/update counters.

Call [measure_block](../../speed_research/measurement.py) around a predeclared number of these complete updates. Pass CUDA synchronization at the **start and end of the block**, not after every kernel. The callback returns a host-side integer valid-target count derived from the known batch manifest; no `.item()` should be added purely for the benchmark. Use actual trainer checks for finite loss/gradients; the timer deliberately does not invent a `finite=True` attestation. The independent exporter adds it only after validating those checks/logs. A skipped optimizer update must fail the block and preserve an incident, not become “fast training.”

Pseudocode (integration contract, not a launch command):

```python
workload = build_from_frozen_trainer_and_inputs(lock)
workload.warm_compile_and_initialize_optimizer()  # charged separately
workload.restore_frozen_model_optimizer_rng_and_cursor()
workload.run_fixed_warmup_updates()                # same count in both arms
reset_device_memory_high_water_mark()
raw_block = measure_block(
    workload.complete_update,
    updates=predeclared_updates,
    synchronize=torch.cuda.synchronize,
    read_memory=external_device_peak_bytes,
)
verify_finite_logs_and_final_state()
export_raw_block_with_hashes_and_complete_billed_duration(raw_block)
```

Warmup must not give arms different learning exposure. If graph capture mutates weights or optimizer buffers, restore **all** state before fixed warmup; a model-only reload is inadequate. Compile caches may be reused, training inputs must advance normally. Native preflight/correctness uses adversarial and real tensors; performance uses actual staged inputs at the full target shape. Synthetic data is allowed for component diagnostics, never as a quality or production throughput claim.

Start from deterministic, externally hashed initial target states if no trained 1.5B checkpoint exists yet; label those as initialization-only diagnostics. Creating warmed dense/QAT reference states on real data is part of the authorized research budget and receives its own receipt. Final numerical/quality qualification needs representative trained states; absence of late-stage 1.5B checkpoints remains an explicit transfer risk, not a reason to imply that random-initialization parity covers them.

Measure both timing profiles: (a) production-local accumulation on one GPU to estimate future local compute; (b) fixed global update on one GPU for quality equivalence. They have different optimizer amortization. Use separate locks and labels. `global_tokens_per_update` in the one-GPU schema is the complete measured research update, and must match the recorded profile. The final quality contract fixes production global update size and rank partition. Do not multiply a quality profile's tokens/s by 32 without checking the production-local profile.

## 3. What to record

Every trial has a unique ID, immutable source commit, diff/parent, proposal, class E/N, controller timestamps, environment/container/compiler and library revisions, GPU UUID/SKU/power/clock/ECC/MIG, topology and phase. Record the exact model/optimizer/data/tokenizer/evaluation/lock identities. Include initial and final checkpoint hashes, sample cursor ranges, actual valid-target/update counters and accumulation schedule.

Every timing block records elapsed host wall time with synchronized endpoints, update count, unique valid-target count, separate cold/setup/warmup time, GPU/VM billed duration, memory high-water data and relevant clocks/power/thermal state. Record torch allocated/reserved **and** device-wide peak/context/non-torch allocations. A final `nvidia-smi` snapshot is not a peak. Prefer external sampled device telemetry plus allocator high-water evidence; account for sampling uncertainty and reserve headroom. Do not read synchronized memory telemetry every microstep just to collect it.

Numerator rules:

- Count loss-bearing targets in the frozen stream once per authorized exposure. Do not count padding, ignored labels, rematerialization, profiler replays or repeated benchmark blocks as production training progress.
- In a measured research block, repeated runs may revisit the same controlled data for comparison, but that count is a *throughput measurement*, not tokens added to the production corpus exposure. The production counter uses actual cursor progress and successful updates.
- For multiple ranks, aggregate distinct sharded targets once; divide by synchronized wall time for the job/slowest rank. Never sum per-rank tokens/s and then multiply by GPU count again.
- Report training running rate, full-update rate and end-to-end effective rate with distinct labels. Global batch changes, optimizer omissions or switching loss reduction invalidate comparison.

For profiler captures: include kernel symbols, NVTX regions, time/rank/stream, HBM bytes, tensor-core utilization, eligible/active warps, register/spill/shared-memory use, launch count, CPU gaps and realized overlap where available. Preserve raw traces outside Git; do not declare a profiler's aggregate percent a physical utilization measure without explaining the counter.

## 4. Promotion ladder and budget

| Gate | Purpose | Evidence required |
| --- | --- | --- |
| S0 | Correct baseline/instruments | Reproduced source/shape/updates, null timing test, hardware and memory inventory |
| S1 | Component correctness/economics | Reference/native forward and backward; complete operand-preparation costs; Amdahl ceiling |
| S2 | Whole-update speed screen | At least six independent AB/BA pairs, ≥20 complete updates and ≥60 seconds per block; fixed workload |
| S3 | Fresh speed confirmation and quality | New, previously unused paired blocks; matched-token learning/held-out comparisons under frozen tolerances |
| S4 | Candidate composition and soak | Both phases, transition, long memory stability, resume correctness, full production features |
| S5 | 32-GPU production acceptance | Actual topology, all-rank numerical/input/NCCL checks and sustained production-rate proof |

Durations are thresholds, not six statistically independent samples taken from six neighboring iterations in one run. Select block length using the S0 baseline, then freeze update counts large enough for both arms to clear the minimum elapsed time. If a much faster candidate makes the block too short, repeat the **pair** at a newly predeclared common count; do not relabel old short data. Each pair starts from equivalent model/optimizer/RNG/data state; independent pairs use predeclared seeds or checkpoints and independently initialized processes/state. Alternate randomized balanced AB/BA order on the same GPU. Stabilize clocks/thermal behavior without claiming a higher power cap than production will supply.

The default screen policy requires ≥2% speedup at the one-sided 95% paired bootstrap lower bound, ≤3% baseline range drift, and a device peak ≤74,000,000,000 bytes. These are initial engineering screening choices, not universal laws. Calibrate noise and adjust the policy **before** candidate trials. Near ties or intervals crossing the threshold are unresolved. Smaller real improvements can be kept for clarity/simplicity or later composition only with an explicit rationale and fresh evidence; don't claim they passed a threshold they failed.

The bootstrap resamples **paired log-rate ratios** across independent blocks. With six pairs it is a screen, not a powerful guarantee. Search selects lucky winners, so S3 must use fresh blocks and predeclared endpoints. Do not repeatedly peek until significance appears. Report all attempted variants and confirm finalists against both their immediate incumbent and the original reference, avoiding accumulated noisy gains. Avoid combining individual speedup percentages multiplicatively.

Before spending, the external supervisor reserves worst-case GPU/VM-on time for the next trial, including compilation, evaluation, diagnostics, retries and archive recovery. Pending reservations count against the cap. Record every actual billable interval, including controller-induced GPU idle time. A timeout kills the entire workload process group via the human's worker policy, preserves logs and finalizes the receipt. Do not confuse the receipt utility's time fields with an enforced billing limit: the CLI has no paid runner. Use the human's approved lifecycle controller with the new systems policy, not the old architecture policy unchanged.

## 5. Numerical correctness: observable obligations

Calibrate class-E tolerances in S0 against the original native path, a higher-precision reference and baseline repeatability. Freeze **per-operation/per-tensor** absolute and relative error limits, gradient direction/norm checks, and short-update drift. Relative error alone is unstable near zero; include absolute error and scale-aware norms. For finite differences use small FP64 reference operations; FP8 straight-through gradients need comparison to the specified estimator, not derivatives of the rounding function. A forward-only test cannot qualify a training kernel.

Mandatory coverage when relevant:

- Layer dimensions, causal mask, local window inclusive boundary, GQA head mapping, RoPE phase/convention and learned scale gradients.
- All 25 layers present and used; canonical parameter count/name/shape manifest unchanged, no frozen/missing/zeroed gradient path hidden by a nominal count. If physical packing changes, require an invertible mapping to the original logical parameters, gradients and optimizer state and validate round-trip checkpoints.
- Every Peri-LN channel gain, Q/K gain, both value lambdas and layer-0 V projection; negative/zero/nonunit coefficients and channels with extreme values.
- Tied embedding storage and sum of input-lookup plus output-classifier gradients; full vocabulary, ignored labels, exact target shifts and all-ignored reference behavior.
- Forward/backward on native H100, current FP8 block scaling/rounding/layout, weight/dgrad/wgrad orientations and data-dependent cache invalidation.
- Accumulation 1 and >1, changes of microbatch preserving global update, last partial batch/update, clipping and scheduled LR changes, consecutive updates and repeated forward before backward.
- Checkpoint on/off, hooks on/off, graph warmup/replay, compile path, cached/source tensor lifetimes and asynchronous stream races.
- Muon logical slices, iteration/math/dtype, all state buffers and Adam steps/decay; compare updated parameters **and optimizer state**, not only the scalar loss.
- Injected nonfinite/OOM/stop paths leave no successful counter/receipt for an unapplied update. Cache reload and dense→QAT transition cannot reuse stale operand data.

Class N has a documented changed representation, scale/error budget and expected sensitive tensors. It passes both tensor checks and S3 learning noninferiority; a single permissive `allclose` threshold is insufficient. If an E implementation exceeds its frozen numerical envelope, either fix it or propose a newly versioned N experiment. Do not reclassify it after a quality failure to excuse the result.

## 6. Quality: same computation, preserved learning

For class E, numerical/optimizer parity is the primary defense, supplemented by short matched-token trajectories at realistic target width and relevant phases. Class N needs longer paired continuations at multiple checkpoints/seeds. The 200M winner can help debug math cheaply; it cannot prove target-width speed, activation ranges or long-run numerical stability. Use target-width data before promotion.

Freeze separate development and untouched confirmation data. Cover the actual intended production mixture (English, Indic languages, code, math as applicable), rare tokens and position buckets. Evaluate with the protected reference evaluator/precision to avoid a candidate-specific loss kernel concealing its own error. Keep contexts/masks/tokenizer/sequence lengths and all loss-bearing tokens identical. Do not tune the next candidate on the final confirmation set.

Predeclare a one-sided noninferiority margin in nats/token by domain and aggregate, anchored in repeatability and acceptable production loss. A planning starting point might be **0.005 nats/token aggregate**, but it is not universal and is **not** a ready-to-use default: the external quality protocol must define and justify margins before any candidate results. Require the upper confidence bound of `candidate − reference` to stay below each material margin, using paired documents/checkpoints/seeds rather than treating correlated tokens as independent. “No statistically significant loss increase” is not evidence of equivalence.

Compare learning curves at matched tokens and complete scheduled update exposure, not at matched five-minute wall time. Report final and trajectory loss, spikes/nonfinite incidents, clipping, layer residual/gain/value statistics and FP8 saturation. A changed numerical path may look fine near initialization and fail after weight/gain dynamic range grows. Include available early/late dense and recovered QAT checkpoints; report unavailable long-horizon evidence honestly.

Dense/QAT needs three checks: dense continuation, exact transition/conversion behavior and QAT recovery/continuation with the fixed quantizer/STE/schedule. No changing QAT fraction or shortening the hard phase to meet the budget. The stack's historical conversion shock was larger than baseline; preserve its recovery and residual stability, not just a final short scalar loss. Joint precision changes need their own combined quality confirmation even if each passed separately.

## 7. Soak, failure recovery and production acceptance

For a finalist, run a prebudgeted sustained test long enough to cover representative thermal steady state, checkpoint/evaluation cadence, data refill and optimizer-state behavior (for example 30–60 minutes per phase, subject to the separate research cap). This is a proposed gate, not work already authorized. Track rising allocated/reserved/device memory, changing rate, cache churn, power throttling and host RSS. Include a stop/checkpoint/restart inside a matched run and verify uninterrupted versus resumed states/cursors/next updates within the frozen numerical policy.

Prepare a rollback package with the instrumented reference and last qualified incumbent, environment image, dispatch config, state conversion (if any, independently checked) and exact restart command. Durable snapshots contain immutable state and are independently hash-verified after transfer. Failure to archive is a live operational incident; preserve the last good checkpoint and follow the owner's recovery/shutdown policy.

S5 can only happen after the owner makes 32 GPUs available. Its first-hour plan is: hardware/fabric identity and rank mapping; input uniqueness and normalized gradient test; one known-update equivalence check against the fixed global contract; small real NCCL test; short complete-update baseline/candidate with the production microbatch; then sustained accepted dense and QAT profiles including intended logging/evaluation/checkpoint features. Reserve its actual cost inside the 6,000-hour production allocation if it occurs there. A 32-GPU hour costs 32 hours of the allowance.

Before committing the full production horizon, recompute remaining token capacity from actual consumed hours and phase-specific sustained rates, with an explicit contingency reserve. Use conservative rates (fresh-block lower confidence bounds or demonstrated sustained lower quantiles with a stated estimator), not the best minute. Classify outcomes precisely: `component_win`, `screen_pass`, `quality_qualified_single_gpu`, `projected_32_gpu_feasible`, `measured_32_gpu_qualified`. The supplied analyzer deliberately outputs only `screen_pass`/`unresolved`/`reject`/`remeasure` and always `production_ready=false`.

## 8. External evidence schema

No measured example is stored in Git. Construct `comparison.json` from the trusted exporter as follows; the angle-bracket placeholders below are documentation, not valid trial data:

```json
{
  "schema": 1,
  "lock_sha256": "<64 hex chars from campaign lock>",
  "stage": "full_update",
  "phase": "dense",
  "world_size": 1,
  "native_sm90": true,
  "hardware": "<exact campaign research_hardware string>",
  "baseline_commit": "<40 hex chars>",
  "candidate_commit": "<40 different hex chars>",
  "pairs": [
    {
      "order": "AB",
      "baseline": {
        "run_id": "<unique ID>",
        "seconds": "<positive measured seconds>",
        "valid_tokens": "<integer>",
        "updates": "<integer>",
        "peak_device_bytes": "<integer peak>",
        "billed_gpu_seconds": "<complete GPU-on charge, at least measured seconds>",
        "finite": true,
        "completed": true
      },
      "candidate": "<same record fields; equal updates and valid_tokens>"
    }
  ]
}
```

Provide at least six independent pairs with balanced order; all numeric fields are actual JSON numbers. Archive the raw logs, tensor/quality reports, hardware telemetry, producer/environment identity and source/checkpoint/input hashes alongside this summary, with a file-hash receipt. The comparison tool checks summary consistency and statistics, not authenticity. It does not consume or automatically approve the separate numerical/quality gates.

Each ledger receipt includes `trial_id`, `billed_gpu_hours`, `outcome` (`screen_pass`, `unresolved`, `reject`, `crash`, `timeout`, or `remeasure`), commit/lock identities, stage/phase, raw artifact manifest hash and the written decision/review. The append utility rejects duplicate IDs and broken prior links, locks the file and fsyncs each append. Archive every new chain head outside the mutable local directory; an attacker who can rewrite the whole chain can recompute unanchored hashes. Corrections are new receipts/IDs referring to earlier ones, never in-place edits. Crashes with no timing pairs still get a receipt and consume budget.
