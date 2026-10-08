# S0 — instrumented reference: complete-update timing boundary

Trial ID: **S0** (baseline instrumentation; not a speed candidate).
Reference/incumbent: winner `8bd552314470586f589d94bdaa6d7e8bc3c0c101`; framework commit `d3b3a083000bbe359b58ffdf6dd74859ed4b44d3`.
Campaign lock: **pending human launch inputs** (draft at external `stack-004/campaign/campaign.draft.json`).
Phase: dense and QAT (both profiled). Class: **E** (execution-only; no equation, mask, rounding, schedule or optimizer change). Atlas item: S0.

## Measured problem

No trustworthy full-update timing exists at the 1.5B target shape. The 800B-token user projection's provenance is unknown; the 200M proxy (143.5K tokens/s, H100 NVL) must not be extrapolated. The research questions S0 must answer: is the gap tensor-core service time, memory/recomputation, optimizer frequency, host/data gaps, an unexpected fallback, or a distributed assumption? The old architecture report's attribution (output-norm fusion, ~11 GiB peak delta) is a hypothesis, not measured causation.

## Mechanism

Factor the existing `run_training` update body in `terminus/train.py` into a `PreparedTraining.complete_update()` seam covering exactly the measurement-contract items: scheduled LR/phase/quantization setup and gradient clearing; next frozen batches with H2D transfer and all accumulation microsteps; forward, full loss, backward and gradient synchronization (`no_sync` rules unchanged); nonfinite detection, global clipping and all logical Muon/auxiliary Adam updates; FP8 cache invalidation and committed data/update counters. The callback returns the host-side valid-target count from manifest arithmetic; no `.item()` is added. Post-update cadence (logging, checkpoint, validation, stop events) stays outside the measured callback in `PreparedTraining.cadence()`. `run_training` keeps its signature and calls the same two pieces in the same order.

A new `scripts/s0_block.py` driver prepares the workload, runs a predeclared warmup-update count (identical across arms), resets the device memory high-water mark, then wraps `speed_research.measurement.measure_block` around N complete updates, validates finite logs/counter advance, and exports the raw block record (seconds, updates, valid_tokens, peak_device_bytes, contract/source/config hashes, checkpoint identities) outside Git.

The trainer's own resume contract already excludes `checkpoint_interval`, `validation_interval`, `validation_batches_per_language`, `log_interval`, `stability_log_interval`, `prefetch_batches` and `time_limit_seconds` as operational (non-semantic) fields. The driver may set cadence intervals beyond the block length so a measured block contains only complete updates; production cadence overhead is accounted separately in the end-to-end effective rate. No other training field is touched.

## Physics

Instrumentation adds zero GPU kernels and O(1) host work per update. S0 itself claims no speedup: it establishes the denominator. Expected trace content per contract §S0: six views (clean end-to-end updates; resident-input diagnostic; annotated profiler capture; tensor/storage lifetime ledger with allocator+device peaks; exact-shape GEMM/component timings; full state schema dump including scalar gains and V₁).

## Whole-update prediction

No time change. Falsifying outcome: instrumented vs original trainer produce different parameter/optimizer states, event semantics, or update arithmetic on identical inputs — then the instrumented baseline is fixed and re-versioned before any speed claim (stop condition).

## Evidence plan

Offline (this commit): CPU FP32 reference-backend equivalence — bitwise model/optimizer state and event-structure comparison against the winner commit's trainer on identical fixtures, including accumulation >1, resume, nonfinite-loss rejection, and the existing training/transition/architecture suites. Native (after human launch): reference/reference AB/BA null comparison to calibrate drift; block-length selection from baseline noise, then frozen; hardware/SKU/power/clock/ECC/MIG inventory; both timing profiles (production-local accumulation; fixed global update) with separate locks/labels.

## Spend

Offline now: no GPU. Native S0 estimate: to be reserved from the human-set research cap once supplied; includes environment bring-up, instrumented-reference qualification, six S0 views, and reference-state creation receipts. Stop condition: reference semantics fail to reproduce, or a fallback/measurement defect explains the gap — fix the baseline first.

## Risk

- Refactor drift altering trainer semantics → bitwise CPU equivalence plus full existing suite as gate; diff kept mechanical.
- Measurement profile confused with production cadence → separate labels and separate campaign locks per measurement contract §2.
- Source fingerprint changes from the instrumentation commit → expected; both arms of every comparison use identical instrumented source.
- No trained 1.5B checkpoints exist → deterministic externally-hashed initialization states used first and labeled initialization-only diagnostics; warmed-state creation is its own receipted spend inside the research budget.
