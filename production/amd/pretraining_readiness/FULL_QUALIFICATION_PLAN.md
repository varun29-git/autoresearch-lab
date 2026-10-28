# Full qualification and tuning plan: 1.446B on eight MI325X GPUs

Engineering plan, 2026-10-28. This is the governing qualification proposal for
the user's request to treat the model as having **no prior hyperparameter
qualification**. It is a plan and implementation backlog, not a record of
passed GPU tests. The human starts and stops paid campaigns.

This plan supersedes the earlier policy of testing the old rates at 2M first
and considering alternatives only after a failure. It also replaces the
earlier 320/480 GPU-hour planning envelopes with the explicit envelope below.
Existing source defaults and production validators still enforce the old
recipe; changing this document does not implement the new qualification mode.

The deliverable is a reproducible, measured recipe for this exact model and
node, with a funded ending and recoverable checkpoints. Historical HPO is
background evidence, never an automatic promotion. No experiment can prove
trillion-token quality before those tokens are trained; the plan manages that
uncertainty through controls, longer finalists, decay probes and production
decision points.

## Navigation

1. [Decisions and scope](#1-decisions-and-scope)
2. [One GPU versus eight](#2-what-one-gpu-and-eight-gpus-can-establish)
3. [Budget and token feasibility](#3-budget-and-token-feasibility)
4. [Engineering before rental](#4-engineering-before-paid-tuning)
5. [Data and evaluation](#5-data-and-evaluation-contract)
6. [Hardware and numerical qualification](#6-hardware-and-numerical-qualification)
7. [Execution optimization](#7-execution-optimization)
8. [Dense hyperparameter experiments](#8-dense-hyperparameter-experiments)
9. [Long-horizon and schedule decisions](#9-long-horizon-and-schedule-decisions)
10. [Ternary QAT qualification](#10-ternary-qat-qualification)
11. [Storage and recovery](#11-storage-checkpoints-and-recovery)
12. [Monitoring and stop rules](#12-monitoring-and-stop-rules)
13. [Production release and operation](#13-production-release-and-operation)
14. [Execution order and artifacts](#14-execution-order-and-artifacts)
15. [Research basis](#15-research-basis-and-limits)

## 1. Decisions and scope

### Fixed model and numerical semantics

| Item | Contract |
| --- | --- |
| Actual trainable parameters | **1,446,478,126**, the repository's approximately 1.5B model |
| Geometry | 24 layers; width 2304; FFN 5760; 18 Q / 6 KV heads; head dimension 128 |
| Tokenization | Frozen 65,536 vocabulary and exact tokenizer digest; 4096-token sequences |
| Attention | Local window 512 including the current token; global layers 6, 12, 18, 24 |
| Model details | Peri-LN, learned Q/K RMS gains, learned value residual, tied embedding/readout |
| Hidden optimizer algorithm | Spectral Muon, logical Q/K/V and gate/up slices, pinned NS5/Nesterov semantics |
| Auxiliary optimizer | AdamW with separately named embedding/head, norm, Q/K and value groups |
| Dense numeric baseline | BF16 residuals/head, FP32 master weights/accumulated gradients/state; independently qualified AMD FP8 projections |
| Hardware | One complete eight-MI325X node, verified usable memory and xGMI topology |
| Allocation | **5,760 aggregate allocated GPU-hours**; eight cards for 720 hours = 30 days |

The optimizer *algorithms* and model are fixed to keep the search affordable.
Learning rates, global batch, warmup, matrix decay and selected small-group
rates are now qualification variables. A different architecture, tokenizer,
Muon polynomial, loss objective or quantizer gradient is a new experiment
outside this tuning grid. Do not slip such changes into a speed optimization.

### Immediate recommendations

- Use eight ranks for every experiment that can select production
  hyperparameters. This removes a world-size change between tuning and the
  accepted configuration and allows eight-rank recovery to be exercised.
- Use individual GPUs for kernel development, isolated arithmetic tests and
  initial full-model fit. Repeat device health checks on every card.
- Search the dense rates on the actual 1.446B model. No additional small-model
  proxy campaign is required by this plan.
- Start execution work at 2,097,152 global tokens/update, but compare both
  **1,048,576 and 2,097,152** in the mandatory learning phase.
- Preserve activations initially. Choose microbatch by measured full-update
  throughput and memory margin. Activation checkpointing is conditional work.
- Qualify the FP8 implementation against an independent oracle and BF16.
  Finite training loss by itself is insufficient.
- Keep the intended final ternary model in scope. Native QAT is a launch
  blocker for a mixed campaign. The old 60/40 token split is an initial
  allocation hypothesis, not a fixed research result.
- Use staged experiments with explicit update/time caps. Launching a pilot
  never authorizes running to the end of its large manifest.

## 2. What one GPU and eight GPUs can establish

One MI325X can train this model **if the full update fits at the selected
microbatch**; actual fit remains unmeasured. DDP replicates model and optimizer
state, so eight-way DDP does not divide their memory by eight.

For 4096-token sequences and microbatch 16:

| GPUs in the job | Accumulation at 1,048,576 tokens | Accumulation at 2,097,152 tokens |
| ---: | ---: | ---: |
| 1 | 16 | 32 |
| 2 | 8 | 16 |
| 4 | 4 | 8 |
| 8 | 2 | 4 |

`global_tokens = ranks × microbatch × 4096 × accumulation`.
Microbatch 8 doubles these accumulation counts. This preserves the intended
averaged gradient when examples and loss normalization agree. Floating-point
reduction order, FP8 rounding, caches and runtime behavior still need testing.

| Work | One GPU useful? | Eight-GPU requirement / policy |
| --- | --- | --- |
| CPU data audits, config generation, accounting, evaluator tests | No GPU needed | Finish before rental where possible |
| Image build/dependency resolution | Linux CPU host usually sufficient for build work | Native import/JIT execution still needs the actual GPU/runtime |
| FNUZ quantizer and GEMM forward/dgrad/wgrad correctness | Yes | Repeat health/smoke on each device; qualify composed DDP updates separately |
| Attention masks, GQA, Peri-LN, value residual and CE kernels | Yes | Full distributed trajectory remains required |
| Full-model optimizer-state creation and conservative memory fit | Yes | DDP buckets, all-rank workspaces and cadence peaks must fit on all eight |
| Per-shape tuning and profiler investigation | Yes | Promote an optimization only after complete eight-rank timing and correctness |
| LR screening with accumulation | Mathematically useful after equivalence checks | **This plan uses eight ranks for selection**, rather than relying on a one-rank winner |
| Global batch / LR / schedule winner | One rank cannot certify the deployed system | Eight ranks, real data, intended precision, matched evaluation |
| RCCL, xGMI, bucket overlap, slowest-rank tail | No | Eight ranks on the actual node |
| Distributed checkpoint, rank loss, S3 restore and restart | No | Eight ranks with the intended storage/network |
| QAT arithmetic / STE debugging | Yes | Conversion, recovery, learning and export checks on the selected full setup |
| Allocation-level throughput and token forecast | No | Measured eight-rank, cadence-inclusive rates |

Do not split the node into eight independent one-GPU HPO jobs by default.
That can be an efficient future scheduling option, but requires separately
proved equivalence and gives checkpoints that the current exact-resume
contract cannot resume at eight ranks. A single-card debugging job on an
already rented eight-card node is charged for **all eight allocated cards**.
Independent per-device tests may run concurrently once shared CPU/I/O demands
are controlled; they are not distributed-training evidence.

## 3. Budget and token feasibility

### Qualification envelope

Recommend reserving **960 GPU-hours**, or five eight-GPU node-days, for this
from-scratch qualification. This is a spending ceiling with stage budgets,
not a prediction that all stages will need their full allocation.

| Work package | GPU-hours | Eight-GPU node-hours | Includes |
| --- | ---: | ---: | --- |
| A: device/runtime/native arithmetic | 64 | 8 | Per-device checks, kernels, initial full-size updates and initial repair allowance |
| B: DDP/precision/performance | 80 | 10 | One/eight equivalence, BF16 control, FP8 trajectory, microbatch/backend finalists |
| C: dense learning experiments | 512 | 64 | LR grid, small groups/decay, batch interaction, longer finalists, second seed, decay probes |
| D: QAT implementation acceptance/learning | 160 | 20 | Native full-model conversion and bounded QAT/control trajectories |
| E: operating acceptance and soak | 48 | 6 | Fault/restore drills, cadence, sustained full-node operation |
| F: contingency | 96 | 12 | Measured overruns, one ambiguous comparison or one corrective experiment |
| **Total** | **960** | **120** | Leaves 4,800 GPU-hours / 25 node-days if fully spent |

Build/debug on CPU hosts before this envelope begins where possible. Native
bugs can still consume it. If a correctness blocker exhausts its stage, stop
that sequence, repair it and explicitly reallocate within the total ceiling.
Do not spend indefinitely because a process happens to be running.

Before each trial, the supervisor must check:

1. Allocated time already spent, including idle/build/evaluation time.
2. Conservative cost to the next declared complete checkpoint/milestone.
3. Remaining mandatory controls, restore work and QAT qualification.
4. Remaining funded production time and its revised token forecast.

Do not double-count the earlier 480-hour reserve; 960 replaces it for this
proposal. The existing phase calculator defaults to 480, so explicitly pass
`--reserve-gpu-hours 960` until the new campaign tooling records actual costs.
Any separate pre-month qualification rental would preserve more production
time, but no additional allocation is assumed here.

### Dense experiment volume and affordability

Let `U = 2,013,265,920 tokens`, or 960 updates at the 2M batch. This is an
accounting unit, not a claim that its precise value is statistically special.
The complete dense sequence in section 8 has a conservative planned volume
of **40U = 80.5306368B training-token exposures**, counting repeated data
across candidates. This is not 80.5B tokens learned by one final checkpoint.

| Measured tuning tokens/s per allocated GPU, including trial overhead | Cost of 80.53B exposures |
| ---: | ---: |
| 20,000 | 1,118.5 GPU-hours: does not fit this envelope |
| 30,000 | 745.7 GPU-hours: does not fit work package C plus contingency |
| 40,000 | 559.2 GPU-hours: needs 47.2 hours from contingency |
| 45,000 | 497.1 GPU-hours |
| 50,000 | 447.4 GPU-hours |
| 60,000 | 372.8 GPU-hours |

These are sensitivity calculations, **not expected MI325X rates**. Measure
the first full trial's allocated-time rate, not just a timed optimizer block.
At 30k or below, reprice a smaller search before launching the grid: reduce
exploratory breadth or finalist duration in a recorded plan revision, or
obtain a separately funded qualification allocation. Keep numerical checks,
the BF16 control, at least two batch/LR alternatives, recovery and a longer
learning comparison mandatory. If no informative comparison fits, the
configuration is not ready to consume the remaining production budget.

### Production token forecast

Use whole-update rates measured on eight ranks, including communication and
optimizer work. If checkpoint/evaluation time is excluded from these rates,
apply a separately measured duty factor once. If rates already include all
operational time, use duty factor 1 instead of discounting twice.

For dense fraction `f`, dense rate `rd`, QAT rate `rq`, remaining allocated
GPU-hours `H` and duty factor `d`:

```text
effective_productive_rate = 1 / (f/rd + (1-f)/rq)
funded_tokens = H * 3600 * d * effective_productive_rate
```

Illustration only: full 960-hour reserve, remaining `H=4800`, `d=0.9`, and
60/40 dense/QAT tokens:

| Dense / QAT tokens/s/GPU | Funded combined tokens |
| --- | ---: |
| 40k / 30k | 549B |
| 50k / 35k | 664B |
| 60k / 45k | 823B |
| 70k / 55k | 982B |

One trillion tokens in this example requires an effective productive rate
of about **64,300 tokens/s/GPU** across both phases. A 1T target is therefore
conditional, even with a good optimization result. A shorter well-trained,
properly decayed model can be the funded outcome. Increasing the manifest's
token count does not increase available compute.

Freeze the final horizon after the native measurements and pilot costs are
known. Round phase boundaries to complete updates. Count a reusable pilot
prefix toward the final model's tokens only when its checkpoint, data,
optimizer, schedule, source, runtime and world-size lineage permit it; count
its GPU-hours exactly once regardless.

## 4. Engineering before paid tuning

The current code is not a turnkey implementation of this new search. These
are explicit, reviewable work items; a document or a generated JSON file is
not evidence that they are finished.

| Priority / work item | Existing starting point | Required implementation and acceptance |
| --- | --- | --- |
| P0: qualification launcher and candidate registry | `production/amd/train.py`, `recipe.py`, `profiles.py` | Add a bounded qualification mode that pins the model/optimizer algorithms but permits a versioned allowlist of HP values. Require an external candidate spec, unique run ID, update limit, wall-time limit and allocated-hour ledger. Preserve strict production validation. Tests must reject model changes, missing bounds, undeclared LR changes, stale receipts and accidental use of smoke schedules. |
| P0: dense fork/continuation tooling | `terminus/transition.py` currently converts dense to ternary | Implement a distinct dense-to-dense fork for diagnostic LR/schedule/decay branches. Preserve weights, compatible moments, counters, per-rank RNG and proven data prefix; record allowed changes and exact old/new schedules. Do not edit a checkpoint contract or masquerade as QAT. Prove parity for a no-change fork and rejection of hidden data/shape/world-size changes. |
| P0: full-size DDP comparison harness | CPU DDP tests and shared `PreparedTraining.complete_update()` | Capture controlled full-update losses, gradients before/after clipping, logical matrix deltas and moments on 1 and 8 ranks. Use known identical global examples and initialization. Test partial tails and save/restore. Reports stay outside Git. |
| P0: numerical telemetry and watchdog | `terminus/stability.py`, train incidents, `production.status` | Add role-resolved full-tensor or well-defined block update statistics, native FP8 scale/saturation/zero diagnostics, rank health and budget/backup checks. Existing probes are sampled, use the first microbatch and recompute a quantization diagnostic; they do not cover every native cached operand or every rank. Measure overhead. |
| P0: live durable backup/retention | Atomic local checkpoints; stopped filesystem archive tool | Implement committed-pair upload, content verification, bounded queues, protected lineage and live-safe retention; test S3 restore and failed uploads. Existing ingestion and stopped retention are insufficient for unattended live backups. |
| P0: native AMD ternary path | H100 ternary semantics; AMD FNUZ GEMM adapters | Port effective ternary projection codes/scales, identity STE semantics and cache invalidation, then extend AMD model/launch validation for an explicit QAT candidate. Do not merely remove the current rejection. Test zero/tie cases, scale gradients, transposes, latent weights, full updates and export. |
| P0: generated task evaluation | `production.evaluate` currently supports reference NLL | Add pinned base-model multiple-choice, math-answer and code execution protocols with complete item coverage, bounded generation and a sandboxed code evaluator. Separate development and final tasks. Teacher-forced NLL must not be labeled accuracy or pass@1. |
| P1: performance candidates | Existing TE/AITER/Triton, CK, chunk and bucket options | Benchmark exposed bottlenecks first. Add new fusion/checkpointing/graphs only against the numerical and complete-update criteria in section 7. |
| P1: machine-readable release record | Existing runtime/staging/preflight identities | Bind selected HPs, data, evaluations, image, source, memory, throughput, QAT, recovery and budget evidence into one external release record. Production launch must reject a missing/failed mandatory gate. |

The new launcher should have separate namespaces for candidate-generation,
qualification execution and production execution. A convenient `--override`
must not bypass architecture, dtype, data, runtime or budget invariants.
All native repairs and source changes require fresh affected receipts.

The supervisor may enforce the operator-configured limits of a human-launched
campaign. It must not rent hardware, start an unregistered trial, silently
increase a budget, or blindly retry a deterministic numerical failure.

## 5. Data and evaluation contract

### Data readiness before HPO

1. Inventory actual sources, licenses/provenance, usable unique tokens,
   languages, code/math domains and planned repetitions. Freeze the initial
   production mixture and its phase changes outside Git. Do not substitute
   an English-only stream for the intended multilingual/domain workload.
2. Validate tokenizer digest, vocabulary bounds, special IDs, encoding,
   `uint16` layout, byte order, sequence shift and valid-target counting.
   Test loss at the first/last shard boundary and on partial final updates.
3. Audit exact and near duplicates; source contamination; train/development/
   final overlap; repeated boilerplate; malformed text; language-ID failures;
   and long repeated n-grams. Inspect representative decoded examples in
   every scope. Retain the actual audit methods and exclusion counts.
4. Include legitimate code/math/Indic edge cases in filtering checks. A
   repetition rule that removes useful tables or code is not automatically
   beneficial. Study pathological repeated sequences before freezing the
   corpus; do not silently mask or skip loss-bearing data during HPO.
5. Freeze document-boundary/packing behavior. If packed examples allow
   cross-document attention, declare that policy; changing to document masks
   later changes the training experiment. Check EOS and position behavior.
6. Validate the ordered global stream across ranks and microbatch regrouping,
   then make staged snapshots read-only. Hashes prove identity, not quality.
7. Stage a representative pilot prefix and sufficient future data for every
   declared trial. Build the final corpus before any run intended to be an
   exact production prefix. The current streaming contract must not be
   assumed to support a later arbitrary manifest replacement.

Dataset mixture percentages are not invented by this plan. They must come
from the product's language/domain priorities and available vetted corpora,
be frozen before comparisons, and be identical for paired HP candidates.
Changing the data mix and LR simultaneously prevents a clean LR conclusion.

### Evaluation layers

| Layer | Evaluation | Use |
| --- | --- | --- |
| Every update / sample cadence | Global token-mean loss, gradient norm, clipping, LR, step/data time; sampled role/FP8/attention telemetry | Detect broken execution and instability |
| About every 0.5B pilot tokens, plus warmup end and every rung | At least 1,024 fixed 4096-token development sequences per declared scope, with source/document identities | Compare per-domain learning and guard against aggregate improvements hiding damage |
| 4B and longer finalists | Pinned multiple-choice and targeted math/code/Indic development tasks; small human-readable generation pack | Check whether lower reference loss is translating into useful behavior |
| Decay/QAT phase decisions | Same held-outs and tasks for parent, dense control, candidate QAT and exported model | Quantify conversion loss, recovery and final tradeoff |
| Final release | Untouched final sets and inference/export verification | Report actual task quality without tuning on final results |

Use a predeclared production-weighted NLL aggregate plus separate mandatory
language/domain guardrails. Keep weights fixed. Resample independent
documents/examples, not individual correlated tokens, for paired confidence
intervals; retain grouping metadata when evaluation sequences are contiguous.
Record uncertainty and correct simultaneous per-scope claims, or mark them
as exploratory. Inspect at least two consecutive post-warmup evaluations.

The existing **0.02 nats/token** upper paired-NLL degradation guardrail is a
proposed precision/execution non-inferiority margin, not an optimizer optimum.
For choosing among learning candidates, report the signed effect, confidence
interval, slope, stability and actual allocated cost. If finalists are
statistically indistinguishable, prefer the robust interior of a stable LR
region and simpler settings. A difference smaller than evaluation noise must
not become a scientific claim of superiority.

At very early training, generated task accuracy may be near chance. It cannot
replace learning curves at that stage. Conversely, reference-answer NLL is
not a substitute for generated accuracy later. Compare external base models
using identical tasks/protocols; raw token NLL across different tokenizers is
not directly comparable. MiniCPM5's RL/OPD scores are not a pretraining target
measured under our recipe.

The primary reporting axes are quality at **matched training tokens** and
quality at **matched allocated GPU-hours**. A larger batch can win throughput
and lose sample efficiency. Count all trial setup, validation and checkpoint
time in the second axis.

For the generated development suite, use pinned base-model protocols covering
English commonsense/knowledge (for example ARC/PIQA/HellaSwag and a declared
MMLU subset), exact-answer math (GSM8K-style), code (HumanEval/MBPP-style in an
isolated evaluator), and the required Indic languages (MILU-style coverage).
Freeze prompts, shots, option normalization, generation limits, sampling,
answer parsing and item IDs before comparisons. Reserve disjoint final items
or task families. These are proposed benchmark families; missing data or a
missing evaluator is an open gate, not a silently skipped score.

## 6. Hardware and numerical qualification

### A1. Inventory and device health

Record all eight device IDs, gfx942 target, usable HBM, xGMI/peer-access matrix,
NUMA placement, CPU/RAM, driver/firmware, configured power limits, clocks,
temperatures, throttling and RAS/ECC state. Confirm the actual allocation is
MI325X, not a different SKU, partition or topology. Test each card; one healthy
card does not clear the node. Preserve vendor-approved power settings while
establishing the baseline. Clock/power changes are separate measured trials.

Record sustained host-storage and provider-to-S3 transfer behavior. Verify
container device access, shared memory, CPU affinity and that another tenant
or process is not consuming the assigned GPUs. Save the inventory externally.

### A2. Runtime and kernel arithmetic

Build the pinned image, resolve dependency/API failures, run `pip check`,
warm required JIT shapes and freeze the resulting image/source/cache identities.
Use the selected ROCm torch/Triton pair, not an untested package upgrade during
the campaign. Existing `Dockerfile` and build scripts are starting artifacts;
the resulting Linux runtime has not been certified here.

For every backend proposed for use:

- Validate E4M3 FNUZ interpretation, maximum finite value, scales, zero blocks,
  tiny values, outliers, padding and explicit nonfinite propagation.
- Validate forward, dgrad and **FP32 wgrad** against independently quantized
  reference operands. Cover all logical Q/K/V/O/gate/up/down shapes, real
  strides, transpose paths and sequence/microbatch sizes.
- Cover tiny upstream gradients around the existing `2^-24` case and long
  wgrad reductions. A near-zero absolute-error exemption must not permit an
  all-zero gradient to pass.
- Check activation/gradient requantization after transpose, 1x128 versus
  128x128 scale layout, and every supported weight-cache invalidation path.
- Check causal and local-window boundaries at 511/512/513 positions,
  global layers, GQA head mapping, learned Q/K gains, RoPE and cached offsets.
- Check Peri-LN/value-residual gradients, packed logical matrix boundaries,
  tied input/head gradient addition, CE target shifting and ignored tokens.
- Check Muon NS5, momentum/Nesterov, per-logical-matrix aspect factors,
  independent weight decay, all Adam groups and changing scheduled LR.

Use the existing preflight thresholds as written, including its strict FP8
relative-error requirement. Save individual case errors. A failed case is
investigated and fixed; thresholds are not widened to promote a backend.

### A3. Full-model update and memory

Start with microbatch 4 or 8 and enough accumulation for the target global
batch. Instantiate the entire model, FP32 gradients, optimizer moments and
FP8 caches. Exercise at least several warm complete updates, evaluation and
checkpoint writing. A forward-only or optimizer-free fit is not a fit result.

Measure allocated, reserved and device-level usage on every GPU. Retain at
least **15% of actual usable HBM** at the worst peak, not 15% of nominal
marketing memory. Record workspace/JIT/allocator variability and host-memory
peaks. DDP qualification repeats this measurement with buckets and collectives.

### B1. One-rank versus eight-rank equivalence

Use a frozen initialization and optimizer state and the exact same global
examples. Compare:

1. Forward loss sums and valid-token counts.
2. Gradients before clipping, clipping coefficient and gradients after clipping.
3. Each logical Muon update and each auxiliary Adam update.
4. Momentum/first/second moments, FP32 parameter deltas and the next loss.
5. Stream/counter advancement and a resumed next update.

Run the test first in BF16/reference mode, then in the selected FP8 mode.
For a nontrivial global update, test both microbatch regrouping and rank
partitioning. Test a partial final update separately. Eight ranks must agree
on model/config/data/runtime identities and enter/leave collectives together.

The comparison harness must distinguish update error from whole-weight error:
an unchanged large weight can conceal a missing small update. Report role-wise
relative RMS error, maximum error, cosine agreement and nonfinite counts.
Freeze composed-update tolerances before the pass/fail run using independent
reference/reduction controls. Existing per-kernel tolerances cannot simply be
declared full-model update tolerances. Missing contributions, wrong scale by
rank/accumulation count, or zeroed significant gradients are hard failures.
If the reference controls cannot support a trustworthy numerical envelope,
the gate remains unresolved.

### B2. FP8 versus BF16 learning control

Use the same full model, LR recipe, seed, global examples, batch and token
schedule. Run a central dense candidate through approximately 2B tokens in
both modes, including warmup and plateau. An existing FP8 HPO arm may provide
its half of the comparison if identities match exactly; account for reuse.

Require finite updates, appropriate parameter movement, no unexplained
systematic collapse of any role's gradients, and the predeclared per-scope
held-out non-inferiority check. Compare the trend through at least two
post-warmup evaluations. Extend an inconclusive comparison rather than
declaring parity from a few finite steps. A BF16 control also helps determine
whether a failure is the optimizer/data or the FP8 implementation.

If FP8 fails and BF16 passes, repair or choose a qualified FP8 backend. A BF16
fallback requires a revised measured token budget. Failing FP8 never justifies
continuing on the assumption that more tokens will repair a kernel error.

After selecting final HPs, repeat a bounded matched FP8/BF16 continuation if
their rates or operating statistics lie outside the earlier control's tested
regime. Use the same trained parent state and next data, with explicitly
declared precision branches and state handling. Price this in B/contingency;
do not claim that a low-rate central control certifies arbitrary higher-rate
FP8 trajectories.

## 7. Execution optimization

Choose execution settings before the learning grid, then hold them fixed for
paired quality comparisons. A later change needs equivalent evidence and a
recorded runtime/source transition.

### Ordered optimization queue

| Order | Variable / candidates | Selection rule |
| --- | --- | --- |
| 1 | Qualified TE/HipKittens versus AITER/local-wgrad; independent Triton as diagnostic/contender | Quantization-inclusive full forward/backward/optimizer time, numerical gates and BF16 trajectory evidence |
| 2 | Microbatch 4/8/16, then 32 if it fits; fixed global batch | Highest sustained throughput with 15% HBM headroom through cadence and no numerical regression |
| 3 | Local/global attention backend; AITER CK versus independent CK where supported | Preserve exact local mask/GQA/value semantics; use traces to identify exposed attention work |
| 4 | Local wgrad split-K 1/2/4/8 and shape-specific tiles | Count FP32 scratch/reduction overhead; TE owns its internal split policy |
| 5 | BF16 tied head and CE chunks 2048/4096/8192; applicable hipBLASLt tuning | Measure full head forward/backward and tied-gradient correctness |
| 6 | RCCL/DDP bucket cap 25/64/128 MiB; rank/CPU affinity | Minimize exposed slowest-rank communication, not just collective microbenchmark time |
| 7 | Data-loader CPU threads, prefetch, mmap locality and validation cadence | Remove measured data waits without starving collectives or serializing CPU work |
| 8 | Optimizer batching and native pointwise/quantizer fusion | Preserve NS5, FP32 state and BF16 rounding; implement only for a measured exposed cost |
| 9 | Selective activation checkpointing | Only if memory prevents a faster microbatch and recomputation wins whole-update time; preserve first-layer value-source and RNG/cache behavior |
| 10 | Compile/HIP graphs or FP8 head | Separate engineering/quality work; current AMD production validator rejects graphs/compile and the existing graph trainer is single-rank |

Do not optimize all ten items by default. A change that halves a component
occupying 10% of step time can improve total time by at most about 5.3%.
Prefer fixes with meaningful measured end-to-end benefit and low risk.

For valid finalists, use warmed alternating timing pairs rather than testing
one configuration only while the node is cold. The existing baseline protocol
is five warm updates, three alternating 20-update pairs and a 40-update soak.
Record all-rank p50/p95 time, slowest-rank throughput, data wait, memory,
thermal behavior and whether stability probes are enabled. Add a longer
cadence-inclusive soak in work package E. Profiled timings are diagnostic;
use unprofiled runs for acceptance.

Promote an execution change only with at least a credible **3% whole-update
improvement**, no repeated tail regression, and the same numerical/quality
gates. This is an engineering threshold intended to avoid spending time on
noise. Include development and validation cost in the payback decision.

Large HBM is used to retain activations, enable efficient microbatches and
avoid memory sharding when beneficial. It does not require increasing the
global batch or filling unused memory with the entire dataset. Start with
replicated DDP; add FSDP/ZeRO/TP only if measured memory or communication
constraints justify the extra implementation and checkpoint complexity.

## 8. Dense hyperparameter experiments

### Candidate contract

All promotion-bearing runs use the actual 1.446B model and eight GPUs.
Architecture, tokenizer, corpus prefix, precision/backend, logical optimizer
algorithm, evaluation set and seed are fixed within each comparison. Trial
specifications live outside Git and contain every optimizer field, not just
overrides whose defaults can drift.

Initial engineering search grid, **not claimed optimal**:

| Variable | Initial values / policy |
| --- | --- |
| Spectral Muon peak LR | **0.006, 0.012, 0.024** |
| Tied embedding/head AdamW peak LR | **0.0005, 0.001, 0.002, 0.004** |
| Ordinary/Peri/final norm and value-coefficient peak LR | 0.00015 initial calibration value; test half/double in the auxiliary stage |
| Learned Q/K gain peak LR | 0.00005 initial calibration value; test half/double or isolate if diagnostics require |
| Matrix weight decay | 0.02 initial; 0.08 controlled alternative, and 0 only if shrinkage diagnostics call for it |
| Tied embedding weight decay | 0 initially; do not automatically apply matrix decay |
| Muon momentum / Adam betas | 0.95 / (0.9, 0.95) initially |
| Adam epsilon | 1e-8 initially |
| Global gradient clipping | 1.0 initially; clip once after accumulated/reduced gradients |
| Global batch for LR grid | 2,097,152 tokens |
| Warmup for the initial grid | 1,006,632,960 tokens = 480 updates at 2M |
| Subsequent schedule during screening | WSD plateau; no forced decay at each early stopping rung |

The LR ranges span meaningful coefficients under this model's actual
parameterization. They are deliberately broad enough to discover an unsafe
upper edge and a slow lower edge. The old winner receives no exemption from
the new criteria. If the best viable region lies at a grid boundary, use a
preregistered refinement/expansion instead of declaring a boundary point an
optimum. The refinement must fit the same budget and displace a lower-value
exploratory trial, or use contingency explicitly.

For separate qualification runs, use a common pilot horizon of at least
**16U = 32.2123B tokens**, with decay in its final 10%, and explicit stop
bounds for every rung. Its cooldown then lies beyond all the dense plateau
checks. A default 10B manifest is insufficient for the 12B finalists. This
pilot horizon is a deliberate experiment contract, not a claim that those
checkpoints already have the correct trillion-token schedule. A reusable
production prefix instead needs the actual funded production contract or an
explicitly tested, recorded fork into it.

The initial norms, decay and moments are provisional defaults; they are not
asserted to have been tuned. Exhaustively crossing every scalar is infeasible.
The staged design measures main LR effects first, tests the most consequential
remaining groups, then validates complete recipes from initialization.

### D1. Coarse joint LR screen: 12 candidates to U

Run all 3x4 Muon/embedding combinations from the same initialization and data
prefix, each to **U = 2.013B tokens**. This is 960 updates at 2M: 480 warmup
and 480 plateau updates. Start with the central candidate before running the
most aggressive combinations; inspect finite updates and parameter movement
at early checkpoints. Keep the full node at eight ranks for each trial.

Stop a true numerical failure immediately. Do not rank a warmup-only run or
prune simply because it learns more slowly during the ramp. At U, remove
clearly unstable/dominated cases, but preserve up to four candidates spanning
the viable LR region, including a lower-rate contender when statistically
plausible. The early winner alone does not get all remaining budget.

Cost allowance: **12U = 24.159B exposures**. A partial failure is charged its
actual GPU time and is recorded as failed/incomplete rather than ranked.

### D2. Four candidates to 2U

Continue up to four survivors to **2U = 4.0265B tokens**, retaining their exact
run contracts. Each receives another U; allowance **4U = 8.053B**.
This provides 1,440 plateau updates beyond the 480-update ramp at 2M.

Select two credible base recipes using post-warmup NLL, learning slope,
generated development signals where informative, clipping and role-wise
update/weight behavior. Prefer an interior stable region when scores overlap.
Record any LR ranking change between U and 2U.

### D3. Auxiliary groups and matrix decay: four matched branches

From one credible 2U parent, run four diagnostic continuations for U each:

1. Unchanged control through the new dense-fork tool.
2. Norm/value and Q/K gain rates multiplied by 0.5.
3. Those small-group rates multiplied by 2.
4. Matrix decay 0.08 instead of 0.02, with the other rates unchanged.

Allowance: **4U = 8.053B**. Compare every branch with the unchanged control
from the same parent, same next data, same counters and moments. These are
local continuation tests; they do not establish a from-scratch optimum for
the modified recipe. The selected complete combinations are restarted from
initialization in D4. Isolate Q/K gains from other norms if their signals
disagree; that requires replacing or explicitly funding another branch.

A changed decay value does not mathematically require resetting moments, but
the current QAT converter does not support that dense branch. The new dense
fork must explicitly permit and record it. Changes to betas, momentum, epsilon
or clipping are conditional follow-ups driven by measured behavior, with
matched state policies and later from-scratch confirmation. Do not change
multiple untracked variables to rescue a candidate.

### D4. Joint global-batch and recipe comparison

Take two complete candidate recipes, P and Q, incorporating only supported
auxiliary choices. Run the full 2x2 comparison **from initialization**:

| Arm | Recipe | Global tokens/update | Microbatch 16 accumulation |
| --- | --- | ---: | ---: |
| B1-P | P | 1,048,576 | 2 |
| B2-P | P | 2,097,152 | 4 |
| B1-Q | Q | 1,048,576 | 2 |
| B2-Q | Q | 2,097,152 | 4 |

Use the same global data prefix, token-based warmup/schedule and evaluation
coverage. Run each to **2U**. Conservative allowance **8U = 16.106B** assumes
no reuse of an earlier identical arm. If one is reused, record identities and
subtract its cost rather than silently double-counting it.

This design can reveal LR/batch interaction: one recipe may be preferable at
1M and the other at 2M. Do not extrapolate the best LR from one batch solely
with a linear or square-root multiplier. Compare per-token and per-hour
learning separately. Holding a token warmup fixed means twice as many warmup
updates at 1M; that is an explicit policy being tested, not identical dynamics.

If warmup-to-plateau behavior is the problem, test the best 2M candidate with
**2,013,265,920 warmup tokens** versus 1,006,632,960, using a common later
evaluation endpoint. A run ending at warmup cannot rank warmup policies.
This extra arm consumes a priced contingency slot. It may rescue an LR that
failed a shorter ramp; do not mislabel a warmup failure as proof the LR can
never work.

The old 393k batch is optional diagnostic evidence. A 3M/4M batch enters only
if a measured execution advantage makes its additional quality comparison
worthwhile. This plan does not require a huge Cartesian sweep of every batch,
microbatch and LR. Microbatch changes at fixed global batch belong to section 7.

### D5. Two complete finalists to 6U

Continue the two selected batch/recipe arms from 2U to **6U = 12.0796B tokens**.
Allowance **8U = 16.106B** for the additional tokens. Both remain eight-rank
runs with their original manifests and schedule identities. At 2M this is
5,760 updates in total; 1M has twice as many updates at the same token horizon.

Review curves at 2U, 4U and 6U. Require continuing useful learning, absence of
unexplained progressive norm/embedding/FP8 degradation, and no critical-domain
regression hidden by the average. Use held-out uncertainty, not the last noisy
training batch. Preserve a lower-rate finalist until these comparisons are
complete if its longer-horizon performance remains credible.

Twelve billion tokens is a materially longer test than 500M, but still about
1.2% of a 1T program. It does not certify the final LR ranking. The long-run
decision therefore also uses D7 and the production review points below.

### D6. Independent-seed confirmation

Run the selected complete recipe from a second initialization through **2U**.
Allowance **2U = 4.0265B**. Check warmup safety, learning trend and the chosen
region's robustness. One extra seed tests reproducibility of the chosen recipe;
it does not provide a high-powered statistical comparison of all candidates.
If the finalists' ranking is ambiguous, a paired second seed for both is a
separate contingency expense, or retain the simpler/lower-risk interior choice
without claiming a decisive optimum.

### D7. Matched short decay probes

From the same 6U dense parent, create two diagnostic branches, each U long:

- Stay at the peak for U/2, then cosine-decay during U/2 to 0.1 of peak.
- Cosine-decay throughout U to the same 0.1 endpoint.

Both end at **7U = 14.0929B tokens**, consume the same next data and start from
the same moments. Allowance **2U = 4.0265B**. These compare cooldown duration
at a common endpoint. They do not reproduce a 100B cooldown or make a cooled
branch eligible for ordinary continuation under a still-high production LR.
Record both final NLL/tasks and the learning lost while annealing.

If the selected LR and lower-LR finalist are close and the decision depends
on annealing, repeat the same branch design on the second finalist before
claiming their final quality differs. Fund that explicitly from contingency;
otherwise qualify the chosen schedule without making that comparison claim.

### Dense selection record

The stage allowances sum to **40U = 80.5306368B**. Warmup rescues, grid
refinements, an additional seed, extra controls or longer runs are not hidden
inside this number. Every such proposal names the uncertainty it resolves,
estimated cost, affected budget and replacement trial if necessary.

The chosen record must specify every rate/group, momentum/betas/epsilon,
decay, clipping, global batch, microbatch, accumulation, schedule, source,
runtime, exact data and evaluation identities, and reasons for rejection of
the alternatives. A winner means best supported among the measured choices,
not a mathematical global optimum.

## 9. Long-horizon and schedule decisions

The original HPO ranked after roughly 250M warmup plus 250M peak-rate tokens.
The new longer runs reduce that limitation; they do not remove it. OLMo 2's
reported LR crossover beyond 200B tokens is a direct reason to avoid selecting
solely on rapid early loss reduction. Its annealing results also show why a
late cooldown can matter as much as the early plateau ranking [R1]. The exact
result does not transfer automatically from AdamW/OLMo to spectral Muon/CI-1.

### Production schedule proposal

- Use WSD for the initial production policy: a tested linear warmup,
  stable phase and predeclared final cosine cooldown.
- The starting dense cooldown candidate remains **the final 10% of dense
  phase tokens to 0.1x peak**. The starting QAT cooldown candidate is **the
  final 20% of QAT phase tokens to 0.1x peak**. D7 and QAT pilots inform these
  choices; they do not prove those fractions optimal.
- Choose the warmup after the production-width/batch comparisons. Do not
  scale a 250M warmup to 10% of a trillion tokens merely because a small
  historical schedule used a 10% fraction.
- Keep global batch fixed within a phase initially, and preferably across
  dense-to-QAT conversion. Any later batch ramp is a new state/data transport
  feature with separate tests, not an automatic response to spare memory.
- Prefer robust behavior in a stable LR region over a marginal early gain at
  the edge of numerical failure. Do not multiply/divide LR by the horizon
  ratio without a validated law for this mixed optimizer.

### Prefixes, forks and funded endings

A qualification manifest can be a bounded, separate experiment. If it is
intended to become production, it must start with the final validated data
contract and the correct long schedule, then stop at explicit update bounds.
The present exact-resume contract rejects changes to optimizer, horizon,
schedule, world size and other semantic fields. A 14B decayed pilot cannot be
rebranded as the first 14B of an undecayed 1T run by editing JSON.

Complete the dense-fork tool before relying on budget-driven schedule changes.
It should preserve a source checkpoint, make a new child lineage with declared
future changes, prove the consumed data prefix and record state handling. A
no-change fork must agree with ordinary continuation. Do not claim bitwise
identity for a deliberately changed schedule. Never erase the parent.

Review the production trajectory at approximately **25B, 50B, 100B, and every
subsequent 50B tokens**, plus every data/precision/phase transition. Round these
to complete updates. These are production checkpoints, not automatic new
sweeps. Examine domain quality, long-term gain/embedding trends, learning per
GPU-hour and the remaining funded horizon. A targeted continuation/annealing
branch is justified by a specific unresolved decision and a recorded budget.

At every review, price the remaining plateau, cooldown, QAT, evaluation,
checkpoint/restore and export work from measured rates. Maintain sufficient
time to complete the declared ending. If the forecast deteriorates, use the
tested fork policy to shorten the future horizon/cooldown while protecting the
parent, or stop at a scientifically interpretable checkpoint. Do not spend
the last day discovering that the planned QAT or decay has not started.

## 10. Ternary QAT qualification

### Software and arithmetic gate

Ternary training uses FP32 latent weights and optimizer state with effective
projection values in {-1, 0, +1} and scales. Those codes can be executed by
MI325X FP8 matrix hardware. The current AMD implementation rejects this mode;
there are no native acceptance receipts. Lack of ternary cores is not the
blocker.

The port must preserve the selected quantization grouping, rounding/ties,
scale computation and STE gradient behavior. Verify effective weights,
zero/tiny blocks, transpose layout, activation quantization, FP32 wgrad,
latent-gradient path and optimizer updates against the reference. Ordinary
finite-difference checking does not validate a deliberately defined STE;
compare against its explicit intended backward rule. Test dense/QAT caches
and conversion immediately after checkpoint loading.

Use the same architecture/tokenizer and a proven continuing stream. Preserve
compatible optimizer moments and per-rank RNG unless an explicit matched
reset experiment is being run. Transition receipts must contain the actual
complete-update token boundary, not the old 500M HPO endpoint. Test same-global-
batch microbatch regrouping and cursor rebasing. A changed global batch at
conversion is currently rejected and is not part of the initial campaign.

### QAT rate/recovery pilot on the full model

Use a credible **6U dense parent**, identical data continuation and frozen
held-outs. The parent can be reused from dense qualification if identities
match. Register one dense continuation control and three QAT candidates:

| Candidate | Muon multiplier versus selected dense peak | Embedding multiplier | Norm/QK/value multiplier |
| --- | ---: | ---: | ---: |
| Q1 | 0.25 | 0.25 | 0.25 |
| Q2 | 0.5 | 0.5 | 0.5 |
| Q3 | 0.5 | 1.0 | 0.5 |

These are a bounded initial search, not a claim that QAT must use half the
dense LR. Keep state carry and the quantizer fixed while comparing rates.
Use a phase-relative **1.0066B-token re-warmup** for these pilots, then a stable
phase. The dense control continues its qualified dense schedule. Initial
conversion loss is measured before any update.

Run each QAT candidate and the dense control for **U = 2.013B tokens**, then
extend the best credible QAT branch and the dense control to **2U = 4.0265B
additional tokens**. Total allowance: **6U = 12.0796B exposures**, of which
4U are QAT and 2U are dense control. At illustrative all-in rates of 20k QAT
and 40k dense tokens/s/GPU, that costs about **139.8 GPU-hours**. Actual rates
and operating tests determine whether it fits the 160-hour package.

Use a declared QAT pilot phase horizon of at least **4U = 8.053B tokens**, so
the final 20% cooldown lies beyond the bounded 2U recovery test. The child
manifest's absolute horizon includes the dense parent's already-consumed
tokens; warmup and cooldown are phase-relative. Stop the pilot at its declared
bound. This does not spend or authorize all 4U of the planned phase.

Measure recovery trend relative to the immediate conversion loss, same-data
dense-control gap, per-scope losses, generated tasks, FP8 errors, latent-weight
motion, code occupancy/flip rates and scale evolution. An initial finite loss
jump is expected to be possible; persistent worsening or nonfinite gradients
is not excused as normal QAT. Do not choose a split from a pilot that has not
shown a credible recovery trajectory.

State reset or gradual quantization strength can be investigated only if the
initial pilots expose a specific problem. A strength ramp is not implemented/
qualified by removing the backend's current guard; it changes the experiment
and needs its own arithmetic and learning controls.

### Phase allocation and late-parent confirmation

The historical 60/40 dense/QAT token split remains one candidate. An 80/20
alternative is credible only if recovery and final-quality evidence justify
the shorter QAT period. A short early-parent pilot cannot identify the best
switch after hundreds of billions of dense tokens. Keep this uncertainty in
the budget and final claims.

Before the real switch, preserve the trained dense parent and run a bounded
confirmation on that parent: conversion evaluation, initial re-warmup,
recovery trend and eight-rank checkpoint/restore. A successful QAT branch can
continue its declared schedule. A failed branch retains the dense parent and
must not consume the rest of the allocation while waiting for unspecified
recovery. Charge this late confirmation to the planned phase/remaining reserve
and track any failed branch separately.

Select quality tolerances before viewing final results. Proposed final-release
targets are an upper paired NLL increase no larger than **0.05 nats/token per
critical scope** against the protected production dense parent, and no more
than **2 percentage points** loss on the preregistered core accuracy aggregate,
with separate critical-language/task reporting. A 0.05-nat increase is about
5.1% relative perplexity. These are explicit engineering acceptance proposals,
not published guarantees; revise them only before the relevant comparison if
the project's product requirements differ. Require adequate sample size and
report inconclusive intervals rather than treating them as a pass.

The early pilot gate is finite execution and credible recovery/learning, not
instant satisfaction of the final-release margins. The final dense-parent
comparison uses checkpoints with different token exposure, so it is not a
matched-compute claim that QAT beats continued dense training. Dense-only
output is a fallback requiring an explicit project decision because it does
not fulfill the intended ternary deliverable.

Finish with native-QAT versus exported-model reference loss, deterministic
logit checks within the declared numerical envelope, generated task coverage,
file integrity, load/cache behavior and actual inference memory/latency.
FP8 representation of ternary weights does not itself imply a training speedup.

## 11. Storage, checkpoints and recovery

### Capacity and data path

Use immutable local/mounted filesystem token shards for the existing mmap
loader, and independent durable storage such as AWS S3 for recoverability.
The loader is not a production object-streaming implementation. Stage the
needed working data before it is consumed; do not make training wait for an
untested on-demand S3 fetch path.

- One trillion **unique stored uint16 tokens** occupies approximately 2 TB
  decimal before manifests and preprocessing workspace. Repeated exposure
  does not require repeated stored copies. Conversely, raw text,
  deduplication intermediates and temporary downloads need additional space.
- The selected optimizer/model has an estimated **12.18 GB initialized
  checkpoint** before serialization metadata/buffers. Measure the actual
  dense and QAT payload sizes. DDP should save one complete payload per
  checkpoint, plus all required rank RNG/state, not eight identical weights.
- Provision working data + simultaneous staging/download files + live and
  protected checkpoints + runtime/JIT caches + preprocessing scratch, then
  leave at least 20% volume free in the capacity calculation. A 4–6 TB working
  volume is an example for roughly 2 TB of token shards and bounded staging;
  two complete 3 TB snapshots would require more. Derive the actual order
  from the real corpus and measured retention requirements.
- Measure host RAM and pinned-memory peaks during loading, validation,
  checkpoint serialization and restore. Avoid swapping. Do not assume the
  host has enough RAM because GPU memory is large.
- Benchmark reads while training and writes/uploads while checkpointing.
  Average packed-token bandwidth can be modest, but page faults, validation
  access patterns and checkpoint bursts can dominate stalls.

### Checkpoint policy

Use a proposed **30-minute maximum local recovery-point interval**, shortened
during risky transitions if measured overhead permits. Convert it into the
trainer's update cadence using the measured rate; the current approximately
2B-token default can be much longer in wall time and must not be accepted
without this calculation. At 50k tokens/s/GPU on eight cards and the 2M batch,
30 minutes is about 343 updates. Slowdowns lengthen a fixed-update interval,
so the supervisor should also watch actual checkpoint age.

Keep at least the last three committed local checkpoints and protect the
selected candidate parents, best evaluated state, pre-conversion parent,
phase ends and final state. Use a bounded durable policy: recent rolling
copies, selected daily/quality milestones, all protected lineage points and
final artifacts. Retaining every 30-minute checkpoint for a month creates
many terabytes and is not required for routine recovery.

Upload only committed payload/receipt pairs. Verify content hashes and the
restorable metadata; multipart S3 ETags are not generally file SHA-256 hashes.
Never prune a local checkpoint before a verified durable successor exists.
Publish a durable manifest only after all required objects are committed.
Preserve the exact source revision, image digest, tokenizer, config, data
identities and all rank RNG/state needed for recovery.

A 12.18 GB payload at 50 MB/s takes about four minutes to upload; at 100 MB/s,
about two. Test the **actual provider-to-AWS path**, including retries and
download. The upload queue must drain faster than checkpoints arrive, with
space for failures and protected copies. S3 versioning/lifecycle policies
must not delete the only restorable lineage.

### Mandatory fault drills on eight ranks

| Drill | Required result |
| --- | --- |
| Graceful stop after a complete update | Committed checkpoint; correct counters and no partial update treated as committed |
| SSH disconnect | Supervisor/job remains in the intended state; no accidental launch or duplicate process |
| One worker killed during backward | Remaining ranks exit/fail within a bounded timeout; no hang consuming the allocation indefinitely |
| Interruption during checkpoint write | Previous committed payload remains recoverable; temporary/orphan files cannot be selected as valid |
| Corrupt/truncated payload or receipt | Hash/contract failure, clear incident and safe fallback to a known committed state |
| S3 outage/slow upload | Bounded retry queue, no premature pruning, backup-age/free-space alarms |
| Restore into an empty run working directory | Download, verify and resume all eight ranks with the original source/runtime/config/data contract |
| Replay following restore | Correct next examples/counters and declared numerical agreement with uninterrupted reference; moments/RNG covered |
| Dense-to-QAT restore | Correct ternary mode, FP32 latent state, refreshed caches and phase-relative schedule |

Do not simulate destructive faults on the only copy of a valuable production
checkpoint. Use qualification copies. Measure save, upload, download and
restore times; include them in the budget and watchdog thresholds. Configure
collective timeouts above legitimate worst-case cadence stalls with margin,
but detect actual hung ranks. The current five-minute collective timeout must
be reviewed against measured storage/evaluation time.

## 12. Monitoring and stop rules

### Minimum record

Each training log/receipt needs run/candidate ID, source/image/data hashes,
phase, global tokens/update, cumulative tokens and updates, all optimizer-group
LRs, LR phase position, elapsed allocated GPU-hours, global loss and valid
targets, pre-clip gradient norm and clip coefficient, memory, full step time,
data wait, and checkpoint/backup age. Store sampled per-layer activation,
Q/K/logit, norm/value-gain, logical gradient/update/weight and FP8 diagnostics
with explicit sampling frequency/coverage.

Record per-device utilization, power, temperature, clocks, RAS errors and
slowest-rank time. Diagnostic sampling should cover different microsteps and
ranks over time. Measure its overhead; keep rich diagnostics during
qualification, and retain enough in production to detect drift.

### Three levels of response

| Level | Trigger | Action |
| --- | --- | --- |
| Immediate numerical/data stop | Nonfinite loss/gradient/master weight/state; missing required gradient; invalid IDs/targets; source/data/runtime mismatch; failed checkpoint checksum; uncorrectable GPU error | Prevent further updates, preserve the last valid committed checkpoint and incident artifacts, fail the bounded job. Investigate before any retry. |
| Immediate operating stop | Allocation cap reached; no safe checkpoint/queue disk margin; prolonged lost rank; required durable backup unavailable beyond the configured limit | Graceful checkpoint if safe, bounded termination otherwise; preserve recoverability and operator-visible cause. |
| Review alert | Persistent heavy clipping, unusual update/weight growth, attention/norm/FP8 drift, deteriorating held-out learning, thermal throttling or throughput loss | Inspect matched diagnostics/data; run only to the predeclared safe boundary while reviewed. Do not automatically change LR, skip batches or reset moments. |

Starting alert policies to freeze after the qualified baseline:

- After warmup, clip coefficient below 0.1 on more than 20 of the last 100
  updates, or a persistent material increase relative to the qualified
  recipe. This is a diagnostic alert, not a universal proof of bad training.
- Training loss outliers relative to a robust local baseline, with associated
  gradient/activation anomalies. For example, flag five updates in twenty
  above the rolling median by the larger of 0.25 nats or six scaled MADs.
  Use phase/data-aware baselines; do not interpret a planned QAT conversion
  jump using the dense steady-state threshold.
- Per-role update/weight, Q/K/logit or FP8 error statistics persistently above
  three times their qualified baseline range. Review small/zero-denominator
  groups separately. A fixed universal ratio is inappropriate for all roles.
- Held-out degradation beyond the predeclared per-scope margin at two
  consecutive evaluation points, or a persistent loss-slope deterioration
  against the credible comparison curve. A nonstationary data phase must be
  evaluated with its declared schedule and fixed held-outs.
- Unexplained sustained step-rate regression over 10%, or repeated p95 tails
  that threaten the funded horizon. Separate validation/checkpoint work from
  unexplained idle time.
- Local checkpoint age above 30 minutes plus measured save tolerance, durable
  backup age above a proposed 60 minutes, or upload backlog approaching the
  protected free-space limit. Tighten these if the measured recovery objective
  and network support it.

These alert numbers are engineering starting points. Register any changes
using qualification measurements before the long run; do not widen numerical
or quality limits after a candidate fails merely to promote it. Implemented
hard stops must be tested by fault injection. Existing sampled telemetry is
not an already-implemented all-rank automatic safety system.

### Incident handling

1. Preserve the last good checkpoint and the failing global batch identity,
   counters, logs, node health and relevant tensor summaries.
2. Determine whether the issue is hardware/I/O, data, implementation/precision,
   or optimization. Reproduce boundedly; use the BF16/reference path where
   useful. Avoid repeatedly paying to reproduce the same deterministic fault.
3. A transient infrastructure failure may resume the same contract after
   verification. A source, schedule, data or optimizer correction creates a
   reviewed child experiment and new evidence; it is not an invisible retry.
4. Limit automatic transient retries, for example two attempts with bounded
   backoff, and include them in the allocated-hour ledger. Never continue
   from a corrupt/nonfinite payload.
5. Recompute the remaining budget and ending after material lost time.

## 13. Production release and operation

### Release gate

The external release record must show all of the following as passed with
actual receipts, not merely listed as planned:

- Final model/algorithm identity and complete chosen optimizer configuration.
- Frozen tokenizer, audited data, development/final protocols and mixture.
- Native kernel/composed arithmetic, one/eight-rank equivalence and BF16
  learning control for the exact runtime/backend.
- Full-update/evaluation/checkpoint memory margin on every rank.
- Selected complete recipe through longer learning and seed confirmation,
  with per-token and per-allocated-hour comparisons and known limitations.
- Declared warmup, cooldown, phase budgets and intentional-fork support.
- QAT implementation, parent/child state/data continuity, recovery trajectory
  and measured throughput if the deliverable includes ternary weights.
- Fault drills, verified independent durable restore, retention/watchdog and
  operator incident procedure.
- Eight-rank unprofiled sustained throughput with operating overhead, and a
  funded final schedule including cooldown, QAT, final evaluation and export.
- Exact source/image/runtime/data/config fingerprints and a bounded launch
  command. Qualification receipts must match the released implementation.

No source change during a trial is silently accepted. Reuse unaffected
evidence, but rerun the tests whose computation or contracts changed. Preserve
the original runtime/checkpoints so repairs cannot erase the best known state.

### First production day and later reviews

Run the selected recipe with the registered limits and monitoring. Check the
first updates, warmup milestones, warmup exit, first plateau evaluations and
first durable checkpoint/restore verification. If a qualified prefix is
reused, document that exact lineage and the first newly consumed token.

At least daily, inspect quality by domain, sample generations, numerical
trends, hardware health, actual tokens/hour, checkpoint/backup status and the
remaining funded horizon. Use the 25B/50B/100B/50B review points for deeper
learning/schedule decisions. Avoid reacting to one noisy loss sample; also
avoid waiting for NaNs when multiple independent learning signals deteriorate.

Keep the QAT, cooldown and final validation time reserved in the rolling
forecast. The phase boundary must be affordable at the measured QAT rate.
The program ends with a verified final checkpoint, final evaluation report,
recoverable runtime/data lineage and tested export, not just a token counter.

### What success can honestly claim

The run should demonstrate numerical correctness within declared tolerances,
recoverability, stable observed training, useful measured learning and the
best supported quality/cost choice among its tested candidates. It cannot
claim a globally optimal LR, a guaranteed 1T/2T budget, H100 superiority,
unmeasured generated accuracy, or optimal late QAT allocation from a short
pilot. Report actual paid hours, rejected trials, production tokens, quality
and limitations together.

## 14. Execution order and artifacts

### Dependency order

```text
CPU data/evaluation/build/launcher/fork/backup/QAT implementation
    -> A: actual node + native arithmetic + full-size fit
    -> B: DDP/FP8-vs-BF16 + measured execution choices
    -> C: dense LR/group/batch/horizon/schedule experiments
    -> D: full-model QAT/control experiments
    -> E: eight-rank operating acceptance + restore + soak
    -> final measured budget and release record
    -> bounded production with scheduled reviews
    -> late-parent QAT confirmation, cooldown, final evaluation/export
```

CPU engineering/data work can proceed concurrently without a GPU rental.
Per-device arithmetic tests can be parallelized on the node. Selection-bearing
trials remain eight-rank jobs for this plan. Do not attempt to save time by
sharing one GPU between independent training processes or by mixing kernel
experiments with a quality comparison on the same live model.

The envelope corresponds to this maximum scheduling template, assuming each
package completes within its hours:

| Node-hour window | Work |
| --- | --- |
| Before rental | Complete section 4 work and stage data/evaluators |
| 0–8 | A: hardware/runtime/arithmetic |
| 8–18 | B: distributed/precision/performance |
| 18–82 | C: dense search and longer finalists |
| 82–102 | D: QAT/control qualification |
| 102–108 | E: recovery/soak/release validation |
| 108–120 | Contingency if required; otherwise return time to production |

This is a resource allocation, not a promised finish date. Failures block
dependent work. If unused time is returned, compute production from actual
spent hours. If costs exceed projections, issue a smaller/revised experiment
plan before the next paid trial; do not silently remove mandatory evidence.

### External artifact structure

Use actual provider paths and durable prefixes, for example:

```text
/external/ci1/
  frozen/          # source/image/tokenizer/data/evaluation identities
  data/            # immutable token shards and external manifests
  candidates/      # complete candidate specs, bounds and experiment reasons
  runs/<id>/       # immutable config, events, incidents, checkpoints
  comparisons/     # paired evaluation, gradient and throughput records
  accounting/      # allocation start/stop, stage spend, forecasts
  release/         # chosen recipe and all required evidence links
```

The candidate registry must include: parent/seed; fresh versus continuation;
scope of comparison; all rates and optimizer groups; global batch/microbatch/
accumulation/ranks; precision/backend; schedule and absolute/phase-relative
counters; token and wall-time cap; data/evaluation hashes; expected spend;
mandatory gates; stop reason; actual allocated spend; result and promotion.
No raw corpora, configs, checkpoints, trial results or credentials enter Git.

For each decision, retain one sentence explaining what the experiment could
change and what result would reject it. If that cannot be stated, the trial
is not yet worth launching. Maintain an explicit list of unresolved questions
and retire them with evidence rather than expanding the search indefinitely.

### Existing commands and missing commands

The repository already provides CPU count/config tooling, AMD runtime and
staging receipts, kernel preflight, bounded training/benchmark entry points,
collective probes, phase/budget calculators, local checkpoint audit and
reference-NLL evaluation. See [the AMD entry point](../README.md),
[acceptance requirements](../ACCEPTANCE.md) and
[operating behavior](../../OPERATIONS.md).

Do not launch a new LR grid through `production.amd.train` before implementing
its explicit qualification mode: `validate_production` currently rejects
optimizer values different from `SELECTED_OPTIMIZER`. Do not use the archived
200M HPO runner as a shortcut; it fixes a different model, batch, horizon and
research contract. Do not claim dense forks, live S3 backup, generated task
evaluation or AMD QAT are turnkey while their section 4 work is outstanding.

The calculators can already price scenarios without GPUs. For example, pass
measured phase rates to `python -m production.amd.phase_plan` with explicit
`--gpu-hours 5760 --gpus 8 --reserve-gpu-hours 960 --duty-cycle <measured>`
and the chosen QAT percentage. Its output is arithmetic, never a training
configuration or an authorization to launch. Replace 960 with the final
accounted qualification spend only when the remaining obligations are funded.

## 15. Research basis and limits

Existing primary-source audits and implementation references are collected in
[SOURCES.md](../SOURCES.md), [LR_REFERENCES.md](LR_REFERENCES.md),
[BATCH_STABILITY.md](../BATCH_STABILITY.md),
[SPEEDUP_TRIAGE.md](SPEEDUP_TRIAGE.md) and
[TRAINING_RISKS.md](TRAINING_RISKS.md). Their historical recommendations do
not override this new from-scratch qualification proposal.

| Ref | Primary evidence | Decision supported / limitation |
| --- | --- | --- |
| R1 | [OLMo 2, §§3.1, 3.4, 3.5 and 6.3](https://arxiv.org/html/2501.00656v1) | Repetition can trigger spikes; embedding decay/Adam epsilon affect stability; some LR rankings cross beyond 200B tokens, while annealing can equalize results. Supports data checks, role diagnostics, longer/annealed comparisons and hardware burn-in. Different architecture/AdamW; its LR values are not CI-1 prescriptions. |
| R2 | [MiniCPM, §§3–5](https://arxiv.org/html/2404.06395v2) | Width parameterization, batch/data-efficiency tradeoffs and WSD branch experiments. Supports explicit parameterization and separate schedule analysis, not copying base LR 0.01. |
| R3 | [MiniCPM4, §2.3 and §3.3](https://arxiv.org/html/2506.07900v2) | Small-model/task-conditioned search and QAT allocation studies. Supports measuring downstream learning and quantization recovery. Its tiny pilots/final 8B recipe do not select CI-1's optimum. |
| R4 | [Muon/Moonlight, §2.2 and Table 2](https://arxiv.org/html/2502.16982v1) and [our spectral equation](../../../docs/MUON_SPECTRAL_IMPLEMENTATION.md) | Numeric LR comparisons require the actual shape scaling and separate auxiliary groups. Keep the selected Muon algorithm while tuning coefficients. |
| R5 | [The Best Optimizer Depends on Batch Size, v2](https://arxiv.org/pdf/2610.08975v2), earlier source audit S30 | Batch/LR/decay interactions in a different MuonW/AdamW model. Supports testing both recipes at both batches. No automatic LR, beta or decay transport is selected. |
| R6 | [Gradient noise scale](https://arxiv.org/html/1812.06162v1) and [non-Euclidean batch diagnostics](https://arxiv.org/html/2602.03001v2) | Useful batch depends on learning state and optimizer. Diagnostics may guide a comparison; a scalar noise estimate does not certify stateful mixed Muon/AdamW. |
| R7 | [ParetoQ](https://arxiv.org/html/2502.02631v1), [Continual QAT](https://arxiv.org/html/2502.11895v1), [BitNet 2B4T](https://arxiv.org/html/2504.12285v1) | Published quantization regimes and allocations differ. Supports independent QAT rate/state/schedule tests rather than a universal 60/40 claim. |
| R8 | [MiniCPM5 official release](https://huggingface.co/openbmb/MiniCPM5-2B), [UltraData](https://arxiv.org/html/2602.09003v1) | Data quality/staged training matter; released configs and auxiliary data experiments do not disclose a directly copyable final optimizer recipe. |

All numerical grids, resource allocations, alerts and promotion criteria in
this plan are explicit engineering proposals unless identified as measured
local behavior or a quoted research result. None are native MI325X results.
