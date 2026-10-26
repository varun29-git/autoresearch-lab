# Acceptance record and hardware gates

Prepared 2026-10-24. This handoff is ready for **native testing**, not certified
for a month-long production campaign. All paid commands remain human-run.
Keep test reports, profiles, configs, evaluation records and checkpoints
outside Git. Do not record a gate as passed without its external receipt.

## Local checks

The development host is macOS with Python 3.14 and PyTorch 2.13; it has no
ROCm GPU. Test dependencies were installed outside the repository. Native
GPU tests are skipped rather than represented as passing CPU checks.

The completed checks were:

- Full suite: **360 passed, 27 skipped, 183 subtests passed** in 232 seconds,
  using `python3 -m pytest tests production/amd/tests -q`.
- After the last constructor/runtime refinements: **48 passed, 3 skipped,
  20 subtests passed** across AMD, dense-FP8, FP8 and production-config tests.
  This overlaps the full suite and must not be added to its count.
- All four external config-builder/count-only CLI runs: preserved exactly
  1,446,478,126 parameters and the example global batch/accumulation.
- Budget CLI arithmetic, all eight AMD command help entry points, local
  documentation links, Python byte-compilation, both shell scripts' syntax,
  and `git diff --check`.

The first full run lacked Protobuf in the local test environment. Installing
the declared dependency externally resolved the seven tokenizer setup errors;
the full suite above then passed. Existing tests emitted one tensor-to-scalar
warning from a test error metric. None of this is a native AMD numerical result.

CPU evidence covers exact parameter count and architecture preservation,
unchanged optimizer selection, FNUZ reference scale/transpose semantics,
autograd operation wiring, FP32 accumulated wgrad, accumulation and cache
refresh including inference, immutable data receipt invalidation, global
batch derivation, slowest-rank accounting and phase/budget arithmetic.
It cannot verify Triton gfx942 code generation, vendor binary interfaces,
native numerical agreement, actual memory fit or training speed.

## Native gates, all unrun on this host

| Gate | Required evidence | Blocking consequence |
| --- | --- | --- |
| Image build | Clean Linux build, preserved ROCm torch/Triton, `pip check`, dependency freeze, final image digest, compiler/native source identities | No native kernel certification from a source-only Dockerfile |
| Device/topology | Actual MI325X name/capacity, gfx942, eight visible cards, xGMI peer layout, driver/firmware/power policy, CPU affinity, storage | Do not substitute an unknown partitioned or host-fallback topology |
| Kernel preflight | `production.amd.preflight` passes all attention/FP8/pointwise/CE/model/optimizer checks on the frozen runtime | No training through an absent/stale/incompatible receipt |
| Native arithmetic | FP32 wgrad output, nonfinite propagation, zero padding, cache invalidation; independent math comparisons and small native trajectories | A forward-only GEMM/attention success is insufficient |
| Full-size memory | Selected 1.446B geometry, full optimizer state, real sequence/batch, allocation and reservation peaks, cadence/workspace headroom | Do not promote an OOM-prone maximum batch |
| DDP correctness | One rank and 8 ranks consume the same global stream/update; compare loss, gradients, clipping, logical optimizer states and parameter deltas | Do not count wrong normalization or missing reductions as a speed gain |
| Sustained rate | Three alternating 20-update pairs, 40-update soak, all-rank timing, real data, exact numerical candidate and source/runtime | No synthetic-component or warm-cache short-screen throughput claim |
| Operations | Timed scheduled checkpoint/validation, graceful stop, abrupt worker loss, committed checkpoint recovery and durable restore | Do not rely on a single untested local checkpoint copy |
| Quality | Preregistered language/domain development metrics and tolerances at matched tokens, plus matched-GPU-hour comparison | FNUZ/BF16/kernel loss proximity is not convergence evidence |
| Budget | Actual eight-rank phase rates, elapsed bring-up hours, measured cadence/recovery duty cycle, selected token horizon | No 1.5T/2T claim from vendor peaks or the pasted estimates |
| QAT | Explicit phase tokens/rates/data/state policy; native quantization/transition tests and dense→QAT held-out comparison | No AMD ternary launch from the dense candidate |
| Multi-node scale | Repeat topology, RCCL/DDP correctness, timing, data staging and recovery at each 2/4-node size | Single-node results do not certify 16/32 ranks |

## Numerical comparisons

The preflight's engineering tolerances are explicit in `preflight.py`:
attention and reduced-model checks are approximately 3% relative with an
absolute near-zero guard; projection FP8 versus independently quantized
FNUZ reference is 2.5%; quantizer dequantized error is 0.4%; optimized
pointwise/learned-QK/CE comparison is 1.2%. These are finite-precision bug
detection thresholds, **not** release-quality criteria. Preserve and inspect
the per-case errors; do not widen a failed tolerance simply to pass a port.

Use archived H100 activation/gradient/checkpoint dumps if the operator has
them externally. Their exact source, geometry, numerical mode, global batch,
tokenizer and token stream must be known. The pasted dossier's suggested
loss threshold cannot be treated as an already selected production quality
gate. If matching H100 hardware/dumps are unavailable, record that limitation
and compare AMD FNUZ implementations against the independent FNUZ oracle and
the explicit AMD BF16 control, then evaluate real training trajectories.

For single-rank versus DDP numerical comparison, keep total update tokens,
initial model/state, global token order, schedule and clipping fixed; adjust
local accumulation for world size. Record tolerances for floating-point
reduction order. Check Q/K/V and gate/up **logical** Muon slices, tied head
plus input-embedding gradient, learned Q/K gains, Peri-LN gains and value
coefficients. Compare optimizer deltas and moments as well as final weights;
small updates can be wrong while whole-weight relative error looks small.

## Configuration-selection follow-up, 2026-10-25

The researched choices in [CONFIG_SELECTION.md](CONFIG_SELECTION.md) and
`profiles.py` are selected for testing, not native certification. Local checks:

- 2026-10-26 fixed-HPO-batch follow-up: **26 AMD profile CPU tests passed**
  after adding `batch_bridge_mb12` (8 GPUs × 12 sequences × 4096 tokens ×
  1 microstep). The comparison preserves the 393,216-token global update,
  selected optimizer settings, held-out coverage and ordered token stream.
  Full-width learning and native MI325X fit/speed remain unmeasured.
- 2026-10-26 batch/transition follow-up: **80 CPU tests passed, with 7
  subtests**, across AMD profiles and phase planning, dense-to-QAT transition,
  trainer and production-config regressions. The new 1M-token profile preserves
  global update accounting. Changing microbatch at the phase switch now
  rebases the next microstep while preserving the token stream, update count
  and optimizer state; the CPU test exercises conversion and exact resume.
  The 35 transition/profile tests passed again after adding the parent
  complete-update boundary guard.
  Native eight-rank QAT implementation, fit and learning comparisons remain
  unrun. See [BATCH_STABILITY.md](BATCH_STABILITY.md).
- Microbatch-24 follow-up: **23 profile tests and the two-process resume
  regression passed** together (24 tests, 113 seconds). They cover the
  3.146M-token fixed-global-batch pair, equal held-out examples with a partial
  final evaluation batch, and the operational resume contract. A broader CPU
  run passed 71 checks, overlapping this set, and had a resume identity
  mismatch while the source file was edited during that run; the clean rerun
  passed. MI325X fit, full-update speed and batch-quality gates remain unrun.
- AMD tests plus production config, real data-loader and training regressions:
  **67 passed, 1 skipped, 7 subtests passed** in 115 seconds. This included CPU
  DDP and exact-resume checks; the skipped test needs native AMD hardware.
- After adding the explicit backend override: **19 profile tests passed**.
  This overlaps the preceding run and is not an additional independent count.
- The profile tests cover every named choice, real config serialization,
  preservation of optimizer/phase data, exact global stream under microbatch
  and rank changes, fixed held-out coverage, schedule endpoints, metadata
  rejection and shell-safe external output paths.
- Three CLI bundles (`balanced`, `mb8` with explicit AITER, and `mb4` smoke)
  generated successfully from **metadata-only CPU fixtures** outside Git;
  count-only preflight reported 1,446,478,126 parameters for each. No fixture
  was passed off as a validated training corpus or GPU acceptance receipt.
- Generated shell syntax, Python byte-compilation, local documentation links
  and `git diff --check` passed.

The profile policy now selects the initial paired-reference-NLL guardrail
at 0.02 nats/token per task/language confidence bound, rather than borrowing
an unsupported threshold from the migration dossier. Native tests, production
data verification, actual HBM/throughput, S3 recovery and early quality bridges
remain unrun. All native gates above still apply.

## Dense/QAT allocation planning checks

Allocation-planning follow-up (2026-10-25): **64 CPU tests passed** across
the new dense/QAT budget planner, selected profiles, AMD contracts and existing
CPU transition mechanics. The planner was cross-checked against the original
fixed-token budget calculator; it preserves the requested token ratio,
rounds to complete updates, and accounts for both phase rates and overhead.
External JSON CLI output, local documentation links and `git diff --check`
also passed. An extreme-input overflow found during testing was corrected.
These checks do not enable native AMD ternary execution or resolve the data
prefix-equivalence audit described in [QAT_ALLOCATION.md](QAT_ALLOCATION.md).

## Checkpoint and recovery drill

1. Run a bounded real-data training job to a committed checkpoint; retain its
   exact source, config, native preflight/runtime/staging receipts and image.
2. Continue uninterrupted for a declared number of updates and retain its
   checkpoint, token cursor, RNG/optimizer state and evaluation results.
3. Restore the earlier exact checkpoint into a compatible run and repeat
   those updates. Verify stream/cursor, update counts and restored states;
   compare numerical results under the declared native nondeterminism policy.
4. In a separate bounded job, have the human terminate one worker abruptly.
   Verify the worker group fails promptly, partial payloads are not committed
   as latest, and the last valid receipt remains recoverable.
5. Archive the committed checkpoint with `production.checkpoints`, verify
   digests, simulate loss of the working copy using separate test storage,
   and restore from durable storage. Never delete the sole recoverable copy.
6. Measure checkpoint+validation, restore, staging verification and JIT costs.
   Feed the observed costs into the month forecast and cadence choice.

An exact-resume contract intentionally rejects changes in source, runtime,
world size, candidate arithmetic, data and schedule. Do not edit a receipt
to bypass it. Backend/world-size migrations need a separate reviewed state
transfer procedure; this handoff does not silently reset optimizer state.

## Production decision

Freeze the winning **measured** candidate and global batch before selecting
the final data/token horizon and decay schedule. Do not leave a 1.5T manifest
in place when measured cost only funds approximately 1.0T and expect WSD to
finish correctly. Likewise, do not call a longer repetition-heavy run better
without held-out quality evidence. The quality objective, data mixture,
token budget and checkpoint durability remain part of the production plan.

The delivered launcher enforces runtime, native preflight, staging, reviewed
model/optimizer, global batch and world-size identities. Full-size quality,
soak and recovery are operator-reviewed gates; the launcher does not pretend
that the kernel receipt certifies them.
