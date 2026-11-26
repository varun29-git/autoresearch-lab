# Five-hour initial pilot on eight MI325X GPUs

Proposal, 2026-10-28, in response to the requested 3–5 total clock hours.
This is an initial viability/learning decision, not completion of the
[full qualification plan](FULL_QUALIFICATION_PLAN.md). Its hard allocation
ceiling is **five wall-clock hours × eight allocated GPUs = 40 GPU-hours**,
approximately 0.69% of 5,760 GPU-hours. It is counted within, not added to,
the larger qualification envelope. Further work is a separate priced decision;
the 960-hour envelope is not an instruction to spend all of it.

## Prerequisites

Before starting the paid clock, finish the CPU engineering needed for bounded
dense candidate execution, the image/dependency preparation, the pilot data
and frozen evaluation sets, and the checkpoint/restore procedure. The current
AMD launcher locks the old rates, so the qualification mode is required before
these new candidates can run. Native imports/JIT and hardware correctness still
need the node; if they fail or exceed their time window, investigate and stop
within the cap instead of skipping a required check to fit the timetable.

Native QAT implementation/learning is not completed by this dense pilot. A
mixed dense/QAT production run still needs the corresponding full-plan gates.

## Recommended revision: maximize correctness and early stability evidence

The five-day plan includes substantial hyperparameter experiments, precision
controls, QAT and operating qualification. Five hours cannot produce equivalent
evidence about those questions. There is no defensible percentage of overall
assurance to assign. Much of the **deterministic software test coverage** can
be moved outside the rental window or exercised directly instead of waiting
for a training failure. Long-horizon learning and rare operational failures
still require exposure over time; even five days cannot prove a trillion-token
run will succeed.

For the user's assurance-first objective, prefer **one predeclared dense recipe
on all eight GPUs** to the two-recipe screen below. This buys a longer observed
trajectory but deliberately gives up comparative LR evidence. Treat the recipe
as an unvalidated candidate under the requested fresh-HPO assumption. A smaller
LR is not automatically safer overall: it can waste the budget by learning too
slowly. Do not retrospectively call a stable candidate optimal.

### Prepare the evidence before allocating the node

- Freeze model/optimizer semantics, one candidate, global batch, warmup,
  data mixture, evaluation sets and acceptance thresholds. Select a plateau
  horizon beyond the pilot with explicit stop bounds. Record any eventual
  production-fork requirements before training.
- Build and pin the image; stage representative, checksummed training data;
  audit tokenizer IDs, packed targets/masks, document boundaries, duplicates,
  held-out overlap and mixture weights. Run CPU tests and corrupted-artifact
  failure cases before paying for GPUs. ROCm imports, JIT, performance and
  hardware-dependent correctness still require the actual machine.
- Implement and CPU-test the missing bounded qualification runner, full-update
  comparison harness, trained-state reference probes, live backup and watchdog
  behavior. Predeclare numerical tolerances from independent controls; do not
  choose tolerances after inspecting the candidate's results. These additions
  are engineering prerequisites, **not functionality implemented by this plan**.
- Prepare bounded, machine-readable checks and one consolidated evidence
  report. Rehearse data cursor, optimizer/RNG restore, failed upload and
  incomplete-checkpoint handling on CPU. Keep generated artifacts outside Git.

### Five-hour node schedule

This timetable is a **conditional allocation**, not a promise that all native
checks finish within it. Failed or incomplete mandatory tests consume the cap
and block promotion. Freeze one FP8 backend and a conservative fitting
microbatch; do not add a backend/compile/graph/FP8-head sweep to this window.
Global batch stays fixed when adjusting microbatch and accumulation for fit.
The proposed starting batch remains **2,097,152 tokens/update**: at context
4096 and eight ranks, microbatch 16 with accumulation 4, or microbatch 8 with
accumulation 8. This pilot tests the chosen recipe at that batch; it does not
establish that 2M learns better than 1M. The initial microbatch still requires
native memory qualification.

| Wall-clock window | Work and required evidence |
| --- | --- |
| 00:00–00:40 | Device/topology health and per-device numerical smoke; prepared native kernel tests; warm full-model updates including optimizer-state creation; memory at validation/checkpoint cadence; same-global-examples one/eight-rank comparison. Require all gates to pass before the trajectory. |
| 00:40–04:10 | One full-model, eight-rank trajectory, through warmup and onto the plateau. Reserve 20 minutes inside this window for trained-state reference probes; use the rest for committed training and routine frozen held-out evaluations. |
| 04:10–05:00 | Complete the real eight-rank interruption/restart drill, independently download and verify an S3 checkpoint, compare replay against the saved uninterrupted reference, assess final held-outs and account for every GPU-hour. |

Required numerical coverage comes from section 6 of the
[full plan](FULL_QUALIFICATION_PLAN.md): FNUZ scales/rounding and tiny gradients,
forward/dgrad/wgrad at actual projection shapes, local/global causal attention
and GQA, logical spectral-Muon updates, tied embedding gradients, accumulation,
global loss normalization and clipping. One/eight-rank comparisons must use the
same global examples and account for legitimate floating-point reduction-order
differences. Require at least 15% of usable HBM free at the measured full-update
and cadence peaks, on every rank. Merely running a collective benchmark or
seeing a falling loss is insufficient.

At initialization, warmup exit and a later plateau checkpoint, use the prepared
harness to compare a few **identical-example updates from the same complete
state** against the BF16/reference path. Preserve optimizer moments, schedule,
RNG and data cursor; isolate probe branches from the committed trajectory.
Measure gradient direction/magnitude, clipping and per-role parameter deltas
and moments, rather than whole-weight similarity that can hide missing small
updates. BF16 is a precision control, not an exact oracle for FP8 rounding;
independent kernel/optimizer references and declared tolerances remain required.
These probes catch trained-state numerical errors but **do not replace** the
full plan's paired multi-billion-token FP8/BF16 learning comparison.

Exercise the backup path while training so the final drill measures the
intended provider network and concurrent upload behavior. Verify content hashes,
the committed checkpoint pair, next examples, counters, RNG, optimizer state and
replay deltas. Do not count a same-process local reload as an S3/distributed
recovery test. If checkpoint transfer or native testing takes longer, shorten
the learning window and report reduced evidence; never silently extend the cap.

### What fits at measured throughput

With 40 minutes for bring-up, 50 for final recovery/evaluation and 20 for special
reference diagnostics, a five-hour cap leaves **190 minutes** for ordinary
training. Rates here are measured all-in tokens/s per allocated GPU during that
ordinary training window, including routine evaluation, checkpoint and logging
overhead. They exclude separately charged reference/replay tokens, which do not
advance the selected model's committed token counter.

| Measured ordinary-trial tokens/s/GPU | Five hours: 190 training minutes | Three hours: 70 training minutes |
| ---: | ---: | ---: |
| 30k | 2.736B tokens | 1.008B tokens |
| 40k | 3.648B tokens | 1.344B tokens |
| 50k | 4.560B tokens | 1.680B tokens |
| 60k | 5.472B tokens | 2.016B tokens |

These are approximate capacity figures, rounded down to complete updates when
setting executable bounds, not MI325X speed predictions. With the proposed
1.0066B-token warmup, the five-hour/50k case observes about 3.55B plateau tokens;
the three-hour/50k case only about 0.67B. At 30k, the three-hour case barely
finishes warmup. Prefer five hours for this gate and report observed updates,
not just elapsed time. Do not shorten warmup after seeing slow hardware simply
to manufacture a passing learning curve.

### Which evidence is compressed, and which remains missing

| Question | Evidence available in a successful prepared five-hour pilot | Remaining limitation |
| --- | --- | --- |
| Are arithmetic, accumulation and distributed updates implemented correctly? | Direct reference checks, including trained checkpoints and all eight ranks | Only tested shapes, states and operating conditions; no absolute guarantee |
| Can this recipe learn without early numerical failure? | One complete warmup plus a measured plateau, held-out trends and role-level telemetry | No LR/batch comparison, independent seed or long-horizon optimality evidence |
| Can we recover this state and data position? | Actual rank interruption, verified S3 restore and replay | Not a multi-day reliability/rare-fault soak |
| Is FP8 preserving learning quality? | Kernel controls and brief matched-state update probes | Long paired FP8/BF16 trajectories remain outstanding |
| Is dense-to-ternary training ready? | No new qualification from this dense pilot | AMD QAT implementation, numerical checks, learning controls and actual late-parent tests remain mandatory |

After a pass, propose the next **bounded monitored dense stage**, with its
cost, remaining gates and evaluation decisions recorded. Production-like
continuations can accumulate useful tokens while providing longer evidence,
but only under the existing exact schedule/data contract or a tested explicit
fork. Preserve recoverable parents and use the full plan's predeclared
nonfinite/rank/data/backup hard stops and learning-review alerts. Monitoring and
checkpoints limit some loss; they cannot recover GPU-hours spent on a stable
but inferior recipe. The full fresh-HPO requirements are not marked complete
by this cheaper route. It is a conscious reduction in comparative evidence,
not the five-day study accelerated 24-fold.

## Alternative: prioritize a two-recipe comparison

This was the initial five-hour proposal. Use it only when comparing two rate
policies matters more than observing one recipe farther beyond warmup. It is
not additional work to fit inside the assurance-first schedule above.

The following schedule assumes a **measured all-in training-trial rate of
approximately 50,000 tokens/s per allocated GPU**. This includes in-trial
logging/evaluation overhead; the separate bring-up and restore windows are
charged separately. It is a scenario, not a measured MI325X forecast.

| Wall-clock window | Work |
| --- | --- |
| 00:00–00:45 | Native kernel/device checks, full-size memory/update and one/eight-rank comparison using the prepared harness. Only proceed if required checks pass. |
| 00:45–02:10 | Lower-rate complete dense recipe A, from initialization, to 2,013,265,920 tokens. |
| 02:10–03:35 | Higher-rate complete dense recipe B, same initialization/data/batch/schedule, to the same token endpoint. |
| 03:35–04:25 | Extend the more credible candidate by approximately 1.2B tokens; if ranking is unclear, report the tie and extend the predeclared default without declaring it safer or optimal. |
| 04:25–05:00 | Final held-out assessment, checkpoint/upload/download/restore check on eight ranks, accounting and the next-stage decision. |

Candidates are engineering screening points, not transferred winners:

| Candidate | Spectral Muon | Tied embedding/head AdamW |
| --- | ---: | ---: |
| A | 0.006 | 0.001 |
| B | 0.012 | 0.002 |

Keep norm/value LR 0.00015, Q/K LR 0.00005, matrix decay 0.02, embedding decay
0, momentum 0.95, Adam betas (0.9, 0.95), epsilon 1e-8 and clipping 1.0 fixed.
Use the same qualified FP8 implementation, the full 1.446B model, eight ranks
and **2,097,152 global tokens/update**. Choose a fitting microbatch at fixed
global batch. Use 1,006,632,960 warmup tokens and a plateau extending past the
pilot endpoint, with explicit update and wall-time bounds.

These compare two complete rate policies. Because Muon and embedding rates
both change, they do not isolate the optimum of either rate individually.
This pilot also does not compare 1M versus 2M batch, determine long-horizon
decay, or qualify QAT. Keep those questions explicitly open.

### Rate sensitivity for the two-recipe alternative

Each 2.013B-token trial takes `tokens / (8 × measured_rate)` seconds:

| All-in tokens/s/GPU | Minutes per trial | Two trials plus 80 minutes of checks |
| ---: | ---: | ---: |
| 30k | 139.8 | 5.99 hours: does not fit |
| 40k | 104.9 | 4.83 hours: almost no extension time |
| 50k | 83.9 | 4.13 hours: about 52 minutes for extension |
| 60k | 69.9 | 3.66 hours |

At 50k, three hours has about 2.4B training tokens available after the same
80-minute allowance. That supports one initial candidate with checks, not
the two-candidate comparison above. Parallel one-GPU jobs do not create more
aggregate GPU-hours and would provide shallower individual learning curves.

## Decision at the cap

- **Fail/block:** numerical error, invalid data, inadequate memory margin,
  failed recovery, or no useful learning. Preserve evidence and repair before
  more training. Do not retry an unsafe candidate until its cap is exhausted.
- **Promising:** required checks passed and a candidate has credible held-out
  improvement through warmup and plateau. Price a longer monitored dense
  continuation and the remaining qualification work. This is not a claim that
  its long-horizon LR or batch is optimal.
- **Inconclusive:** learning evidence is insufficient, compared candidates are
  tied, throughput was insufficient, or the
  operating checks used more time. Report the unresolved question and the cost
  of the next informative experiment. Do not substitute a shorter warmup or
  weaken a gate after seeing results simply to announce a winner.

Any continuation uses the correct existing schedule/data contract or the
tested explicit fork mechanism. Do not rebrand a short-decay pilot as a
trillion-token prefix. Passing this pilot permits consideration of the next
bounded stage; it does not clear an unattended month-long mixed campaign.
