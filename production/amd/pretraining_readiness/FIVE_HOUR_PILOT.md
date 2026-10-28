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

## Concrete five-hour schedule

The following schedule assumes a **measured all-in training-trial rate of
approximately 50,000 tokens/s per allocated GPU**. This includes in-trial
logging/evaluation overhead; the separate bring-up and restore windows are
charged separately. It is a scenario, not a measured MI325X forecast.

| Wall-clock window | Work |
| --- | --- |
| 00:00–00:45 | Native kernel/device checks, full-size memory/update and one/eight-rank comparison using the prepared harness. Only proceed if required checks pass. |
| 00:45–02:10 | Conservative complete dense recipe A, from initialization, to 2,013,265,920 tokens. |
| 02:10–03:35 | Moderate complete dense recipe B, same initialization/data/batch/schedule, to the same token endpoint. |
| 03:35–04:25 | Extend the more credible candidate by approximately 1.2B tokens; if ranking is unclear, extend the safer interior choice without declaring it optimal. |
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

## Rate sensitivity and the three-hour alternative

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
- **Inconclusive:** candidates are tied, throughput was insufficient, or the
  operating checks used more time. Report the unresolved question and the cost
  of the next informative experiment. Do not substitute a shorter warmup or
  weaken a gate after seeing results simply to announce a winner.

Any continuation uses the correct existing schedule/data contract or the
tested explicit fork mechanism. Do not rebrand a short-decay pilot as a
trillion-token prefix. Passing this pilot permits consideration of the next
bounded stage; it does not clear an unattended month-long mixed campaign.
