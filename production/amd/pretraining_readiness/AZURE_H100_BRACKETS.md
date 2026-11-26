# Focused batch transfer on one Azure H100

**Latest refinement:** the user requests lower published-rate controls as
well as batch transfer. Follow [LR_BATCH_MONITORING.md](LR_BATCH_MONITORING.md)
for the current LR comparison on the 194M width-scaled proxy at fixed 2M global
batch, monitoring contract and subsequent 1M check. Transfer the winner for
one short full-size confirmation. The latest proxy horizon is **5B per trial**;
the optional extra batch comparison is deferred to preserve the 50-hour ceiling.
The two approximately 2B trials and
conditional decay-only ordering below are superseded as immediate work;
their source review and batch arithmetic remain relevant.

## Current decision: reuse completed HPO, test the batch change

Revision, 2026-10-28. The user asks to use the completed tuning and published
Muon runs rather than repeat the bracket search. The external 200M ledger has
12 completed trials and the 700M ledger has six; all 18 copied event files
match their ledger hashes. They already include Muon, embedding and decay
comparisons and width-transfer controls. This supersedes the broad proxy
search below as the immediate use of the available **50 H100 hours**.
The actual ledgers, evaluations and reports remain outside Git under
`/Users/varundaiya/terminus-optimizer-evidence/hpo-results/`.

The evidence supports reusing the selected optimizer family and rates as the
first candidate. It does not establish a new 24-layer production optimum:
the old geometry, 393,216-token batch, English scope and approximately 500M
observed tokens differ. The purpose of the next experiment is to challenge
the changed batch on the actual model, not rediscover the original LR bracket.

### Eight GPUs do not require a larger global batch than the H100 test

The proposed **2,097,152 tokens/update is the total over all eight GPUs**.
It is 5.333 times the old HPO global batch, not eight times a 2M per-GPU batch.

| Job | Per-rank microbatch | Context | Accumulation | Global tokens/update |
| --- | ---: | ---: | ---: | ---: |
| One H100, conservative fit candidate | 4 | 4096 | 128 | 2,097,152 |
| Eight MI325X, fit candidate | 16 | 4096 | 4 | 2,097,152 |
| Eight MI325X, smaller microbatch | 8 | 4096 | 8 | 2,097,152 |

Microbatch fit is unmeasured on the actual nodes. Correct accumulation and
normalization let the H100 test the intended global batch; they do not test
AMD arithmetic, communication or reduction ordering. At H100 microbatch 4,
accumulation 64 gives the 1,048,576-token control. No GPU-count LR factor is
applied when moving to eight ranks at the same global batch.

### Published Muon evidence used for the decision

- [Moonlight, Table 2](https://arxiv.org/html/2502.16982v1): its 1.1B and 1.5B
  non-embedding scaling models use 192 and 256 examples at 8192 context,
  respectively: 1,572,864 and 2,097,152 tokens/update, over 28.54B and 38.91B
  training tokens. This is direct precedent for the batch scale, not proof
  of CI-1's LR, data efficiency or final quality. Its RMS-matched Muon and
  auxiliary AdamW coefficients cannot be copied into spectral Muon.
- [Kimi K2, section 2.5](https://arxiv.org/html/2507.20534v1): MuonClip trains
  with a 67M-token global batch and 15.5T-token main horizon. This establishes
  large-batch Muon-family feasibility, but the much larger MoE, MLA and QK-Clip
  make its batch and LR unsuitable numerical targets for our dense model.
- [Dang, Wen and Malladi, v2, sections 3.2/E.3](https://arxiv.org/pdf/2610.08975v2):
  216 MuonW/AdamW transfer rules were tested from 131,072 reference tokens to
  targets through 2,097,152. The best common language-model rule kept both
  LRs, Muon momentum and Adam beta1 fixed, scaled matrix decay linearly with
  batch, and transported Adam beta2. Keeping beta2 fixed was also competitive.
  Square-root LR scaling won at some smaller targets; there is no universal
  rule. The reference loss was 3.26547 and the best tested 2M transfer-rule
  loss was 3.37977 at the same approximately 1.7B-token budget, so the selected
  rule did not preserve small-batch sample efficiency. These are published
  experimental results, not CI-1 measurements. Its BF16, 12-layer/width-768,
  1024-context, NS12 model and 80% cooldown differ substantially from ours.

All three primary sources were rechecked for this revision. The Muon batch
paper's downloaded PDF/text stays under `/tmp/ci1-batch-review/`, outside Git.
An earlier cached HTML request for that paper was only a no-HTML notice;
the evidence above was checked in the actual v2 PDF.

### Two core comparisons; one conditional correction

Use the **actual 1,446,478,126-parameter production geometry** for the first
pair, if native H100 full-update fit passes. This focuses the available compute
on the remaining production-width/batch question. The smaller proxy remains
an optional diagnostic if a specific failure needs cheaper investigation.

| Trial | Global batch | Matrix decay | Purpose |
| --- | ---: | ---: | --- |
| A | 2,097,152 | 0.02 | Test the intended batch with the existing production starting recipe |
| B | 1,048,576 | 0.02 | Test whether twice as many updates per token materially improves learning |
| C, conditional | 2,097,152 | 0.1066667 | If A loses to B, test the batch-scaled decay hypothesis before changing LRs |

Keep Muon 0.017, tied embedding/head AdamW 0.003, norm/value 0.0003, Q/K
0.0001, momentum 0.95, Adam betas (0.9, 0.95), epsilon 1e-8, embedding decay
zero and clipping 1.0 fixed in the pair. This preserves the code's reviewed
starting rates; it does not claim that AdamW 0.003 is the strict width-law
prediction. The 0.002 alternative remains a focused follow-up if embedding
diagnostics or width behavior justify it, not another mandatory grid.

C's proposed decay is `0.02 * 2,097,152 / 393,216`. It approximately preserves
the nominal decay action per token at fixed LR relative to the HPO batch.
It is an independently reviewed experimental override, **not an automatic
production change or a guarantee**. Adam beta2 transport and new momentum
rules are not bundled into C; the paper's long beta2 memory differs from ours.

Each core trial runs 2,013,265,920 tokens from the same initialization seed
and ordered representative data. Use a complete token-scaled WSD with 10%
warmup, 70% stable and 20% cosine decay to 10% of peak. A has 960 updates
(96/672/192 across the segments); B has 1920 (192/1344/384). Record held-outs
before decay and at the endpoint. These are short-horizon comparisons, not
production-prefix schedules. An ended cooldown cannot be resumed as though
it were the long production plateau.

Freeze held-outs and a decision margin before launch, compare paired losses
with document-level uncertainty, and inspect per-domain loss, clipping,
logical update statistics and FP8 incidents. A falling training loss alone
does not establish good transfer. Compare at matched tokens and measured
allocated GPU-hours. Prefer 2M if its learning is acceptable within the
predeclared margin and it offers a useful execution advantage; otherwise
retain 1M or investigate C. H100 timing cannot decide the final AMD timing
tradeoff. Native failures take precedence over an endpoint ranking.

At a measured effective full-model rate of 40k tokens/s, A+B take **27.96
H100 hours**, and A+B+C take **41.94 hours**, leaving approximately eight
hours in the 50-hour cap for setup, checks and overhead outside that rate.
At 30k, A+B take 37.28 hours and all three take 55.92 hours, so C does not
fit before overhead. At 50k the corresponding times are 22.37 and 33.55
hours. Measure rates separately for the two batches and sum their costs;
these common-rate figures are scenarios, not new Azure benchmarks. C and
other follow-ups run only if their measured cost fits the remaining cap.

The human controls paid launches. The bounded qualification runner and native
preflight remain prerequisites. AMD kernel/FP8, eight-rank correctness, full
memory cadence, recovery, longer learning and QAT gates are not cleared by
this H100 comparison. The narrower plan reduces repeated HP work; it does
not claim that the 1T run is already qualified.

## Earlier proxy-search option — superseded as the immediate next step

The following earlier proposal is retained for reference if new evidence
justifies reopening a broad search. Its six proxy trials and budget split
are not added to the focused 50-hour plan above.

Engineering proposal, 2026-10-28. The user has up to **50 H100 hours** available.
Use that capacity to narrow the AMD recipe before renting the eight-MI325X
node. This supersedes the earlier requirement to run every selection-bearing
trial at full size on eight GPUs. Final AMD distributed qualification remains
required. No GPU campaign has been launched and no new HP result is claimed.

The exact Azure SKU, usable HBM, power limit and billing granularity are still
unknown. Budget below assumes one allocated/billed H100 for each elapsed hour;
an eight-card VM with seven idle cards would not meet that accounting. Record
Azure use separately from the 5,760 AMD GPU-hour allocation unless the user
specifies that the budgets are shared. The human launches and stops paid work.

## What is an educated guess here?

Historical HPO supplies a reason to choose the center of a bracket, not a
preapproved winner. The archived candidate used spectral Muon 0.017 and tied
embedding/head AdamW 0.006 at width 768. The recorded width-1536 comparison
favored embedding LR 0.003 over 0.00424 and 0.006. The current production
builder retains 0.003; continuing strict inverse-width scaling would instead
give 0.002 at width 2304. These are competing starting hypotheses.

Keep the actual spectral optimizer equation. Do not simultaneously replace it
with Moonshot RMS matching, change the context to 8192, set matrix decay to 0.1
or collapse the norm/embedding LR groups. That would mix several experiments
and invalidate the interpretation of these brackets. All settings below are
candidate specifications; production defaults are not changed by this document.

## Build the proxy from production, not the archived HPO configuration

The archived proxy has 25 layers, a 1024-token local window and a global layer
every fifth layer. Production has 24 layers, a 512-token local window and a
global layer every sixth layer. Start from `production.recipe.SELECTED_MODEL`
and change only the four width-family fields below.

| Field | H100 search proxy | Optional intermediate | Production confirmation |
| --- | ---: | ---: | ---: |
| Width | 768 | 1536 | 2304 |
| FFN width | 1920 | 3840 | 5760 |
| Query / KV heads | 6 / 2 | 12 / 4 | 18 / 6 |
| Parameters, CPU meta-device count | 194,329,390 | 676,486,702 | 1,446,478,126 |

All use 24 layers, head dimension 128, context 4096, the same 65,536 vocabulary,
tied embeddings, Peri-LN, learned Q/K gains and value residual. Preserve the
selected initialization, residual scale and projection/head precision rules.
The 676M model is optional; do not require it at the expense of the final
production-width check. CPU counts do not establish native fit or speed.

Use representative production-mixture data and frozen, disjoint held-outs.
The old English-only results cannot certify all intended domains. Compare
complete recipes using matched initialization seeds, token prefixes and
evaluation records; different widths have their own shape-appropriate initial
weights. Store manifests, generated configs, evaluations and results outside Git.

## Initial brackets and a small interaction check

At proxy width 768, start at global batch **2,097,152 tokens/update**:

| Trial | Spectral Muon LR | Tied embedding/head AdamW LR | Question |
| --- | ---: | ---: | --- |
| C | 0.017 | 0.006 | Does the historically informed center learn on the new geometry/batch? |
| M-low | 0.0085 | 0.006 | Does halving Muon improve held-out learning or numerical behavior? |
| M-high | 0.034 | 0.006 | Does doubling Muon help enough to justify the larger updates? |
| A-low | 0.017 | 0.003 | Is the tied embedding/head rate too aggressive at the center? |
| A-high | 0.017 | 0.012 | Is the center under-updating the tied embedding/head? |

These are log-spaced experimental brackets, not assertions that high/low
endpoints are safe or that the optimum lies inside. Nonfinite or otherwise
failed trials stop under predeclared rules; do not run an unsafe endpoint to
its token cap. Low LR can also fail the useful-learning objective.

The sixth trial follows a predeclared rule: if both axes prefer noncentral
values, test that unmeasured combination from initialization. If no new joint
combination is indicated, repeat the current leader with a second seed. This
is a sparse screen, not a complete interaction study. If a bracket edge wins
or comparisons remain tied, report the unresolved direction and price the
next trial; do not declare an optimum or spend beyond the cap automatically.

Hold other groups at the reviewed baseline for this first question: ordinary
norm/value LR 0.0003, Q/K LR 0.0001, matrix decay 0.02, embedding decay zero,
momentum 0.95 with the existing Nesterov implementation and NS5, Adam betas
(0.9, 0.95), epsilon 1e-8, global gradient clip 1.0. These fixed values are
controls, not newly qualified optima. A failure in one of these groups can
make the LR screen inconclusive rather than proving Muon/embedding is wrong.

## Complete token-scaled proxy schedules

Each core trial runs **1,006,632,960 tokens**, exactly 480 updates at 2M:

| WSD segment | Tokens | Updates at 2M |
| --- | ---: | ---: |
| Warmup | 100,663,296 | 48 |
| Stable peak | 704,643,072 | 336 |
| Cosine decay to 10% of peak | 201,326,592 | 96 |

This is a deliberately complete short-horizon experiment. Rank primarily by
frozen held-out token-mean NLL at the final matched endpoint; also record the
pre-decay endpoint, critical-domain losses, gradient/clipping behavior and
per-role update statistics. Predeclare the evaluation method and tie handling;
use paired document-level uncertainty rather than treating correlated tokens
as independent samples. One seed does not establish seed robustness.

If spare proxy budget permits, compare 1M versus 2M batch with the leading
complete recipe. The 1M comparison has 960 updates and the same token schedule.
Initially preserve LRs to isolate the batch change; any square-root-scaled LR
is a separate candidate. Proxy batch ranking does not establish the production
model's critical batch. Defer 4M/8M unless measured learning or execution gives
a reason to add them. A proxy run at 4M has only 240 total updates here.

On one GPU, microbatch 8 at context 4096 with accumulation 64 gives the 2M
batch; microbatch 4 with accumulation 128 gives the same batch. Measure a fitting
microbatch and full-update speed; do not inherit MI325X microbatch 16 blindly.
The resulting averaged gradient is intended to match the global examples,
subject to floating-point ordering and correct normalization. It does not test
eight-rank collectives. Larger accumulation alone does not guarantee higher MFU.

## Transfer candidates, then challenge them at production width

Use the existing transfer generator's **alpha=1** as the initial embedding
mapping hypothesis: `eta(width) = eta(768) * 768 / width`. Keep the spectral
Muon base rate and auxiliary group rates unchanged. Thus proxy AdamW values
0.003 / 0.006 / 0.012 map to **0.001 / 0.002 / 0.004** at production width.
This is not a new guarantee of full muP, and the old 0.003 production value
is inside this bracket rather than its automatically selected center.

Plan two fresh full-size, approximately 1B-token H100 comparisons. If proxy
ranking is unresolved, compare the top two mapped complete recipes. If one
recipe clearly leads, compare its strict alpha=1 embedding rate against
1.5 times that rate with the same Muon and auxiliary settings. At the historical
center this is exactly 0.002 versus the retained 0.003. This tests a specific
transfer ambiguity, not every possible width law. Preserve the same global
batch, token-scaled schedule and held-outs within the comparison.

Before these comparisons, measure full-update fit including optimizer state,
validation and checkpoint peaks. Keep at least 15% usable HBM headroom. Start
with a small microbatch and accumulate to the required global batch. If fit
requires a correctly tested checkpointing mode, record it and its time cost.
Do not assume an Azure H100 has the memory or throughput of the old NVL card.

The confirmation output is a narrow dense recipe shortlist and its measured
limitations. It is not a proof of trillion-token LR optimality. A short-decayed
H100 checkpoint is not silently relabeled a production prefix. Cross-vendor
or schedule-changing continuation would need a tested explicit fork; transferring
the recipe alone does not require transferring the checkpoint.

## Fifty-hour envelope and throughput-dependent decisions

| Allocation | Maximum H100 hours | Evidence sought |
| --- | ---: | --- |
| Native setup, numerical checks, fit and rate measurements | 4 | The intended H100 path executes correctly enough to begin bounded learning trials |
| Proxy screen and, if affordable, batch/seed follow-ups | 22 | Bracket direction, interaction warning and a short-horizon ranking |
| Full-size fit/control overhead and two mapped comparisons | 18 | Does the shortlisted recipe remain useful at actual width? |
| Contingency / most informative unresolved question | 6 | Repair/repeat a gate or resolve an ambiguity without exceeding the total |
| **Total ceiling** | **50** | No obligation to spend unused hours |

Prepare data/image and qualification code before this clock where possible.
Hardware checks can consume more than their window. Reallocate inside 50 hours
or stop with incomplete evidence; never drop a required check to fit a table.
Actual batch comparison and seed/precision controls must be charged, not
described as free additions. If rates are slow, reserve the production-width
comparison budget before adding optional proxy trials.

Six core proxy trials process 6.0398B tokens. Two full-size trials process
2.0133B. Illustrative measured-effective-rate scenarios, not benchmarks:

| Work | Effective rate on the one H100 | Training-trial time |
| --- | ---: | ---: |
| Six proxy trials | 75k tokens/s | 22.37 hours |
| Six proxy trials | 100k tokens/s | 16.78 hours |
| Six proxy trials | 150k tokens/s | 11.18 hours |
| Six proxy trials | 200k tokens/s | 8.39 hours |
| Two full-size trials | 30k tokens/s | 18.64 hours |
| Two full-size trials | 40k tokens/s | 13.98 hours |
| Two full-size trials | 50k tokens/s | 11.18 hours |

Rates must include ordinary evaluation/checkpoint overhead; initial setup,
special reference diagnostics and recovery probes are charged separately.
The historical approximately 41k NVL figure describes a different 25-layer
full model, not this Azure proxy. A prior expectation that the proxy is faster
is reasonable, but dividing parameter counts is not a throughput measurement:
attention, the fixed vocabulary/head, kernel sizes and fixed overhead matter.
Measure both proxy and full-size throughput early enough to revise the budget.

## What the H100 evidence cannot establish

- AMD FNUZ FP8 representation, ROCm kernels and their trained-state behavior.
- Eight-MI325X memory cadence, RCCL/xGMI performance, rank loss and S3 recovery.
- A trillion-token learning optimum or final dense-to-ternary phase allocation.
- AMD QAT implementation, conversion, late-parent quality or training throughput.

Transfer the shortlist to AMD with an FP32/BF16 reference path and controlled
same-state updates; expect format/reduction differences, not bitwise H100/AMD
identity. Run the full-size eight-rank learning and operating gates on AMD.
Falling H100 loss alone cannot qualify another backend. The resulting AMD
gate may be shorter because fewer HP questions remain; its duration is still
conditional on measured tests, not guaranteed to be five hours.

## Engineering status and sources

This is a plan. `hpo/loop.py` explicitly locks width-768 runs to 1,272 updates,
393,216 tokens/update and its old reference. The production builders lock the
selected architecture/rates. Neither is an unchanged ready-to-run launcher for
this proposal. Implement a separate bounded qualification path with explicit
candidate identity, external artifacts, arbitrary reviewed token milestones,
native preflight and budget enforcement. Do not loosen archived campaign locks
or production acceptance globally. Record data/loss normalization and schedule
boundary CPU tests before native trials. Paid work remains human-controlled.

Repository evidence: [production architecture](../../ARCHITECTURE.md),
[selected recipe](../../recipe.py), [historical candidate](../../../hpo/candidate.json),
[historical reference](../../../hpo/reference_200m.json),
[transfer code](../../../hpo/transfer.py),
[spectral semantics](../../../docs/MUON_SPECTRAL_IMPLEMENTATION.md),
[historical NVL timing](../../../docs/speed/TRANSFER_32GPU.md), and the
[full qualification controls](FULL_QUALIFICATION_PLAN.md).
