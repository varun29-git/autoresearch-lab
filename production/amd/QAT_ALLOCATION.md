# Dense and ternary allocation for the MI325X month

Recommendation, 2026-10-25: plan **60% dense / 40% ternary QAT by training
tokens**, assuming the final ternary model is the primary deliverable. Keep
**80/20** as the main challenger if native recovery/quality evidence supports
a later switch. A 90/10 split is a conditional late-switch experiment, not
the default. This is an engineering starting policy, not an established
optimum or a passed dense-to-ternary transfer.

The new [phase_plan.py](phase_plan.py) computes both phases inside one budget.
It produces an arithmetic plan, **not a runnable QAT configuration**. Existing
AMD constructors and preflight still reject ternary mode. No restriction is
removed by selecting an allocation, and no paid campaign was launched.

## What changes at the switch

“FP8 versus ternary” combines two different concepts. The dense stage learns
ordinary projection weights using FP8 matrix operations. The proposed QAT
stage uses scaled ternary effective projection weights, with straight-through
gradients into FP32 master weights. Its GPU matrix operations can still use
FP8. Parameters, gradient accumulation and optimizer state remain floating
point during training; this is not packed 1.58-bit training state.

Preserve the current ternary reference: 128×128 mean-absolute scales,
round/clamp codes in {-1, 0, +1}, detached scales and an identity STE. The
1.295B projection parameters are eligible; the tied embedding/head and other
unquantized parameters remain outside that set. Keep the selected architecture,
tokenizer, attention masks, optimizer algorithm and logical matrix slicing.
Learned scales, changed ternary thresholds, a BF16 backward or removal of FP8
activation quantization are separate numerical candidates, not implied by
the duration recommendation.

The deployment artifact may be much smaller, but QAT still performs forward,
backward and optimizer work. It may have a similar rate to dense or lose
throughput to different weight quantization/cache behavior. Use a measured
QAT rate; neither bit width nor sparsity implies a large training speedup.

## What the evidence actually supports

| Evidence | Observation | Limit for CI-1 |
| --- | --- | --- |
| [ParetoQ, §2.1](https://arxiv.org/html/2502.02631v1) | Roughly 90/10 was strongest in the fixed 100B-token MobileLLM-125M allocation experiment; each phase had its own LR cycle | Smaller model; different quantizers, scale/gradient rules and training recipe. Not a universal 10% QAT rule |
| [Continual QAT, §4–5](https://arxiv.org/html/2502.11895v1) | On OLMo-1B over 10k updates (~41.94B tokens), switching at 2k steps beat later tested switches and ternary from scratch; retaining state/ramping smoothed shock | Shorter horizon, AdamW, different architecture/activation quantization; not a Muon or long-run scaling law |
| [MiniCPM4 / BitCPM4, §3.3](https://arxiv.org/html/2506.07900v2) | FP8→ternary pilots report loss approaching ternary-from-scratch training when QAT exceeds about 40% of tokens; 5M pilot, 150M verification; final QAT uses 350B tokens | The pilot comparison is against ternary-from-scratch, not proof of dense parity. Weight-only QAT and different models; percentages and “twice decay” do not transfer automatically |
| [BitNet b1.58 2B4T](https://arxiv.org/html/2504.12285v1) | Demonstrates successful native ternary pretraining at 2B parameters / 4T tokens | Does not establish the best late conversion point or cost for CI-1 |
| [Archived CI-1 Lean 150M review](../../experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md) | 10B dense + 2B QAT recovered from a large conversion spike but finished worse on every tracked source | Different architecture/optimizer rates; QAT mixture changed; no matched dense-continuation control. The raw run was not repeated here |

The local review reports English NLL 2.7597 at the dense parent, 10.1949 on
immediate ternary conversion, and 2.9857 after QAT. Under a fixed dense-mixture
weighting, final QAT was +0.1848 nats/token worse. That does **not** prove that
duration alone caused the gap; LR, architecture, quantizer and mixture are
confounded. It does show why “a few final QAT steps will preserve quality” is
an unsafe assumption for this project.

The papers disagree because their experiments differ. Choosing 60/40 reserves
substantial representation-learning time under the actual ternary constraint.
It does not promise that 40% always recovers dense quality, nor that more QAT
is automatically better. At 1T total tokens it means **600B dense + 400B QAT**;
at 800B it means **480B + 320B**. Both stages count toward the same total.
Do not plan 1T dense and then add “40% QAT” outside the allocation.

## Token shares, time shares and the 5,760-hour limit

For dense rate `Rd`, QAT rate `Rq`, and QAT token fraction `q`:

```
productive GPU-seconds = Td / Rd + Tq / Rq
effective tokens/s/GPU = 1 / ((1 - q) / Rd + q / Rq)
QAT share of production time = (q / Rq) / ((1 - q) / Rd + q / Rq)
```

Rates must be complete-update per-GPU rates on the actual eight-rank node.
Communication and optimizer work are already inside them. The weighted
arithmetic mean of the rates is wrong for a token-fraction allocation.

For the combined program, budget **480 GPU-hours for qualification**, including
the previous dense qualification envelope and an additional 160 GPU-hours
for QAT port validation/recovery work. This is a total reserve, **not 480 plus
the old 320**. It leaves 5,280 GPU-hours (27.5 node-days). Applying the same
initial 90% production duty assumption leaves 4,752 productive GPU-hours.
The remaining 2.5 node-days cover the qualification reserve. Unspent reserve
can return to production; charge actual elapsed time before fixing the horizon.
If qualification overruns, reduce the horizon rather than invent extra hours.

These are **sensitivity scenarios**, not new estimates of measured AMD speed:

| Dense rate/GPU | QAT rate/GPU | Total tokens | Dense tokens | QAT tokens |
| ---: | ---: | ---: | ---: | ---: |
| 55k | 55k | 941B | 565B | 376B |
| 55k | 45k | 864B | 518B | 346B |
| 55k | 35k | 766B | 460B | 306B |
| 80k | 65k | 1.253T | 752B | 501B |

At 55k/45k, QAT's 40% token share takes about **44.9% of production time**:
approximately 15.15 node-days dense, 12.35 QAT, plus 2.5 qualification days.
At those same rates, 80/20 fits about 901B tokens and 90/10 about 920B. Their
higher token count does not establish a better final ternary model. Compare
quality under the fixed allocation, not just total token exposure.

The calculator rounds down to complete global updates while preserving the
exact chosen token ratio. It does not silently truncate a real manifest or
choose dataset mixture phases. Generate a real data plan only after the
measured rates and duration policy are accepted.

```bash
# Arithmetic example only; replace both rates with native measurements.
python3 -m production.amd.phase_plan --dense-rate 55000 --qat-rate 45000 \
  --qat-percent 40 --output /external/mi325x/dense-qat-plan.json
```

This command's JSON is a planning artifact, not input to `train.sh`.
Use `--qat-percent 20` for the challenger. For a fixed 2T feasibility check,
the existing calculator can price the 1.2T dense / 0.8T QAT workload:

```bash
python3 -m production.amd.budget --dense-rate 55000 --qat-rate 45000 \
  --target-tokens 2000000000000 --qat-tokens 800000000000 \
  --reserve-gpu-hours 480 --duty-cycle .9
```

All-in phase rates that already include cadence should use duty cycle 1 for
those same overheads. Checkpoint transfer, conversion, evaluation and downtime
are not free. SFT, alignment and extensive deployment benchmarking are not
additional token phases already funded by this dense/QAT allocation.

## Coordinate the two learning-rate schedules

The existing `balanced` profile is a **dense-stage** configuration. Its
manifest horizon should be the chosen `Td` once a mixed campaign is frozen.
For the initial comparison, complete its selected WSD schedule, including
the final 10% dense decay, and preserve that dense checkpoint as a useful
fallback. Then use a separate phase-relative QAT schedule over `Tq`. This
supports comparison to the dense parent and follows the two-cycle approach
tested by ParetoQ; it is not a proof that a flat dense-to-QAT switch is worse.

Suggested QAT starting candidate for the later native configuration builder:

- Keep global batch 2,097,152, world size 8 and initially microbatch 16 /
  accumulation 4, provided the QAT memory test passes. Preserve initialization
  lineage, optimizer moments, RNG and the exact next-token cursor.
  If the dense 3,145,728-token batch is promoted, review the QAT batch and
  schedule separately; dense speed alone does not qualify its QAT transfer.
- Switch effective projection weights to full ternary immediately. Use LR
  re-warmup, not an unvalidated gradual weight/activation interpolation.
- Re-warm over `min(1,006,632,960 tokens, 5% of Tq)`, rounded to full updates;
  plateau, then cosine decay over the final **20% of the QAT phase** to 0.1×
  its peak. The extra decay share is a QAT candidate informed by the local
  recovery curves, not a change to the selected dense schedule.
- Start the QAT peak-rate comparison at **0.5× dense peak base rates**:
  Muon 0.0085, embedding AdamW 0.0015, norms 0.00015 and Q/K norms 0.00005.
  Retain the algorithm, momentum/betas, weight decay and clipping. These are
  candidate rates for testing; the dense HPO does not certify them for QAT.

Use the dense **peak base** rates as the reference, not its final decayed LR.
A half-peak QAT rate is five times the dense terminal rate when dense ends at
0.1× peak. This makes the explicit re-warmup and transition diagnostics
necessary. Reusing the dense terminal LR or applying global-token decay to
the child phase would create a different schedule.

The child `phase_start_tokens` must equal the parent's consumed tokens for a
continuing stream, and its total manifest horizon must cover `Td + Tq`.
Protect exact-resume identity within each phase. Check that “continue” really
preserves the same source sequence, mixture and offsets; a matching tokenizer
and numeric cursor alone do not prove prefix identity. That audit is needed
before native promotion. A changed QAT data mixture should be tested separately
after the initial same-mixture comparison, rather than confounding conversion.

## Decide early enough to preserve the ternary budget

Finish the native QAT path and test recovery **before starting the month-long
production campaign**. Do not first discover at the 60% checkpoint that the
kernel, transition or data plan cannot run.

Within the additional 160 GPU-hour QAT allowance, use one full-architecture
trained dense parent around 10B tokens, then bounded ~4B QAT and matched ~4B
dense-continuation branches. Match data, LR schedule, optimizer settings and
evaluation coverage between those continuation branches. Evaluate immediately
after conversion, through re-warmup, before decay and after decay. A real
compatible parent from dense qualification can be reused; count its cost
once only if checkpoint, source, runtime, initialization and history match.
Otherwise include its full cost. At the illustrative 55k/45k rates, the
10B parent + both 4B branches costs about 106 allocated GPU-hours at 90% duty,
leaving about 54 of the extra hours for native checks and recovery overhead.
This estimate must be replaced with measured costs before launch.

This is a recovery test, **not a matched-total-token proof of 60/40 versus
80/20**. A real allocation comparison needs different dense switch points,
fully completed child schedules and equal total exposure or measured compute.
Short probes and smaller proxy models cannot settle the long-run fraction.
Branches only share computation while their histories and schedules are
identical; dense stages with different scheduled decay starts cannot share
arbitrary checkpoint prefixes under the existing exact-resume contract.

Retain a checkpoint around 40% of the eventual planned token horizon for an
earlier-switch contingency, then schedule the intended switch at 60%. A new
branch consumes allocation time, and discarded dense work is not free. Use
the real long-run parent to confirm recovery before committing the full QAT
tail. Moving to 80/20 needs convincing evidence before spending the reserved
QAT time on extra dense updates. If transfer is poor, diagnose the quantizer,
precision path and LR rather than assuming extra QAT tokens alone will fix it.

The primary metric is final ternary quality by language/domain and target
tasks. Compare against both the dense parent and a matched dense continuation.
The previous 0.02-nat execution-comparison guardrail is not evidence that
ternary will reach that gap. Report the actual gap and uncertainty; do not
silently relax a failed threshold or describe lower NLL as generated accuracy.

## Required native work, still outstanding

1. Implement/test gfx942 ternary weight codes/scales, both matrix orientations,
   padding/nonfinite handling, and cache refresh against the existing STE
   reference. Forward/dgrad must use ternary effective weights; latent wgrad
   must retain the selected FP32-output/full-backward semantics.
2. Add explicit AMD QAT recipe/preflight/launcher support and then pass it.
   Do not just remove `weight_mode == dense` guards or silently substitute
   a BF16 backend. Existing transition checks also reject a backend change.
3. Audit parent/child data-prefix equivalence, phase counters, carried optimizer
   rates/state and exact within-QAT resume on the actual eight ranks.
4. Measure complete QAT updates, peak memory and cadence; stage enough data
   for both phases with unchanged initial mixture and held-outs. Repeat dense
   gates affected by any shared implementation change.
5. Demonstrate full-model conversion/recovery, durable restore and final
   packed-export parity. Keep the dense checkpoint until ternary quality is
   accepted. FP8 and ternary state checks are separate from deployment speed.

No QAT model constructor or training behavior is changed in this allocation
work. Source citations were inspected on 2026-10-25; all performance examples
above are arithmetic scenarios and all new AMD QAT gates remain unrun.
