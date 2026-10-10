# Dense HPO loop after optimizer selection

**Objective:** Find the best attainable dense hyperparameters for the fixed
25-layer Peri-LN + learned-value-residual Stack 4 model. Use one H100 NVL and
at most **8 allocated GPU-hours total for the 200M campaign**, including
startup, validation, and any idle time while the GPU remains allocated. The
human launches and stops the paid GPU. Kimi runs consecutive full experiments
within that cap, then reports the best measured recipe. This is dense FP8
only; ternary QAT is a later separate campaign.

Before allocating the GPU, read the factual `MODEL_PROFILE.md`, then perform
**your own deep research** into first-party papers and released training
code. Investigate Moonlight/Kimi, MiniCPM, MiniCPM4, Qwen3, Gemma 3, Gemma 4,
and any other production implementation you judge important. Find the
optimizer assignment, effective Muon and AdamW group LRs, scaling
conventions, weight decay, momentum/betas, batch, and schedule where a source
actually discloses them. Mark missing information as missing. You, not this
framework, must decide which rates, group splits, and decay values deserve
experiments for this model. Keep a compact source-backed note and candidate
rationale. If the evidence calls for a different optimizer-group partition or
decay field, implement and test it **before the first trial**, then freeze the
campaign source. Prepare the first candidates before GPU allocation.

The source is yesterday's selected E3/E4/E9 speed stack. The 200M reference
in `reference_200m.json` fixes its model, prepared English token stream,
393,216-token global update, seed, validation examples, the archived 2.5B-token
WSD schedule, and scorer. Every HPO candidate stops at its first ~500M tokens;
the schedule's decay starts near 2B tokens and is not observed here. Before
starting, incorporate the optimizer winner from the separate Hyperball
comparison. If spectral Muon wins, keep the current `spectral_unclamped`
implementation; research and tune its weight decay. If Hyperball wins,
merge its tested implementation and set the reference rule to `hyperball`.
Do this **once**, before the first HPO trial. The same printed LR has a
different meaning under the two rules.

One loop iteration is: make a specific hypothesis, edit `candidate.json`,
commit it, run the candidate through **1,272 updates = 500,170,752 tokens**,
read its held-out English cross-entropy curve, keep or revert, then choose
the next idea. Every valid 200M candidate must complete this full comparison;
only an actual failure (OOM, nonfinite loss, or the hard eight-hour cutoff)
may end early, and an incomplete run cannot be ranked. The editable HPO
fields are the global Muon LR; optional packed-QKV, attention-O, gate/up, and
down Muon LRs; tied embedding/readout Adam LR and its own weight decay;
input-norm LR, Peri-LN gain LR, final-norm LR, Q/K gain LR, learned-value-mix
LR; matrix decay, Muon momentum, Adam betas/epsilon, and global clipping.
Unset role-specific rates inherit the candidate's global rate. The model, data,
effective batch, warmup, decay, and scorer stay fixed. Use the same trial name
only to **resume the same candidate and code**; use a new name for a different
recipe. The runner checks that condition.

The 250/500-update validations are inside the 250M-token warmup and cannot
select a winner. The 750/1,000-update validations are useful trajectory
diagnostics, but **rank only the 1,272-update endpoint** in the stable-LR
phase. These runs do not measure the later decay or long-horizon norm drift.
At the archived 143.5k tokens/s, one complete run would take about 58 minutes
before setup and validation; the optimized 200M
speed is unmeasured. Eight hours is a cap, not a promise of eight candidates.
Before each new candidate, use the last complete run's real allocation time
to decide whether a full 500M-token run still fits. Keep the GPU busy with
ready candidates; deallocate when another complete run cannot fit. Report
the full CE curve, tokens/s, clipping, FP8/stability incidents, elapsed
allocated GPU time, and any incomplete run separately.

Choose the initial Muon LR, tied-table AdamW LR, matrix weight decay, and
embedding weight decay from your research and the model profile; write all
four explicitly in every candidate. Decide which optimizer groups need
distinct rates from measured CE curves and update/weight ratios. Design high-information experiments that
fit the time cap; coupled group changes are allowed when the hypothesis
requires them. **Optimizer weight decay operates during every one of these
500M-token trials**, even though the WSD learning-rate decay happens much
later. Test matrix and tied-embedding weight decay empirically at matched
learning rates where the budget allows; do not call either tuned from a
single uncontrasted value. Consider the effect of `LR × weight_decay` when
interpreting a result. The framework does not prescribe numerical rates,
weight-decay values, trial order, or a fixed list of group splits. An
unsuccessful full candidate is evidence.

Example on the human-provided H100, with the exact archived manifest staged
on its persistent disk:

```bash
python -m hpo run --trial researched_candidate_01 --updates 1272 \
  --manifest /external/stack004-manifest.json \
  --data-root /srv/terminus-10b/data/packed \
  --results-root /external/hpo-results

python -m hpo rank --updates 1272 --results-root /external/hpo-results
```

The reference requires the exact manifest SHA-256. Training still checks the
real prepared shards and performs full validation and checkpoint work. The
one-GPU research runner skips only the older native preflight **report**.
Do not burn paid time on a new profiling campaign or repeat long preflights.

## Transfer before production

The model family keeps 25 layers and head dimension 128: width 768 has
6 Q / 2 KV heads and FFN 1920; width 1536 has 12 / 4 and FFN 3840; width
2304 has 18 / 6 and FFN 5760. A bridge changes only width-related geometry
and a memory-fitting microbatch. Preserve the **393,216-token effective
batch**: for microbatch sizes 32, 8, 4, or 2 at sequence length 4096,
use accumulation 3, 12, 24, or 48 respectively. Keep the same ordered
training examples and 32 fixed held-out sequences by adjusting the number
of validation batches. Each bridge gets a separately hashed manifest view
and reference file; the rank command never mixes those with the 200M stream.
Prepare each bridge **once**, outside Git, from the fixed manifest (for
example, batch 4 for width 1536 and batch 2 for width 2304):

```bash
python -m hpo.bridge --manifest /external/stack004-manifest.json \
  --width 1536 --microbatch 4 --output-dir /external/hpo-bridge
```

Pass the resulting `--reference` and `--manifest` paths to `python -m hpo run`.
This changes how the same ordered examples are grouped into microbatches;
the 96 training examples per optimizer update and 32 validation examples
remain fixed. If a selected microbatch does not fit, choose another divisor
of 32 and 96 and regenerate a **new** bridge before running its campaign.

Under spectral Muon the stated hidden-matrix hypothesis is the raw NS5
update times `η√(d_out/d_in)` for each logical Q/K/V/O/gate/up/down matrix,
with `η` held constant as width grows. This is a mathematical **candidate**
for width transfer. The tied embedding/readout is Adam, and the present
initialization/readout is not a complete μP parameterization. Hyperball's
candidate width hypothesis instead holds its angular LR fixed; its radius
projection cancels any scalar applied to the update first. Neither law
transfers the best LR across batch size, horizon, data mixture, or QAT by
theorem.

The width bridges need a separately authorized GPU budget; they are not part
of the eight-hour 200M search. For each width bridge, carry **four complete
200M recipes** on the same fixed stream and effective batch: the winner,
its closest credible runner-up, and
two stable settings that were clearly worse at 200M. Choose the weaker
settings on opposite sides of the matrix-LR optimum (one too low, one too
high), while holding their other settings equal to the winner where possible.
Their 200M CE gaps must exceed ordinary validation noise; an unstable or
divergent setting is not a useful negative control. Run all four at width
1536, then all four at width 2304 for a width-transfer claim. Compare them
at the same post-warmup token milestones, preferably 750 and 1,272 updates;
earlier points are only trajectory or stability checks. If either former
loser becomes best,
the 200M ranking did not transfer, even if the old winner still trains well.
Compare **where the LR minimum moves**, not only each width's raw loss.
Measure logit/activation RMS, logical weight and update RMS, clipping and
stability. If the optimum shifts, fit and test a simple role-specific
width exponent rather than calling the base rate transferable. The target
1.5B check should also use the intended production data mixture and a
declared global-batch bridge before a final production decision. If the
budget covers only 200M English screens, report an English proxy recipe,
not a production recipe.
