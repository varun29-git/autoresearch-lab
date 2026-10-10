# Dense HPO loop after optimizer selection

**Objective:** Find a low-loss, stable, affordable dense recipe for the fixed
25-layer Peri-LN + learned-value-residual Stack 4 model, then measure whether
its rates transfer across width and training horizon. Work on one H100 NVL
for research. The human supplies the total paid GPU-hour cap and authorizes
the campaign. Kimi runs experiments continuously within that cap and stops
at an informative result or the cap. This is dense FP8 only; QAT is a later
separate campaign.

The source is yesterday's selected E3/E4/E9 speed stack. The 200M reference
in `reference_200m.json` fixes its model, prepared English token stream,
393,216-token global update, seed, validation examples, full 2.5B-token WSD
schedule, and scorer. Before starting, incorporate the optimizer winner from
the separate Hyperball comparison. If spectral Muon wins, keep the current
`spectral_unclamped` implementation and weight decay. If Hyperball wins,
merge its tested implementation and set the reference rule to `hyperball`.
Do this **once**, before the first HPO trial. The same printed LR has a
different meaning under the two rules.

One loop iteration is: make a specific hypothesis, edit `candidate.json`,
commit it, run one trial, read the held-out English cross-entropy curve at
matched tokens, keep or revert the candidate, then choose the next idea.
The editable HPO fields are matrix Muon LR, tied embedding/readout Adam LR,
main/Peri/final norm LR, Q/K gain LR, matrix/embedding weight decay, Muon
momentum, Adam betas/epsilon, and global clipping. The model and data are
fixed. Use the same trial name only to **resume the same candidate and code**;
use a new name for a different recipe. The runner checks that condition.

The 125-update point (49.15M tokens, roughly five minutes at the archived
200M speed) is a stability screen, not an LR winner: it is still in warmup.
The 250/500-update points are likewise early. The 750 and 1,250 points
(294.91M and 491.52M tokens) give a first post-warmup ranking. Continue only
promising and nearby candidates. One or two finalists can be explicitly
continued to update 6,358 (2.500B tokens) to observe the completed decay;
the runner requires `--finalist` beyond 1,250. Do not infer a long-horizon
optimum from a five-minute or 500M-token checkpoint. Report the full curve,
tokens/s, clipping, FP8/stability incidents, and active GPU time.

Start with a **wide but reasoned matrix-LR bracket** for the selected optimizer.
The old Moonshot-style 0.003 cannot be copied numerically to spectral Muon
or Hyperball. Inspect the initial logical-matrix update-to-weight ratios,
especially small-initialized O/down, then narrow the bracket; stop nonfinite
or clearly collapsing settings. After matrix LR, tune the tied embedding LR,
then norm and Q/K gain LRs. Test weight decay and momentum only around a
stable rate region. When LR changes, record the resulting decoupled
`LR × weight_decay` shrinkage; if a gain might be due to decay, run a
matched-shrinkage counterfactual. Avoid a Cartesian sweep. Change one
major axis at a time, except for a justified coupled calibration.

Example on the human-provided H100, with the exact archived manifest staged
on its persistent disk:

```bash
python -m hpo run --trial spectral_lr_003 --updates 125 \
  --manifest /external/stack004-manifest.json \
  --data-root /srv/terminus-10b/data/packed \
  --results-root /external/hpo-results

python -m hpo run --trial spectral_lr_003 --updates 1250 \
  --manifest /external/stack004-manifest.json \
  --data-root /srv/terminus-10b/data/packed \
  --results-root /external/hpo-results

python -m hpo rank --updates 1250 --results-root /external/hpo-results
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

Carry the best 200M rate and neighboring sentinel rates to width 1536,
then at least a short width-2304 check on the same fixed stream and batch.
Compare **where the LR minimum moves**, not only each width's raw loss.
Measure logit/activation RMS, logical weight and update RMS, clipping and
stability. If the optimum shifts, fit and test a simple role-specific
width exponent rather than calling the base rate transferable. The target
1.5B check should also use the intended production data mixture and a
declared global-batch bridge before a final production decision. If the
budget covers only 200M English screens, report an English proxy recipe,
not a production recipe.
