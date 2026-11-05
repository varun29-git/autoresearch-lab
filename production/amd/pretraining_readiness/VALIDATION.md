# Freeze validation before production

Use the current data engineering curriculum as the source contract, then
generate exact cell weights, manifests, task records and reports outside this
repository. Historical source maps and proxy mixtures do not override that
curriculum. This workflow preserves model/optimizer semantics and authorizes
no paid campaign.

The current data engineering preferences omit global MinHash and benchmark
overlap scans. Preserve these choices and report overlap as unaudited.

## Data and independent roles

`production.validation_data build` reads indexed uint16 document spans without
re-encoding, pins S3 identities and range hashes, preserves IDs, inserts EOD 4
only at true document ends, and writes continuous blocks with L inputs and L
already-shifted targets. A sequence never combines different language/source
cells. Declare file selection and bounded document windows; this is not an
unbiased population sample or a complete translation-family audit.

Published document IDs assign the same family to development or final across
languages/sources. Unknown translations with different IDs remain unresolved.
Prefix/suffix token splits and loss-masking heldout text do not establish
document independence. Reserve entire documents, including their context,
from every training phase.

`production.validation_data filter-index` produces retained document spans
and a hashed receipt. The packer must read those spans; concatenating the old
bin would retain excluded documents. The receipt does not certify packed
output. Audit the actual packed corpus before recording exclusion application
in its provenance. `verify --training-manifest` checks that declaration and
its identities; it is not an independent proof that the packer followed it.

Use existing physically separate `.val` parts where available. Otherwise,
reserve indexed documents without modifying the bins. Both roles stay out of
training. Development is for monitoring/selection; final is for predeclared
reporting checkpoints. A CLI flag prevents accidental final access, not
deliberate access.

## Scores and permitted decisions

`production.validation_evaluate` loads a verified checkpoint and scores each
cell without padding or re-tokenization. Frozen weights require exact coverage.
The evaluator reports target-weighted NLL and language/source/variant summaries,
keeping sample-token-weighted loss separately. Missing cells prevent a complete
contract score; diagnostic scoring without weights is labelled incomplete.
A nonempty coverage floor does not establish adequate statistical power.

Keep weights fixed across warmup, plateau, cooldown, dense and QAT comparisons.
Freeze an exact sequence prefix for quick monitoring; use the larger development
set for consequential comparisons. Tokenizer lineage, backend, target counts,
role, sequence limit and checkpoint identity belong in every result. Aggregate
loss sums and target counts rather than unequal batch means.

`production.validation_generated` scores saved completions against frozen
math/QA/code records and rejects missing/duplicate items. Math uses the last
marked or numeric answer; QA uses normalized EM and whitespace-token F1; code
uses fixed assertions with explicit execution enabled on an isolated worker.
Freeze these rules before comparison. Scores do not verify the generating
model or decoding provenance: bind a generation-runner receipt separately.
Native generation and measured evaluation cost remain acceptance checks.
Reference-NLL is not generated accuracy.

## Monitoring and intervention contract

Freeze thresholds/timing externally against qualified full-size/native baselines.
Short-proxy limits are not universal production standards.

- Persist tokens, updates, actual LRs, raw pre-clip gradient norm, clip scale,
  clipping frequency, group update/weight RMS ratios, optimizer-state norms,
  FP8 saturation/zero fractions and numerical-error flags. Monitor embedding
  and tied-head behaviour together, with ordinary/QK norms separately.
- Record attention-logit tails, Q/K norm trends, attention entropy and output
  logit tails using declared sampled layers/tokens. Qualify instrumentation
  and its overhead on native hardware. Absent telemetry is not a healthy
  measurement. Observe data starvation and rank skew alongside learning.
- Compare validation slopes at matched tokens on a fixed development prefix.
  Report each critical language/domain alongside the aggregate. Separate
  curriculum transitions from plateau regressions. Initial cold-start clipping
  alone does not justify an automatic LR reduction.
- Nonfinite losses/gradients/state, incorrect targets or verified corruption
  stop updates and require a good checkpoint. Sustained learning regressions,
  clipping or group-ratio excursions require confirmation on the larger dev
  set and a recorded intervention. Preserve optimizer state and a recoverable
  checkpoint; do not silently change LR after one noisy evaluation.
- Freeze scored-task prompts/decoding. Use deterministic pass@1 for scored
  comparisons and a separate multi-temperature continuation panel for
  qualitative repetition/coherence review. Compare dense, QAT and export on
  the same permitted tasks and budget their measured cost.

## Remaining corpus and hardware acceptance

Verify source availability after all holdouts, no-refill exposure, unresolved
translation origins, source-spec versus executable tokenizer hashes, actual
packed exclusions and native evaluation correctness/cost. Resolve curriculum/LR
boundaries and valid-target accounting before compiling a production schedule.
Validation preparation alone establishes none of these and does not certify
month-long AMD stability.
