# Quality before extending the MI325X run

**Current data-policy override:** the data engineering preferences explicitly
omit global MinHash and benchmark-overlap scanning. Preserve those choices;
label overlap as unaudited rather than making those scans a launch requirement.
Follow [VALIDATION.md](VALIDATION.md) for indexed holdouts and fixed evaluation
coverage. The historical checks below do not supersede this policy.

Engineering audit and acceptance specification, 2026-11-01; baseline reviewed
at `784dc98`, with the two defensive fixes below included in this milestone.
Complements [ADAPTIVE_COMPUTE_PLAN.md](ADAPTIVE_COMPUTE_PLAN.md).
This document defines work and evidence; it does not certify native AMD
execution or change the selected architecture/optimizer. Actual corpus
inventories, evaluation data, measurement reports and run configs stay
outside Git.

The objective is validated learning per allocated GPU-hour and a usable final
artifact. Filling HBM, minimizing training loss on repeated data and maximizing
nominal tokens are not sufficient quality criteria. We cannot establish that
every possible optimization is beneficial before a run. We can close known
failure paths and require bounded evidence before promoting each plausible
high-value change.

Implemented in this milestone: the trainer rejects unexpected valid-target
counts before applying the optimizer update, and operand telemetry derives
FP8 boundary values from the actual FN/FNUZ dtype. CPU regressions cover both
changes. Native MI325X checks remain necessary. These source changes require
new source-bound preflight/qualification receipts before use; existing pinned
campaigns must retain their original source or undergo an explicit requalification.

Local validation: 26 focused telemetry/qualification-runner/update-block tests
pass with the pinned Muon dependency, and 17 trainer tests pass, including
padded tails, exact resume and two-process CPU DDP. These are tiny-model CPU
checks, not native MI325X or full-size quality/performance measurements.

## 1. Current implementation versus remaining evidence

| Area | Already present | Remaining production requirement |
| --- | --- | --- |
| Model/optimizer | Reviewed 24-layer model, spectral logical-matrix Muon, separate AdamW groups, FP32 masters/state | Preserve semantics; qualify actual AMD forward, backward and complete updates |
| Data construction | NFC normalization, ID/exact/MinHash dedup, source adapters and contamination guard in `terminus/corpus.py`; immutable token contracts | Verify those operations ran on the **actual production corpus**, with representative language/domain quality and independent split/contamination audits |
| Focused qualification | Frozen five 5B proxy contracts, horizon views, bounded runner, selection/transfer, telemetry, ledger and recovery tooling | Complete evidence chain and comparisons; implementation is explicitly single-H100, not an AMD production runner |
| Evaluation | Token NLL, source/language summaries, reference-NLL evaluation contracts, reserved holdout files | Generated capability scoring and document/translation-family isolation; full target coverage and predeclared margins |
| Recovery | Checkpoint contracts, immutable asynchronous snapshots, cursor checks, bounded active S3 upload/verified retention and restore | Real eight-rank restore and provider-network behaviour, including completion after a failed upload/preemption |
| Adaptive production | Fixed-QAT remaining-budget forecast, stopped-checkpoint future WSD/data forks, persistent allocation ledger and failure supervisor | Native all-rank transitions/recovery, actual provider shutdown hooks, rate-evidence binding and automatic multi-phase orchestration; see companion specification |
| Ternary QAT | Reference quantizer/STE, dense-to-QAT transition, AMD FNUZ kernel, explicit recipe and CPU checks | Native numerics, recovery quality and export qualification remain required |

Some external H100 packet gates are marked passed, including staging of a
qualification subset and native preflight/fit. They do not establish that a
trillion-token production corpus is ready or that MI325X passes the same
checks. Reuse compatible evidence rather than repeating completed work, but
verify its exact scope and identity before accepting it.

## 2. P0: corpus quality and actual learning targets

Do this before buying long-running GPU time. CPU/storage work and human
language review can proceed independently of the training node.

- [ ] Freeze the intended scope: English, Hindi, Marathi, Kannada, Telugu,
  Tamil, Gujarati and Bengali, plus math/code capabilities. Explicitly version
  additions rather than accepting an old language list from another campaign.
- [ ] Build the inventory from readable raw sources, then count usable tokens
  with the **frozen 65,536-token tokenizer** after filters and deduplication.
  Do not use compressed bytes, old TinyLlama token counts or SFT packs as
  compatible pretraining token counts. Decode random packed sequences and
  verify tokenizer IDs, Unicode, EOS, shifted targets and numeric ranges.
- [ ] Record raw/accepted/rejected counts, retained bytes/tokens and quality
  samples for each source and language. Review naturalness, language identity,
  code syntax, math correctness, boilerplate, truncation and repeated text.
  A classifier score alone is insufficient, particularly for Indic text and
  translated technical material.
- [ ] Resolve translation versions and common origin IDs before splitting.
  Keep English originals, translations, transliterations, mirrors and near
  duplicates in one train/dev/test family. Reject quarantined or failed
  translations; audit missing quarantine metadata rather than treating its
  absence as a clean bill of health.
- [ ] Require a verifiable upstream preparation receipt. The packer currently
  writes `document_dedup: id_exact_minhash_v1` unconditionally, accepts arbitrary
  correctly shaped JSONL, and does not itself execute deduplication. The
  upstream builder implements deduplication and can disable near-dedup. That
  manifest label is therefore **not evidence that the input was deduplicated**.
  Bind actual preparation settings, inputs and rejection counts to the packed
  corpus; independently audit a sample. Do not silently change existing pinned
  campaign manifests to repair this metadata.
- [ ] Audit contamination with the actual frozen evaluation text and answer
  variants, including translated versions. The shingle guard only knows its
  supplied reference set; empty/incomplete references do not prove cleanliness.
- [ ] Verify code coverage beyond synthetic descriptions of programming. The
  source map is an address book, not proof of a broad usable code corpus.
  Inspect repository/file duplication and testable code quality; similarly
  separate verified math reasoning from plausible but incorrect solutions.
- [ ] Set source-token weights and per-source repetition limits from the
  cleaned inventory. Record unique exposure as well as total exposure. Do not
  compensate for source exhaustion with an unreviewed repeated stream.
- [ ] Ensure each critical language/domain receives useful exposure at the
  **lowest** funded horizon. Longer forecasts may add approved exposure; they
  must not be required just to reach essential coverage.
- [ ] Preserve the reviewed continuous-EOS packing and causal/local masks.
  Check full batches for target leakage, dropped/shuffled boundaries and
  unintended padding. A change to document-reset positions, attention across
  documents, loss masks or sequence length is a learning change, even if it
  improves a throughput benchmark.

Source: [`DATA_SOURCE_MAP.md`](../../../DATA_SOURCE_MAP.md),
[`corpus.py`](../../../terminus/corpus.py),
[`prepare_campaign_data.py`](../../../terminus/prepare_campaign_data.py),
[`data.py`](../../../terminus/data.py). The source map's historical inventory
must be refreshed externally before sizing the production run.

## 3. P0: evaluation that can detect the wrong kind of improvement

- [ ] Freeze a monitoring/dev set, the proxy-selection holdout and an untouched
  final reporting set with explicit permitted uses. Split by document/origin
  family **before** token packing. Physical prefix/suffix files alone do not
  establish independent documents; a long document can cross that boundary.
- [ ] Audit cross-language origin families. `document_uid` currently includes
  the source name, so related texts in different sources do not automatically
  receive the same split. Version preference/exact-text dedup is not a general
  translation-family split guarantee.
- [ ] Fix tokenization, prompts, decoding, task versions, answer parsing and
  score aggregation before comparisons. Give every mandatory domain a fixed
  nonempty set; a missing domain must fail the gate rather than disappear from
  a favourable aggregate.
- [ ] Report per-language/domain NLL and a fixed target-weighted aggregate.
  Keep token-weighted loss too, but do not let a larger English validation set
  hide degradation in Marathi, Kannada, mathematics or code.
- [ ] Add generated evaluation: exact/normalized final-answer scoring for
  math, language-appropriate comprehension/QA, and sandboxed code execution
  with fixed tests. Budget the evaluation workload explicitly. The current
  `production.evaluate` computes teacher-forced reference NLL, not generated
  MMLU accuracy, math accuracy or code pass@1.
- [ ] Record train/validation gap, validation slope during the plateau,
  pre-decay scores and final scores. A short cooldown can improve loss while
  hiding a poor sustained learning trajectory. Keep the existing trial
  decision rules fixed; if a new criterion is needed after seeing results,
  declare it exploratory and use separate confirmation evidence.
- [ ] Predeclare practical superiority/noninferiority margins and sample
  requirements. Use paired document-level uncertainty where appropriate;
  it does not measure seed variance or remove multiple-comparison bias. A
  near-tie is not grounds for claiming an exact optimum.
- [ ] Compare qualified dense, QAT and exported artifacts on the same task
  contract. Select checkpoints on dev criteria, then consult the reserved
  final set according to its frozen policy. Keep the dense fallback available.

Implementation detail requiring review before production selection: the
current proxy selector reports pre-decay NLL but ranks on final aggregate and
domain margins. Its domain comparison can skip a domain missing from a
candidate. Verify exact coverage upstream and add explicit production
coverage rejection; do not interpret a completed selection JSON as proof of
all the gates above.

Sources: [`OPERATIONS.md`](../../OPERATIONS.md),
[`selection.py`](../../kimi_h100_5b/selection.py),
[`horizons.py`](../../kimi_h100_5b/horizons.py).

## 4. P0: the executed update must match the learning experiment

- [ ] Qualify AMD FP8 FNUZ forward, dgrad and wgrad against independent
  higher-precision references, including tiny token-normalized gradients,
  long reductions and zero/subnormal/range boundaries. Finite output alone
  is not acceptance. Check full-width shapes, packed slices and FP32 wgrad.
- [ ] Check the exact 512-token local causal boundary, four global layers,
  GQA indexing, RoPE positions and packed Q/K/V gradient routing. Use tests
  with values near boundaries; random aggregate loss can hide an off-by-one
  attention error.
- [ ] Verify logical Q/K/V and gate/up Muon updates, NS5, Nesterov, spectral
  factors and decay on the intended groups. Verify tied embedding/head is
  updated once, width transfer is applied once, and norm/QK rates remain in
  their own groups. Do not compare published LR numbers without accounting
  for their parameterization.
- [ ] Test FP8 cache invalidation after update, conversion, restore and
  reference branches. Ensure temporary probes cannot mutate the real
  optimizer groups, moments, RNG, scaler/cache state or data cursor.
- [ ] Compare one-rank accumulated and eight-rank updates on the same global
  examples: losses, pre-clip gradients, moments and actual parameter deltas.
  Verify DDP averaging, `no_sync`, partial/final updates and clipping exactly
  once. Declare precision/reduction-order tolerances before running.
- [ ] Qualify the token-normalization contract on the intended eight ranks.
  Core training divides by nominal update tokens and counts targets not equal
  to `-100`. This milestone adds an all-rank mismatch rejection **before**
  applying an optimizer step, alongside the runner's existing post-callback
  check. CPU tests verify rejected masks leave weights/progress unchanged and
  legitimate padded tails remain valid. Supporting intentional new loss masks
  requires a separately reviewed valid-target-weighted objective.
- [ ] Use a real independent BF16/reference trajectory for numerical drift.
  The current qualification `reference_replay` repeats the same implementation
  from a snapshot. This can test repeatability/restoration; it cannot detect
  a systematic mathematical error shared by both executions.
- [ ] Qualify and complete AMD operand telemetry. This milestone replaces the
  hard-coded H100 E4M3FN boundary of 448 with dtype-specific limits, including
  MI325X's selected E4M3FNUZ maximum of 240, with CPU regressions. Still observe
  actual native quantized operands/scales, pre-quantization clipping and
  underflow in weights, activations and backward operands. Weight-cache
  boundary occupancy alone is not all-operand saturation evidence.
- [ ] Run full-state finite/required-gradient checks and measure complete
  optimizer, validation and checkpoint memory peaks. Store calibrated alert
  baselines and critical failure thresholds in the external run contract.

Sources: [`train.py`](../../../terminus/train.py),
[`telemetry.py`](../../kimi_h100_5b/telemetry.py),
[`branches.py`](../../kimi_h100_5b/branches.py),
[AMD acceptance](../ACCEPTANCE.md). The H100 runner explicitly requires one
H100; it cannot be made into an eight-MI325X runner merely by changing a CLI
GPU count.

## 5. P1: preserve learning while spending HBM and compute well

Use the existing [focused LR/transfer plan](LR_BATCH_MONITORING.md), not a new
full-size Cartesian HPO campaign. Compare the five 194M proxy candidates at
the same 2,097,152-token global batch and 5B horizon, then transfer the winner
once and confirm at full width. Actual fit/overhead must respect the separate
50-hour cap. Budget overruns do not justify calling an incomplete matrix a
completed comparison.

On AMD, test microbatch 8/accumulation 8, 16/4 and conditionally 32/2 at the
same global batch, context and stream. Bigger microbatches use HBM to reduce
launch/accumulation overhead without reducing optimizer updates per token.
Choose by sustained full-update rate and memory headroom, not nominal memory
utilization. Start without activation recomputation if the qualified path
fits. Retain the readiness target of at least 15% usable HBM headroom through
the complete workload; measure temporary workspaces and fragmentation too.

Prioritize measured, semantics-preserving improvements:

1. Remove data stalls with local immutable shards, bounded prefetch, correct
   host/NUMA placement and asynchronous durable transfer. Keep S3 network
   stalls outside the training data critical path.
2. Select the fastest numerically qualified FP8/attention kernels on the real
   shapes; reuse valid quantized weights during accumulation and invalidate
   them after every weight change.
3. Tune DDP buckets/overlap and host threading with identical global examples
   and optimizer work. Check all-rank completion and real cadence costs.
4. Consider compile/graphs/fusion only after warm sustained measurements
   identify their value and shape/cursor/gradient semantics pass. Report any
   numerical differences; do not accept a faster altered objective.
5. Keep FP32 masters, accumulated gradients and optimizer state while memory
   allows. Filling freed HBM with extra global batch or compressing state is
   not automatically a learning improvement.

Changes to global batch, warmup, decay, matrix/norm weight decay, Muon scaling,
quantizer granularity, context, data mixture or architecture require a
bounded learning comparison. Preserve the reviewed baseline as the control.
Do not introduce many simultaneous changes simply because other models use
them. Inspect cumulative decay exposure and moment/clip behaviour over the
longer horizon; numerical equality of LR alone is not learning equivalence.

Test a source-mixture or high-quality finishing candidate only after the
corpus/held-outs are ready. Compare matched-token and matched-compute results
and all critical domains. Choose the number/duration of follow-up branches
from measured cost and remaining uncertainty, with a named budget and stop
rule. No automatic 960-GPU-hour retuning requirement is reinstated here.

## 6. P0 before a mixed run: QAT must be a funded quality stage

- Implement/qualify the current ternary semantics on AMD: scaled `-1/0/+1`
  effective weights, FP8 matrix work, FP32 latent weights/state and the
  reviewed STE. There is no ternary-core requirement and no implied 1.58-bit
  training-memory or throughput benefit.
- Test immediate conversion, phase-relative rewarm, recovery, plateau and
  cooldown. Preserve the parent, source cursor and optimizer lineage; test
  reset/carry choices only as declared candidates.
- Use a meaningful trained parent, plus a same-data dense-continuation
  control, to isolate the quantization cost. A fresh random or tiny proxy
  check qualifies plumbing, not late-production recovery.
- Keep historical QAT rates and the 60/40 split as hypotheses. Relative rates
  must refer to the selected dense **peak**, not its terminal decayed LR.
  Do not shorten QAT just to report a larger token number.
- Define critical-domain QAT acceptance and exported-artifact fidelity in
  advance. Native-QAT quality and packed-inference correctness are separate
  gates. Fund conversion, recovery and final tests before spending the last
  plateau hours.

See [QAT_ALLOCATION.md](../QAT_ALLOCATION.md) for evidence and historical
candidates. A dense-only artifact is a useful fallback, but if the ternary
deliverable fails its gate, report that part as incomplete.

## 7. Monitor, recover and decide without wasting the month

| Cadence | Signals | Required interpretation/action |
| --- | --- | --- |
| Every update, cheaply | Valid targets, finite loss/global gradients, real LR per group, pre-clip norm, clip factor, data wait, progress/phase | Abort an invalid update coherently across ranks; incident/restart must not skip examples or reset budget |
| Sampled complete updates | Logical-matrix update/weight ratios, zero/tiny gradients, full-state finiteness, moments, QK gains/logits, residual/value-residual statistics, FP8 operand scales/underflow/clipping | Compare to qualified baselines; persistent clipping, shrinking useful updates or growing logits can precede visible loss failure |
| Fixed validation milestones | Per-language/domain NLL, target-weighted aggregate, train/dev gap, generated capability scores | Distinguish useful learning from source repetition, domain forgetting or a transient training-loss improvement |
| Checkpoints/forecast windows | Slowest-rank rate, HBM peaks, allocation spent, remaining finishing cost, local bytes, upload backlog, verified remote artifact | Reduce uncommitted plateau when needed; do not spend unverified backup/finish capacity |
| Phase boundaries and release | Same-contract dense/QAT comparison, exact restore, tokenizer/data lineage, export fidelity and generated tests | Preserve the verified dense parent; promote only artifacts satisfying the declared quality contract |

Keep expensive full scans and task suites sampled and timed so observation
does not consume an unbudgeted fraction of the run. Numerical hard failures
need immediate action. Nonfatal drift alerts should use qualified baselines,
persistence and independent validation; a universal loss/gradient threshold
copied from a different model is not a reliable policy. Do not silently lower
LR, change mixtures or reset moments after an alert.

Remote durability requires a real drill: interrupt after a complete update,
restore from the remotely verified object on the intended node, and compare
continuation state/data and declared numerical tolerances. Test upload
failures, local retention and storage growth. An upload process that starts
successfully is not a recovery receipt. Include optimizer/RNG/data/schedule
state and exact tokenizer/source identities in the recoverable artifact.

## 8. What needs which hardware

| Work | Resource | What it can establish |
| --- | --- | --- |
| Corpus processing, dedup/split audit, decoding samples, frozen eval preparation, scheduler/ledger tests | CPU, storage and reviewers | Data/contract correctness without consuming eight-GPU time |
| Existing five 5B proxy comparisons and one transferred confirmation | The separately budgeted single H100 | Relative LR evidence at the intended global batch through accumulation; not AMD kernel acceptance |
| FNUZ kernel tests, local masks, full-width backward/optimizer, bounded BF16 comparison, single-device fit | One MI325X, if independently available | Native single-device correctness/fit; record all allocated cards if the provider bills a whole node |
| DDP equivalence/overlap, sustained production speed, all-rank memory, distributed faults/resume, actual provider I/O | All eight MI325X | Production execution and cost evidence; a one-GPU result cannot replace it |
| QAT conversion/recovery and release-quality comparison | Native AMD at the qualified scale; final acceptance on intended topology | A usable ternary path and its real phase cost/quality |

Complete offline work first, retain the focused proxy budget, then use a
bounded native stage with explicit follow-ups only for unresolved gates.
Publish the final external packet as a list of passed, failed and unrun
requirements. Neither code availability nor a high tokens/second result is
the final go/no-go decision.
