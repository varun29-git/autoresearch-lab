# Verified translation pairs and parallel allocation

The selected small paired stream can use **any eligible language pair**.
Marathi–Telugu and English–Indic were examples. Different scripts do not
require a special attention mechanism.

`translation_pairs.py` provides a CPU constructor, source qualification and
single-coordinator quota accounting. It requires matching original source
repository, configuration, source file, source UUID and the exact original
passage SHA. A common question ID does not qualify different generated answers.
Each raw translated text must match its frozen conversion-index SHA, and its
stored token span must match the selected production encoder. A translation
needs a score of at least 95/100, no major/critical/unclassified errors, and
matching numeric quantities and fenced code/formula blocks. These are
conservative eligibility rules, not a measured optimum or complete semantic
verification. Qualified examples still need multilingual human/sample review.

The two complete passages are separated by an ordinary blank line. There are
no language tags and no internal EOD or position reset. One compound fragment
becomes **one causal attention segment**. Global layers can condition the second
passage on the first; local layers retain their 512-position window. Unrelated
documents remain separate masked segments. A whole pair must fit 4096 tokens;
otherwise use ordinary document packing. Arbitrary equal-length translation
cuts are not accepted. Combined-text encoding must exactly match concatenating
the stored component spans and separator tokens.

Each component charges its own existing language/source/dataset-child quota.
Inserted separator tokens are a separate counter; they are scored targets,
but are not additional unique corpus content. Ordinary descriptors retain their
existing bytes when no synthetic tokens are present. Compound descriptors
also bind component offsets, source identities and quota counters.

The selected policy caps paired **source content** at 20B tokens within ordinary
dense stable data, including both sides. It excludes the completed warmup,
protected 40B bridge and both QAT allocations. One counterpart is allowed per
translation family; the pair's records replace their ordinary occurrences.
Different families can alternate direction without replaying the same pair.
Fewer than 20B qualified tokens is acceptable: keep the remainder ordinary with
the same source quotas. Benefit and unwanted language switching need evaluation.

## Parallel reservation engine

`partitioned_allocation.py` supplies CPU catalogue compilation, a native
DuckDB 1.4.2 reservation engine and an exact quota coordinator. Identical text
SHA values always belong to the same deterministic hash partition. The
catalogue compiler excludes a pinned predecessor SHA census and canonicalizes
cross-source duplicates according to an explicit source-priority policy.
Incompatible token lengths for identical text are an error. Frozen heldout
exclusion and verified conversion receipts are upstream catalogue requirements.

Each partition keeps one bounded native database across phases. The
coordinator divides every language/source/child quota using actual eligible
capacity, including empty partitions. It never assumes each worker has 20%
of a scarce source. Native cumulative-token selection preserves exact integer
quotas. Dense suffix offsets can continue without gaps or replay; protected
QAT quota cuts consume ownership of the complete record. Receipts must reconcile
all partitions, predecessor/exclusion identities, actual selection records and
exact child quotas before physical materialization.

This removes the need for a full growing global-ledger copy at every ordinary
tranche. It requires an initial verified catalogue and ownership export. That
initial work, shared storage bandwidth and physical packing still cost time;
five workers are not evidence of a 6–8-hour full-corpus ETA.

Local-file execution uses external plans:

```sh
python3 -m production.partitioned_allocation compile --plan /external/catalogue-plan.json
python3 -m production.partitioned_allocation reserve --plan /external/partition-plan.json
python3 -m unittest tests.test_translation_pairs tests.test_partitioned_allocation
```

Install `duckdb==1.4.2` in the CPU runtime for native allocator tests. The rest
of the current packer does not acquire this dependency. Native allocator tests
are explicitly skipped when DuckDB is absent.

## Deployment status and remaining integration

`fixed_translation_pairs.py` connects pairing to future ordinary dense fixed
materializers using the **existing authoritative reservation ledger**. The
shared 20B ceiling is divided across the 14 selected 50B materializers, about
1.4286B per job. Program validation rejects multiplied caps, repeated family
partitions and QAT/bridge assignment. The final 10B ordinary selector is not
paired because it has no immutable fixed selection. A family hash assigns each
original source UUID to one eligible tranche, preventing repeated pairing across
phases. Source IDs are only discovery hints; raw metadata and content qualify
the actual pair. Up to four language candidates per family can be checked.

Discovery uses bounded native SQL, original Parquet range reads and a 2-GiB
Arrow row-group cache. It has a 30-minute per-job preparation limit. Qualified
pairs replace both owned whole-record occurrences; rejected or unavailable
metadata leaves the records ordinary. A missing source-token identity still
fails mandatory ordinary payload verification. Pair receipts record actual
content, inserted tokens, both source/child charges and rejection reasons.
The accepted volume can be below the ceiling.

Compound shards use a dedicated finite stream. `PackedSequenceReader` reports
both component sources and inserted-token counts, and `mixture_candidate`
reconciles component source totals against the phase quotas. Position IDs,
labels and masks follow the existing segment contract. No model geometry,
optimizer settings or tokenizer IDs change. This is a data-pipeline integration;
native segment-aware training acceptance remains pending.

The general partitioned allocator remains a **local CPU candidate**. Pairing
does not wait for its full catalogue because existing reservations already
provide global ownership. Completed warmup and started tasks are unchanged.
Deployment and source evidence are in the dated
[20B enablement packet](/Users/varundaiya/terminus-pretraining-readiness/2026-11-08/pairing-20B-enablement/README.md).

Before deploying the general allocator, compile the full verified index catalogue against the
actual inherited ledger and exclusions; freeze duplicate source priority;
measure eligible capacity for every child and compare all approved quotas;
bind new partition receipts and compact phase ledgers into the dispatcher;
and preserve existing shutdown/fleet controls. Existing fixed pairing is local
to one completed dense selection. A future cross-partition implementation must
bind both selections to one reconciled commit before removing ordinary records.

Actual eight-MI325X segment masking, positions, valid-target normalization,
graph replay and resume acceptance remain separate requirements. CPU creation
does not publish a training manifest or establish cross-lingual quality gains.
