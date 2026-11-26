# Source index and encoding recovery

Keep original bins, source files and their identities intact. An index repair
changes addresses into a token bin; it does not change its encoding. Preserve
the original source selection, dataset exclusions and holdout families.
All plans, catalogs, source records and results must remain outside Git.

## Recover text split across raw objects

The native converter accepts an explicit `text_source_recovery` recipe for S3
text-line sources. It can append the next object's exact first-line bytes to
complete the preceding object's final document. Pin both object snapshots,
the appended byte count and SHA-256; the prefix must contain exactly one final
LF. Decode after joining the bytes so a split UTF-8 character remains intact.
The receipt hashes the reconstructed source and binds the recovery recipe.

A leading fragment already assigned to the preceding object may be skipped
with `unselected_prefix_lines` (at most two physical lines). The converter
checks every original index record and rejects the recipe if it would skip any
selected training or validation row. Later row numbers retain their original
coordinates. Strict UTF-8 and indexed byte-length checks still apply. Resume
also rechecks the companion object's snapshot. This repairs original source
boundaries; it does not decode historical token bins as input.

## Repair split-local source indices

`source_index_repair` takes a frozen external S3 plan. Each object supplies
`tokens_uri`, `index_uri`, `token_etag`, `token_bytes`, `validation` and `source`.
The plan supplies a fresh `immutable_output_prefix` for corrected indices.

```bash
python -m production.source_index_repair \
  --plan /external/recovery/index-plan.json \
  --output /external/recovery/index-results \
  --profile <operator-profile> --workers 24
```

The tool verifies the bin snapshot, positive document counts, original
nonoverlapping offsets, hash-based split membership and exact exhaustion of
the bin. It changes only `token_start`. Its split predicate is specifically
the English writer's BLAKE2b/little-endian/modulo-1000 policy; do not apply it
to another historical writer without proving the same split policy.

Corrected indices are published with conditional creation under content-derived
keys and independently verified SHA-256 checksums. The journal supports resume.
No source bin or original index is overwritten. Use its explicit original-to-
corrected index catalog instead of changing the old object's identity.
Reject orphan/coarse historical outputs from selection; replacement source
range coverage does not establish identical payloads or complete source QA.

The optional `source_index_repair_lambda.handler` runs the same validation in
the bucket's region. It handles bounded invocations and defers unfinished
objects before timeout. It never deploys resources. An operator must deploy
with source-prefix read permissions and repair-prefix write permissions,
bounded concurrency, an external durable journal and resource cleanup.

## Bind one execution path to the production corpus

An ordered vocabulary hash is insufficient. A byte-level encoder's `café`
can have the same IDs as a literal-Unicode encoder's `cafÃ©`. A decoder cannot
infer which source meaning those IDs had. Fixing the stock export's decoder
also cannot recover its literal-marker/control or prefix/indentation collisions.
Keep the stock decoder candidate as a diagnostic, not a tokenizer promotion.

The existing CI-1 baseline is `Frozen65kTokenizer` through
`FrozenBatchTokenizer`: NFC, ordinary literal special spellings, explicit
document EOD insertion and strict Unicode round trips. Bind the artifact hash
**and the encoder implementation identity**. Any alternative requires an
explicit review of encoder, decoder, corpus, evaluation and checkpoint semantics.

For each selected source document, compare its stored IDs with this execution
path on the original text. Keep exact matches after complete source acceptance;
regenerate incompatible IDs from original source text. Decoding old IDs is
not an acceptable substitute for that source. A matching three-document sample
does not certify an entire source pool.

## Exercise source conversion on CPU

`source_canonicalization` consumes an external JSONL source selection. Each
record requires `text`, `source_uri`, `row_identity`, `doc_id`,
`source_text_sha256`, `expected_policy_text`, `validation` and `split_family_id`.
For the frozen baseline, expected text is NFC of the original source text.
Retain the existing validation assignment; a new hash split can leak old
holdouts into training. Reconcile translated, romanized and repeated records
using their original family identities before emitting a final manifest.

```bash
python -m production.source_canonicalization \
  --records /external/recovery/original-source-records.jsonl \
  --encoder-kind frozen --tokenizer /path/to/CI-1/tokenizer/tokenizer-65k.json \
  --tokenizer-sha256 <explicitly-reviewed-artifact-sha256> \
  --output /external/recovery/conversion-pilot
```

The converter rejects modified source text, repeated source rows, reserved
content IDs and conflicting train/validation families. It preserves row
boundaries and emits little-endian content bins with split-local indices.
It reports decode/reencode and policy equality without claiming production
acceptance. It does not stream or schedule a complete S3 corpus conversion.
Large original-source selection, extraction, batching and publication need
a separately staged CPU campaign. Neither this command nor these docs
authorize a paid VM or GPU campaign.

After complete conversion, remeasure train-only supply, subtract every
reserved holdout family, reconcile source quotas in the new token units,
and rebuild validation and document-aware training packs using the same
encoder. Insert EOD once per document and freeze the final catalog before
native acceptance. Original token counts cannot be treated as new-encoder
counts. CPU qualification does not certify a trillion-token corpus or native
MI325X training.
