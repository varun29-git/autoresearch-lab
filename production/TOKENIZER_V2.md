# Tokenizer decision for fresh pretraining

The requested full source retokenization uses one explicit tokenizer artifact
and execution contract. Ordered vocabulary equality between the old custom
encoder and the HF export was insufficient: their routing and byte/Unicode
interpretation can differ. Recover from original source text, not by decoding
the old token bins. Preserve original holdouts during conversion, then reconcile
shared document families before packing the training and validation streams.

## Reviewed change against the selected model

The atom revision keeps the 65,536-row embedding/head, every previously emitted
content ID, all English merges, the six kept Indic Unicode blocks, NFC policy,
contextual decoder grammar, and optimizer/model architecture unchanged.
It repurposes only twelve retired Indic-space placeholders for missing
single-character atoms: ZWNJ, ಠ, ಃ, ऋ, ॲ, ঐ, ঢ, ઞ, ஊ, ఊ, ఠ and ಊ.
The deterministic patch is `production.tokenizer_revision`.

General reserved IDs **5–31 remain unused**: 27 slots for future control tokens.
Ordinary encoding never emits these slots. This does not mean injecting padding
or blank tokens into pretraining, and does not mean their tied-head rows receive
no gradients. Production generation masks unused IDs, UNK, BOS and PAD; EOS/EOD
remain explicit stopping controls. Training loss and optimizer semantics stay
unchanged. Defining future special tokens requires a new tokenizer/data version.

The source artifact remains reproducible:

- Original: `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`.
- Atom revision: `dc636d96224d059d03619e61f0ecddc3ac1e5c8c6fb46ee051f66cabc65fb35c`.

Build the revision outside Git; never overwrite the historical artifact or a
checkpoint's tokenizer contract. New training configs and manifests must pin
the chosen artifact and regenerated token counts. Historic H100 checkpoints
remain attached to their original tokenization; do not relabel them as v2.

## Acceptance and quality evidence

Required correctness checks cover exact NFC round trips, reference/native ID
agreement, English merge ranks, Indic merge scores/ties, mixed-script routing,
format characters, indentation, literal special-token spellings, all supported
Unicode scalar values, NUL/surrogate rejection, bounded caches, and incremental
decoding across every token boundary. Expanding each of the twelve new atoms
back into its former UTF-8 fallback IDs must reproduce the original token stream.
Whole-source Linux qualification additionally compares file hashes against
reference outputs and checks original selection and holdout assignments.

Quality comparisons use the same raw source texts for Terminus, Sarvam-1,
Qwen3 and MiniCPM4 tokenizers, with no automatic special tokens. The external
decision packet records language-specific token counts, fallback rates,
round-trip results, pinned comparator revisions and sample limitations. This
supports selecting an efficient tokenizer for this English/Indic curriculum;
it does not establish a universal best tokenizer or a downstream model-quality
gain before training. No data-dependent vocabulary retraining is performed.

## Execution and remaining corpus work

Use the [C++ implementation](../native/tokenizer65k/README.md) for conversion,
with Unicode 16.0.0 and regex 2026.7.19 pinned. Stream original Parquet/text,
project only reviewed fields, preserve selected rows, reuse worker vocabularies,
and publish uint16 bins plus indices and checksums. Do not use the stock HF
export as a supposedly equivalent faster encoder.

Source inventories, CPU plans, artifacts, results and throughput forecasts stay
outside Git in the operator's pretraining-readiness directory. Resolve missing
raw-source mirrors and source-specific renderers before launching those jobs.
After conversion: reconcile duplicate/family holdouts, recount each approved
source, rebuild the 1T and extension allocations, and pack with explicit EOD
boundaries. Conversion receipts deliberately do not claim training acceptance.
