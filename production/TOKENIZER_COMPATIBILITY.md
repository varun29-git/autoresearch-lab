# Frozen tokenizer integration and compatibility gate

Use `terminus.tokenizer65k.Frozen65kTokenizer` with the frozen artifact for
training, evaluation and decoding. `production.tokenizer_gate.FrozenBatchTokenizer`
provides a minimal batch API (`encode`, `encode_batch`, `decode`) with exactly
that encoder, NFC normalization and literal special-token spellings. It does
not export an equivalent stock Hugging Face BPE JSON; the stock conversion
cannot express all current routing/seed/decoder semantics. Do not describe the
batch adapter as Rust-speed tokenization. Parallelize independent work using
the existing `scripts/tokenize_parallel.py` pipeline where appropriate.

The content adapter emits no reserved IDs. Insert EOD 4 at real document
boundaries when assembling production streams. Do not concatenate content-only
AWS bins and label them `continuous_eos`. Existing indices identify documents;
source-text checks are needed to establish encoding compatibility.

```bash
python -m production.tokenizer_gate \
  --tokenizer tokenizer/tokenizer-65k.json \
  --candidate-hf /external/candidate-tokenizer.json \
  --source-id-pairs /external/source-and-actual-ids.jsonl \
  --output /external/tokenizer-audit.json
```

Each optional source/ID JSONL row contains `text` and the **actual stored**
content `ids`. The report distinguishes regression fixtures from examined
corpus records; it does not certify unexamined data. It exits nonzero on any
encoding/round-trip mismatch. Omit the candidate to check the exact adapter.
Do not generate the IDs afresh and pass that as an audit of existing shards.

Existing AWS tokenized data is unchanged by this code. Determine affected
sources and the repair/regeneration plan before a human launches new data jobs.
No matching hash or manufactured sidecar can repair incompatible token IDs.
