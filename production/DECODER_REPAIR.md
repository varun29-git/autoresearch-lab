# Encoder-preserving stock decoder candidate

`production.decoder_repair` creates a fresh external Hugging Face tokenizer
directory. It changes only the decoder. The vocabulary, piece IDs, BPE merges,
normalizer, pre-tokenizer, post-processor and added tokens remain identical.
It copies the source directory's tokenizer configuration and special-token
map when present. It neither rewrites corpus shards nor promotes a default.

```bash
python -m production.decoder_repair \
  --tokenizer /external/current-hf-tokenizer/tokenizer.json \
  --output-dir /external/decoder-candidate
```

Optionally add `--source-samples-jsonl /external/original-text-samples.jsonl`.
Each line must have a string `text` field. Reports and samples remain external.
Use original text rather than output from the defective decoder.

## What changes

The candidate preserves literal Unicode pieces and decodes byte-fallback runs
without applying GPT-2 ByteLevel conversion to already-Unicode text. It removes
one synthetic Indic marker using the reviewed router grammar, rather than
globally shrinking every space pair. It restores newline/tab/CR while preserving
the context that distinguishes some literal marker letters from controls.
Its prefix lookahead includes source newlines, which matter for Indic Markdown
tables and math separators. It preserves multiple leading spaces in code;
one genuine leading space can still collide with a synthetic prefix, so the
candidate renders the unindented representative of that ambiguous case.

The builder rejects an unrecognized router and a non-deterministic model.
`qualification.json` binds the original full-artifact hash, candidate hash,
the shared encoder-payload fingerprint and auxiliary file hashes. It records
source recovery, unchanged encoding and streaming consistency per case.

## Streaming

The stock Hugging Face `DecodeStream` can discard Latin context required by
the marker grammar. Loading the revised JSON alone does not qualify streaming
in a serving engine. The conservative reference adapter retains that context:

```python
from tokenizers import Tokenizer
from production.decoder_repair import ContextualDecodeStream

tokenizer = Tokenizer.from_file('/external/decoder-candidate/tokenizer.json')
stream = ContextualDecodeStream(tokenizer, skip_special_tokens=False)
for token_id in generated_ids:
    chunk = stream.step(int(token_id))
    if chunk is not None:
        emit(chunk)
emit(stream.finish())  # Required once, at EOS/end of generation.
```

The adapter delays undecided markers and entire pending byte-fallback runs.
It stops with an explicit error if previously emitted text would change.
It is a correctness reference that repeatedly decodes retained history, not
a claimed high-throughput serving path. Test the actual Transformers, vLLM,
SGLang or other output adapter before adopting its streaming integration.

## Remaining ambiguities and promotion

A decoder cannot recover both originals of identical-ID encoding collisions.
For ambiguous marker letters the candidate generally renders whitespace;
for ambiguous control-following spaces it uses the no-extra-space convention.
Some leading-space choices in mixed non-Latin text are also ambiguous.
The report records differing source outputs and whether re-encoding those
outputs yields the same IDs, providing a direct collision witness.

Targeted fixtures and recorded excerpts do not measure these patterns' corpus
frequency. `production_accepted` remains false. Closing acceptance requires a
representative original-source/stored-ID audit, review of the ambiguity policy
and qualification of the actual output stack. A newly discovered recoverable
failure must be repaired rather than categorized as an encoding ambiguity.

The revised file has a different full SHA256. Keep existing checkpoint and
corpus contracts pinned to their original artifact. Any serving/evaluation
adoption must explicitly bind both the original encoder lineage and the revised
decoder identity. Do not silently replace hashes to pass an existing gate.
