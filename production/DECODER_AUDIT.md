# Diagnose the deployed Terminus HF decoder

`production.decoder_audit` is a read-only CPU tool. It diagnoses the exact
external HF tokenizer artifact used for corpus encoding; it does not require
that its cuts equal the older CI-1 Python encoder.

```bash
python -m production.decoder_audit \
  --tokenizer /external/deployed-hf-tokenizer/tokenizer.json \
  --output /external/decoder-analysis.json
```

The report binds the tokenizer hash and installed `tokenizers` version and
traces every decoder stage. It compares raw text and the deployed NFC/quote/
dash/space normalization policy before routing/marker transformations, so
intentional normalization is not counted as a decoder defect.

Targeted cases exercise literal accents, mixed Indic/English text, byte fallback,
newlines/tabs/carriage returns, code indentation, special-token rendering and
repeated spaces. The tool also checks whether distinct policy-normalized input
strings produce identical IDs and whether appending IDs changes the already
decoded prefix. These are diagnostic fixture results, not a corpus defect rate.

Three decoder-only ablations retain identical vocabulary and encoder behavior:
removing Fuse, removing ByteLevel, and replacing ByteLevel with literal UTF-8
plus explicit control-marker restoration. Their failures are recorded too.
They are experiments, not supported production replacements. Promote a decoder
only after whole-document, streaming and source/stored-ID acceptance checks.

An identical-ID collision between different normalized source strings proves
that a decoder cannot recover both originals from IDs alone. Source/index
provenance is required to decide targeted repair. Such a collision does not
establish its frequency in the production corpus or justify blanket
retokenization. Keep existing bins immutable during diagnosis.

The expected-policy boundary is detected from the deployed artifact's first
Latin/Indic router rule. The report records the prefix length; if that boundary
is absent, it falls back to raw-text comparisons rather than guessing a new
normalization policy. Reinspect policy after changing the artifact.

`tokenizers` must be installed in the analysis environment. Reports, source
samples, candidate tokenizer files and measurements belong outside Git.
