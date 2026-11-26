# Document packing integration contract

The selected external data plan uses 4096-token sequences, one primary
language/script/source leaf per sequence, and multilingual mixing across global
updates. Its document boundaries, position resets and loss policy must be
implemented together. Cross-document attention masking is a fixed production
requirement, confirmed by the operator on 7 November 2026. The
[packed training implementation](PACKED_TRAINING.md) now connects the reader,
attention, loss and resume interfaces. Native acceptance remains unrun here.
The legacy continuous-stream path cannot consume this format.

**2026-11-08 read-order correction:** physical creation remains grouped by
source and dataset stratum. `packed_mixture.py` now provides a CPU reader that
interleaves language, aggregate source and individual dataset children, with
exact finite sequence counts and deterministic bounded permutations. This
prevents a Hindi Specialized aggregate from being read as one Code-Concepts
block followed by one STEM-SFT block. The legacy router remains sequential
within a source and is incompatible with this packed-format candidate.
The reader is now connected to native variable-length attention and the
trainer; native execution checks and graph replay qualification remain separate.

`packed_index.py` builds descriptor byte offsets directly from already published
sidecars, independently of the allocation ledger. The first four-worker
campaign indexed 44.759B tokens across 455 warmup shards; its receipts remain
external. Future fast workers emit offsets during packing and publish a
`MIXTURE_CANDIDATE.json` only for a completed, quota-satisfied physical allocation.
This preserves existing payloads, boundaries and source quotas. A candidate
read-order artifact is not a native training acceptance report.

Generated allocations, source catalogs, audit outputs, sequence descriptors and
shards remain outside this repository. The current planning packet is under
`/Users/varundaiya/terminus-pretraining-readiness/2026-11-07/packing-readiness/`.
Its `PLAN.md` records the source acceptance, multilingual policy, packing
algorithm, target convention, code gaps and CPU/one-GPU/eight-GPU gates.
The later `../warmup-shards/SOTA_PLAN.md` and
`CURRENT_ALLOCATION_OVERRIDE.json` take precedence for the selected replacement
sources and mandatory masking policy. Neither is a completed shard receipt.

Required interfaces before enabling this mode:

- A document/fragment catalog with stable source and family identity, the exact
  tokenizer hash, primary language/source leaf, original quality metadata,
  explicit continuation lookahead and final-EOD semantics. All heldout families
  are excluded from training inputs, not just loss-masked.
- Deterministic sequence descriptors with segment offsets, positions and
  content/slot/valid-target counts; whole-record and tail receipts. The CPU
  implementation is now in `production.document_packing` and
  `production.packing_worker`, described in [DOCUMENT_PACKER.md](DOCUMENT_PACKER.md).
  The older `terminus.pack_tokens` command only assembles stream manifests.
- Variable-length AMD causal attention with the original local window and model
  semantics, plus reference and native forward/backward checks. A plain EOD
  token does not block attention to earlier documents.
- Global valid-target planning before backward. Retain the trainer's
  count-mismatch guard and DDP summed-loss normalization. Nominal packed slots
  must not silently replace the scored-target schedule clock.
- Integer source quota accounting and checkpointed consumption state, without
  0.001 rounding that can erase small sources. Changing rank count must not
  duplicate or skip source records.
- Explicit graph/checkpoint handling for boundary and position metadata. The
  existing captured microstep only updates input IDs, targets and loss scale;
  do not enable it for dynamic document boundaries until replay parity passes.

Document isolation must first be tested with a BF16 reference. FP8 activation
scales can depend on other packed documents; distinguish that numerical effect
from actual cross-document attention. Compare native paths with justified
tolerances and measure useful scored targets/s, full-update memory and data
wait. Historical continuous-stream measurements are comparison data only.

Keep complete short documents intact. Split documents exceeding context into
contiguous windows, then best-fit-pack their final fragments with short
documents without further splitting. All segments retain original document
identity and offsets; only the real document ending emits EOD. This is the
selected default for STEM as well as other pretraining documents. Length
alone does not disqualify a source. No repeated question prefix or overlapping
scored tokens are introduced. Earlier-window context is unavailable after a
window boundary, as in ordinary finite-context pretraining.

At an exact source-quota boundary, reserve the unused document suffix for
later selection, mark the fragment as incomplete, and mask its final label.
The loss counter must reflect this boundary rather than assuming content
tokens always equal valid targets. The ordinary full-window continuation
uses a label-only next token from the same selected span. Source acceptance
and native attention acceptance remain distinct from CPU format verification.

Packing changes do not authorize tokenizer surgery, a new objective, automatic
LR scaling, unapproved data reuse or a paid GPU launch.
