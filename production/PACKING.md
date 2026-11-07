# Document packing integration contract

The selected external data plan uses 4096-token sequences, one primary
language/script/source leaf per sequence, and multilingual mixing across global
updates. Its document boundaries, position resets and loss policy must be
implemented together. This is a candidate relative to the current continuous
stream trainer; no native acceptance is recorded here.

Generated allocations, source catalogs, audit outputs, sequence descriptors and
shards remain outside this repository. The current planning packet is under
`/Users/varundaiya/terminus-pretraining-readiness/2026-11-07/packing-readiness/`.
Its `PLAN.md` records the source acceptance, multilingual policy, packing
algorithm, target convention, code gaps and CPU/one-GPU/eight-GPU gates.

Required interfaces before enabling this mode:

- A document/fragment catalog with stable source and family identity, the exact
  tokenizer hash, primary language/source leaf, original quality metadata,
  explicit continuation lookahead and final-EOD semantics. All heldout families
  are excluded from training inputs, not just loss-masked.
- Deterministic sequence descriptors with segment offsets, positions and
  content/slot/valid-target counts; whole-record and tail receipts. The current
  `terminus.pack_tokens` command assembles manifests and does not implement this.
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
wait, keeping the continuous-stream path as the measured baseline.

Packing changes do not authorize tokenizer surgery, a new objective, automatic
LR scaling, unapproved data reuse or a paid GPU launch.
