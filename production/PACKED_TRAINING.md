# Packed-document training integration

The trainer now consumes the existing packed payload, descriptor and offset
artifacts together. This implementation preserves the selected 24-layer
architecture and optimizer settings. CPU correctness and native hardware
acceptance are separate: no MI325X throughput or month-long stability result is
claimed by this change.

## Attention and objective

`terminus.document_attention.DocumentLayout` validates contiguous segment IDs,
position resets and tail padding on the CPU. Prefetch pins and transfers the
layout with its input IDs and pre-shifted targets. All layers, including
activation-checkpoint recomputation, receive the same layout.

Local layers retain an inclusive 512-token window; global layers retain the
4096-token context. AMD uses the pinned AITER/CK `flash_attn_varlen_func`
autograd interface with cumulative document lengths, causal attention and GQA.
There is no batch-by-length-by-length mask allocation. Full sequences avoid
padding gather/scatter; partially padded sequences gather only valid tokens.
Per-document RoPE resets remain inside the fused learned-Q/K forward and gain
backward kernels. Ordinary continuous inference/cache behavior is unchanged.

An already qualified unlabeled translation pair occupies **one segment**;
its second passage can attend to its first. Other documents remain isolated.
The reader neither constructs new pairs nor changes the 20B allocation ceiling.
Attention stability samples now exclude unrelated documents, padding and keys
outside the actual local window.

Before backward, each rank reads descriptor accounting for the entire next
optimizer update. A single all-reduce establishes the global valid-label count
and content/input-slot counts. Loss sums use `world_size / global_valid_labels`
so DDP averaging produces the correct global mean, including unequal lengths
and an empty tail rank. A discrepancy or nonfinite loss/gradient stops before
the optimizer changes weights or moments.

## Manifest and clocks

`production.build_packed_manifest` converts **ordered, completed** mixture
candidates into a training manifest. A source-acceptance copy must bind each
candidate's `tokenizer_sha256` to the frozen tokenizer contract. This field is
not inferred from the vocabulary or payload dtype. Provide heldout candidates
with the same tokenizer and the existing family exclusions. Domain evaluation
keys can be supplied explicitly per payload URI; do not infer math/code
coverage from a generic English label.

The manifest uses `provenance.packing = "document_segments_v1"` and a versioned
`packed_stream` containing train/val candidate lists and local artifact bindings.
Each candidate retains its finite language/source/dataset-child permutation.
Appending an immutable **new phase** preserves the existing order; adding
shards to a consumed phase would reshuffle it and is rejected by continuation
proofs. Legacy continuous-stream routing is never used for this format.

- `config.total_train_tokens`: complete **input slots**, used for sequence cursors.
- `manifest.scored_train_tokens`: optional lifetime offset plus actual labels,
  used for the LR schedule and useful throughput.
- `input_slots_seen`: real sequence slots consumed in this stream, excluding
  dummy rows assigned to empty ranks.
- `content_tokens_seen`: original source content consumed in this stream.
- `tokens_seen`: actual labels consumed, including the lifetime offset.

The requested global batch remains an input-slot batch. Logs expose actual
scored counts; no LR scaling is introduced automatically. Resolve warmup and
cooldown labels from accepted packing receipts. Source-content quotas remain
unchanged, including the 100B QAT finishing-data allocation.

Full CPU acceptance hashes all three artifacts, checks vocabulary/reserved IDs,
reconciles every descriptor and source/component charge, and rejects duplicate
payloads across train/heldout splits. Production verification uses a bounded
CPU process pool (default at most 16 processes), sequential descriptor traversal
and the pinned Rust JSON decoder. `CI1_PACKED_VERIFY_WORKERS` sets the bound.
Use `production.amd.staging` to seal a read-only local snapshot once; training
then checks snapshot stats and avoids rehashing corpus files per GPU/rank or
rechecking complete offset tables after an LRU reopen. File stats stay outside
the portable checkpoint identity.

Example CPU preparation, with all artifacts outside Git:

```sh
python -m production.build_packed_manifest \
  --base-manifest /data/frozen-source-manifest.json \
  --train /data/accepted-warmup.json /data/accepted-dense-01.json \
  --validation /data/accepted-heldout.json \
  --paths /data/uri-to-local-path.json \
  --evaluation-keys /data/payload-evaluation-keys.json \
  --config /runs/ci1/config.json --data-root /data \
  --output /runs/ci1/packed-manifest.json
```

The config must already contain the selected model and intended microbatch.
Generate final token-clock schedules after acceptance. The command verifies
CPU artifacts; it does not launch a VM or training.

## Dense to QAT

For a separate QAT stream, seal its manifest with `--tokens-before` equal to
the stopped dense checkpoint's **actual** `tokens_seen`. Generate QAT config
with that same `phase_start_tokens`, retaining the selected QAT rates and the
finishing cooldown. Use the existing CPU transition command with
`--data-policy packed_phase`.

This policy requires a completed packed dense parent, the same tokenizer and
no reused parent payload. It starts the QAT sequence cursor at zero, retains
the lifetime label offset, model weights, optimizer moments and rank RNG, and
records local cursor/update-counter rebasing in the receipt. A partially filled
final dense update is supported without skipping the first QAT sequences.
The original `continue`/`phase_mix` policies also support immutable packed
phase prefixes when the carried cursor aligns with complete global updates.
Each transition remains an explicit operator action.

## Acceptance before a paid production run

CPU tests cover independent-document loss/gradient equivalence, Peri-LN and
learned value residual with checkpointing, causal/local boundaries, pair
visibility, padding, exact resume, two-rank unequal-label DDP, accounting
failures before optimizer mutation, parallel verification, and QAT handoff.

The AMD preflight now requires native variable-length attention
forward/backward and isolation checks, plus shared and per-batch learned-Q/K
RoPE checks. Old receipts become stale through source hashes. The native test
is skipped without gfx942 hardware.

On the intended cluster, still run the new preflight, then the real packed
full-size update/memory/cadence test, eight-rank parity/recovery and mature
dense-to-QAT qualification. Validate the tokenizer-v2 heldout source exclusions
and evaluation manifests before launch. CPU passes do not establish these
hardware or long-horizon outcomes. Dynamic packed metadata explicitly rejects
compile/CUDA graphs until replay qualification is implemented; the selected
AMD eager configuration already leaves those disabled.
