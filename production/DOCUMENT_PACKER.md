# CPU document packer

`production.document_packing` splits only documents longer than the selected
context and best-fit-packs the fragments and intact shorter documents. The
bounded integer-capacity algorithm operates on descriptors; NumPy copies the
uint16 payloads. Completed packs are deterministically shuffled. This follows
the practical approach in [Fewer Truncations Improve Language Modeling](https://arxiv.org/html/2404.10830v2).

`production.packing_worker` consumes external source jobs, curriculum quotas,
and a frozen heldout-family exclusion list. It verifies native conversion
receipt identities and downloads/hash-checks each selected payload and index.
It excludes heldout identities and exact-text duplicates before packing.
Source subsegment quotas use deterministic integer allocation; source jobs
and rows are reproducibly shuffled. Candidate payloads, descriptors, source
receipts, counters and suffix reservations are published outside Git.

Each source allocation is exact in **content inputs**. A quota may end within
one document. Its suffix is recorded for a later phase, and no false EOD or
future-phase lookahead target is introduced. During ordinary document
chunking, a full fragment's last label is the next token from that same
selected span. Short complete fragments contain EOD as input with its label
masked; a full-length document ending uses EOD as the label-only lookahead.
Every intended next-token target is scored once. Content inputs, valid
targets, padding, occupied slots and EOD inputs are distinct counters.

The descriptor sidecar defines segment boundaries, document offsets, final
labels, end-of-document state and original source identity. Position IDs
reset per segment. The local attention window is 512 positions including
self, matching the existing native `(local_window - 1, 0)` convention;
global causal attention stays inside each segment. CPU target/position/mask
oracles are provided. Never construct the dense CPU reference mask in the
GPU training path; use variable-length cumulative boundaries there.

`production.textbook_source` reads the pinned UltraData-Math textbook Parquet
`content`/`uid` schema, applies the pinned English language classifier and
native tokenization, and retains source coordinates and content hashes. It
uses a deterministic content-hash validation split for fresh records. This
does not replace the later frozen evaluation-family audit.

Run CPU verification with:

```sh
python3 -m unittest tests.test_document_packing tests.test_packing_worker tests.test_textbook_source
```

The tests cover long-document and exact-window target coverage, short-document
isolation, padding, local/global masks, exact quota suffixes, tokenizer control
ID rejection, integer rare-source allocation, heldout exclusion, payload
publication and pinned raw textbook conversion.

Physical source/stratum order is not the intended training order. The CPU
`PackedMixture` and `PackedSequenceReader` in `production.packed_mixture`
hierarchically interleave language, aggregate source and dataset child using
integer finite populations. They shuffle bounded global read windows, shard
order and rows inside a shard, while visiting every sequence exactly once.
Small sources do not disappear through 0.001 weight rounding. Content-token
budgets, valid-target counts and padded sampling slots remain distinct.

The reader requires a verified descriptor-offset table for each shard.
`production.packed_index` creates these independently from existing sidecars;
new fast workers write them during shard assembly. Completed fast physical
allocations publish a `MIXTURE_CANDIDATE.json` binding the exact population and
all payload/descriptor/offset hashes. This candidate requires the new
segment-aware reader and is incompatible with the legacy continuous stream
loader. The source-selection seed and record ownership are unchanged.

Additional CPU checks:

```sh
python3 -m unittest tests.test_packed_index tests.test_packed_mixture
```

These cover arbitrary Unicode descriptor chunk boundaries, corruption,
finite permutations with no replay, rare-source retention, child mixing in
read windows, rank-independent resume, and preserved targets/segment masks.

The existing training loader and AMD attention path still need integration
with this format. Graph replay, distributed valid-target normalization and
native forward/backward remain unqualified. Candidate creation does not
publish a training manifest. Cross-document masking is mandatory for launch.

After interruption, durable candidate shards retain their provenance. The
worker refuses an implicit restart from a nonempty local selection ledger;
an explicit resume/replay plan must bind published parts and reconstruct
selection state before continuing. It does not quietly repeat a paid job.
