# CPU phase preparation queue

`preparation_queue.py` implements exact token tranches, dependency accounting and a single-task runner. `preparation_dispatcher.py` provides a serverless S3-backed dependency dispatcher. Run configurations, source manifests, state and outputs belong outside this repository.

The active external plan is [next-dense-queue/README.md](/Users/varundaiya/terminus-pretraining-readiness/2026-11-07/next-dense-queue/README.md). It covers the full main 1T: 50B warmup, 710B ordinary dense, a protected 40B bridge and 200B QAT physical output. QAT records are reserved before ordinary dense selection. These are data preparation tasks; the queue cannot launch GPU instances or authorize training.

The current CPU cap is five concurrent workers in total. Protected QAT
materialization depends on its own immutable reservation, so it can overlap
later reservations. The first ordinary dense tranche also separates reservation
from physical packing, keeping packing off the subsequent selection chain.
Exact per-source physical token totals remain unchanged. The original running
warmup must still publish its final allocation ledger before this pipeline
can select records; increasing the cap does not distribute that existing job.

`packing_worker.py` can inherit a hashed SQLite allocation ledger and quota-suffix manifest. It rejects implicit reuse of a partial work directory. Reservation mode protects whole records and emits a fixed selection manifest without pretending to create physical payloads. Fixed materialization checks that every record belongs to its declared protected allocation. Ordinary materialization resumes unused suffixes before selecting fresh records. Only explicitly configured source fallbacks may change an assigned row; the total budget is preserved and the exact change is reported.

The dispatcher validates predecessor quotas and report identities before binding their durable artifacts into a child plan. It uses a persisted launch request, an idempotent EC2 token, one reserved concurrent Lambda invocation, a CPU instance allowlist, fleet concurrency caps, terminate-on-exit workers and independent AWS termination schedules. Failure does not automatically start another paid attempt. A failed predecessor blocks descendants. The queue definition itself is SHA-pinned in the deployed Lambda.

Relevant CPU validation:

```sh
python3 -m unittest tests.test_document_packing tests.test_packing_worker tests.test_textbook_source tests.test_preparation_queue
```

Native training-loader integration, masked attention equivalence, loss normalization and deterministic distributed resume remain required by [PACKING.md](PACKING.md). Candidate output receipts explicitly record that no training manifest or native acceptance has been published.

Dense metadata reservation and physical materialization can be pipelined. Dense reservations may explicitly carry unused protected suffixes into a successor: each immutable ledger binds the allocation that owns that span, and materialization preserves its original document offset. Fixed source ranges split protected QAT selections into disjoint balanced jobs. Tests verify that split payloads reconstruct the original span and that suffix transitions neither replay nor skip content.

A `drained_outputs` ledger records fully consumed/excluded source indexes and their token-payload hashes. Successors skip those indexes without rescanning previously allocated records. Bounded SQLite cache settings support the provisioned CPU RAM. Multipart ledger artifacts retain their composite upload evidence while dependency checks compare full-file SHA-256, byte length and URI.

## Parallel worker and checkpoint boundary

`fast_packing_worker.py` preserves the reference worker's quota decisions,
source/row ordering, packing seeds, suffix offsets, EOD and target semantics.
The runner accepts this module only through an explicit allowlisted
`worker_module` setting. The reference worker remains available for comparison.

The fast path uses a compact SQLite allocation ledger in RAM, 32-byte source
and row-identity hashes, interned source/allocation names and batched inserts.
It migrates a legacy predecessor once, and writes a closed snapshot after
the task finishes. The snapshot must pass SQLite integrity and exact record
count checks before it becomes a published dependency. Original full row
identities remain in the source and segment manifests. Compact ledgers require
the fast worker for their successors.

Working SQLite state uses `synchronous=OFF`, `journal_mode=MEMORY`,
`temp_store=MEMORY` and `locking_mode=EXCLUSIVE`. This is disposable scratch
state, not a resumable checkpoint. If it exceeds the assigned RAM budget,
the worker spills to a bounded file cache. A crash discards the task's working
state; it cannot advance descendants without a verified completion report.
Independent instance termination deadlines and the no-paid-retry policy
remain in force.

On a 16-vCPU, 128-GiB worker the starting budgets are 64 GiB for ledger pages,
8 GiB for the optional membership set, 8 GiB for prefetched input files and
12 GiB for prefetched index metadata. Packing and metadata process pools
use eight and four processes respectively; S3 transfers use bounded threads.
Individual oversized input files spill sequentially to EBS. Packing chunks
and completed shard buffers use `/dev/shm`, with 64-MiB buffered writes.
Only the ordered coordinator changes quotas and deduplication ownership.

Inputs are hashed during download. Packing processes hash encoded chunks;
the coordinator validates their hashes while assembling the original shard
boundaries. Uploads use those digests and verify S3 checksum evidence,
avoiding another complete local file read. Source receipts are consolidated
into a buffered audit archive. These changes reduce small random EBS writes
and overlap source preparation, packing and publication.

The CPU equivalence tests compare every payload and segment-sidecar SHA-256
with the reference, including Unicode identities, duplicate/heldout records,
long-document cuts, protected fixed ranges and cross-chunk shard boundaries.
Additional checks cover legacy migration, bounded spill, invalid checkpoints
and rejection of corrupted prefetched payloads:

```sh
python3 -m unittest tests.test_fast_packing_worker tests.test_document_packing tests.test_packing_worker tests.test_preparation_queue tests.test_native_corpus.MultipartPublicationTests
```

The fast runtime requires `orjson==3.11.4`. Performance measurements, queue
definitions and benchmark source subsets remain outside Git. A small benchmark
does not establish the finish time of the complete corpus, especially when the
global ledger grows and several workers share S3/network bandwidth.
