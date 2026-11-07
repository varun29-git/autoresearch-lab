# CPU phase preparation queue

`preparation_queue.py` implements exact token tranches, dependency accounting and a single-task runner. `preparation_dispatcher.py` provides a serverless S3-backed dependency dispatcher. Run configurations, source manifests, state and outputs belong outside this repository.

The active external plan is [next-dense-queue/README.md](/Users/varundaiya/terminus-pretraining-readiness/2026-11-07/next-dense-queue/README.md). Its 750B dense-stable phase consists of 710B ordinary dense and a protected 40B bridge. QAT records are reserved before ordinary dense selection. These are data preparation tasks; the queue cannot launch GPU instances or authorize training.

`packing_worker.py` can inherit a hashed SQLite allocation ledger and quota-suffix manifest. It rejects implicit reuse of a partial work directory. Reservation mode protects whole records and emits a fixed selection manifest without pretending to create physical payloads. Fixed materialization checks that every record belongs to its declared protected allocation. Ordinary materialization resumes unused suffixes before selecting fresh records. Only explicitly configured source fallbacks may change an assigned row; the total budget is preserved and the exact change is reported.

The dispatcher validates predecessor quotas and report identities before binding their durable artifacts into a child plan. It uses a persisted launch request, an idempotent EC2 token, one reserved concurrent Lambda invocation, a CPU instance allowlist, fleet concurrency caps, terminate-on-exit workers and independent AWS termination schedules. Failure does not automatically start another paid attempt. A failed predecessor blocks descendants. The queue definition itself is SHA-pinned in the deployed Lambda.

Relevant CPU validation:

```sh
python3 -m unittest tests.test_document_packing tests.test_packing_worker tests.test_textbook_source tests.test_preparation_queue
```

Native training-loader integration, masked attention equivalence, loss normalization and deterministic distributed resume remain required by [PACKING.md](PACKING.md). Candidate output receipts explicitly record that no training manifest or native acceptance has been published.
