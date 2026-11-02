# Fixed QAT budget, adaptable dense plateau

The CPU tools implement prospective **stopped-checkpoint** transitions. They do
not edit a running manifest or autonomously orchestrate both phases. The human
still launches paid work. Keep the production warmup, global batch, optimizer
and quality policy fixed; additional affordable tokens go into dense training.

## Forecast without changing the experiment

Use `python -m production.amd.forecast --help`. Required inputs are the original
allocation record, current dense/QAT token counters, qualified fixed QAT token
budget, sustained dense and QAT **cluster tokens per node-second**, global
tokens/update, absolute dense cooldown length, finishing reserve and approved
maximum dense data budget. Include normal checkpoint/validation cadence in the
rates and subtract only additional finishing/outage work as reserve. Supply
`--dense-warmup-tokens` so an early forecast also funds remaining warmup.

The forecast rounds down to complete global updates, caps exposure at the data
budget, funds remaining QAT before extending dense, and rejects an unaffordable
ending. It never scales the LRs or changes a config. Rate provenance/freshness,
curriculum quality, QAT recovery length and the outage reserve still require
review; arbitrary user-entered rates are not measurement evidence.

## Commit a prospective child

Prepare an external child config/manifest with the qualified future data and
endpoint, then stop at a committed checkpoint while still before either old
or proposed cooldown. Use:

```bash
python -m production.amd.horizon \
  --parent-run /external/parent-run \
  --parent-manifest /external/parent-manifest.json \
  --child-config /external/proposed-config.json \
  --child-manifest /external/proposed-manifest.json \
  --data-root /external/staged-data --output-dir /external/child-run \
  --child-staging-receipt /external/child-staging.json \
  --commit-ending
```

Omit `--commit-ending` only for a still-adjustable plateau. Once an ending is
committed, the tool rejects another same-phase horizon change even if decay
has not started. Default `--data-policy continue` requires stream continuation;
an explicitly reviewed `phase_mix` permits future routing changes while proving
the consumed prefix unchanged. The parent must be stopped and source-identical.
This tool cannot authorize a source/runtime migration of an old H100 checkpoint.

The child carries weights, optimizer state and rank RNG unchanged, proves the
past LR schedule and consumed data, and writes a new contract/receipt. It can
rebase microsteps for a different microbatch/accumulation pair at the **same
global batch**. Resume with the child's config/manifest/checkpoint and matching
native receipts. A failed transaction leaves an unlaunched partial directory;
inspect it and create a fresh output rather than treating it as a valid child.

Retain the same allocation ledger through every child. Use a separate durable
backup prefix for each run contract and preserve the parent. The supervisor
currently stops the provider when its training command terminates, so a
supervised one-phase command is not a multi-phase campaign orchestrator. Do not
expect a horizon fork or dense-to-QAT switch to happen automatically at exit.

For QAT use the existing explicit `terminus.transition` transaction and native
acceptance gates. Freeze a qualified absolute QAT token budget; do not copy the
historical short HPO warmup/decay or treat a constant QAT budget as proof of
sufficient recovery. If the remaining allocation cannot fund the commitment,
preserve the last durable artifact and report the unfinished phase.
