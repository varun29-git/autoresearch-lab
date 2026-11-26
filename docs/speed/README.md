# Stack 4 dense-speed research archive

This directory records the completed single-H100 dense-throughput campaign on the 25-layer, 1,500,456,496-parameter Stack 4 shape. It does not describe an active campaign or authorize cloud work. For the release-wide evidence boundaries, start with [RESULTS.md](../RESULTS.md) and [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## Auditable results

- [E3/E4/E9 dense round](trials/e2-e3-e4-e9-dense-round.md): standalone E3 reached 40,187 valid tokens/s versus 36,527 in the 20-update confirmation, +10.0%. E2 was rejected; E4 and E9 were retained in the combined stack.
- [Paired-run summary](TRANSFER_32GPU.md): three paired 20-update runs averaged **40,996 tokens/s** for E3/E4/E9 versus **36,723** for the incumbent, +11.6%, with 55.5 GB peak torch allocation.
- An archived 80-update log line reports 41,075 tokens/s, but its full JSON receipt is unavailable. Use the complete paired receipts as the auditable evidence, not the shorter 41,936-token/s screen.

The measurements cover complete dense optimizer updates on one H100 NVL with a real staged English stream. They do not validate QAT, 32-GPU throughput, final model quality, or production token-budget feasibility. The 32-GPU document is an analytical scenario only.

## Research map

- [MEASUREMENT.md](MEASUREMENT.md) defines the complete-update timing boundary and paired comparison policy.
- [VALIDATION.md](VALIDATION.md) records the available numerical evidence and missing receipts.
- [EXPERIMENTS.md](EXPERIMENTS.md) is the original hypothesis atlas; [PENDING_5H.md](PENDING_5H.md) is a historical planning snapshot, not the current release status.
- [PHYSICS.md](PHYSICS.md) derives optimistic hardware bounds; these are not attainable-throughput claims.
- [SOURCES.md](SOURCES.md) and [PRODUCTION_IMPLEMENTATIONS.md](PRODUCTION_IMPLEMENTATIONS.md) distinguish external implementations from Terminus measurements.

Later N1/N2/E16 and QuACK source experiments are retained for review but are not promoted as validated winners without matched-token quality and sustained full-update evidence. `speed_program.md` and the root `KIMI_START.md` are archived protocols. Raw result receipts, data, checkpoints, and session configuration remain outside Git.

Focused local protocol tests, without VM allocation:

```bash
python -m unittest tests.test_speed_session tests.test_s0_block tests.test_speed_s0 tests.test_speed_research tests.test_hopper_candidates
```

The full release verification commands are in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). CPU tests and skips are not native-H100 validation.
