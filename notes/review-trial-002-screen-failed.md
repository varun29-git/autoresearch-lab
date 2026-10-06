# Trial diagnostic review: trial-002 (value residual) / screen / cycle 20261006T144845Z0fb462ec — implementation failure (0 tokens)

## Outcome and evidence coverage

- Commit `902b606`, 0 tokens, 0.0102 GPU-h, 0.54 VM-on h. Archive verified (`2b8116aa…`), clean deallocation. Read: `train_screen.log` traceback, `incidents.jsonl`, `preflight.json` (passed), worker journal.
- **Failure:** the trial's `BitNetAttention.forward` returned a `(output, values)` tuple when `value_residual` was enabled. The protected `StabilityMonitor` hooks every attention module's output and requires a plain tensor (`stability.py:78`: `'tuple' object has no attribute 'shape'`) on the first training forward.
- **Why the pre-flight gates missed it:** all unit tests passed, but none ran the *training loop with the stability monitor active* on the candidate config — the monitor only attaches when `stability_log_interval > 0`, which the test fixtures never set. Native preflight passed because it runs model numerics without the monitor.
- **Classification:** implementation error, not an architectural result. The mechanism was never measured.

## Fix and gate change

- Fix (commit `06b22d9` on the same branch): attention always returns a plain tensor; layer-0 values pass to the model via a module attribute; `value_residual` now rejects `gradient_checkpointing=True` (attribute stash vs recompute). Stack branch fixed identically (`7015a25`).
- **New permanent pre-launch gate:** dev-mode training-loop smoke with `stability_log_interval=1` on the exact candidate config (regression test added to `tests/test_architecture_vres.py` and `tests/test_architecture_stack.py`; reproduces the crash on the old code path by construction).
- Relaunched as **trial-005** per the rule that a fixed implementation gets a new commit and a new trial ID. The trial-002 record stays in the ledger.

## Cost ledger

Third 0-token pre-training failure overall (two were locked-harness/staging issues, one candidate bug). 0.54 VM-h spent. Each failure has produced a permanent gate; the failure classes so far are disjoint.
