# Trial diagnostic review: trial-005 (value residual, first fix) / screen / cycle 20261006T153203Za2d04f4d — implementation failure (0 tokens)

## Outcome and evidence coverage

- Commit `06b22d9`, 0 tokens, 0.0044 GPU-h, 0.53 VM-on h. Archive verified (`5a296127…`), clean deallocation. Read: `preflight_screen.log` traceback, worker journal, cycle_result.
- **Failure:** my trial-002 fix added a config-level rule rejecting `value_residual` + `gradient_checkpointing`. Native preflight builds a *reduced* model with `gradient_checkpointing=True` (preflight.py:34) to verify checkpointed-vs-retained parity, so the candidate config could not even be constructed. Crash at 15.7s, before any numerics.
- **Root cause is mine:** I forbade a combination instead of making it correct, and did not run the preflight path locally against the candidate config. The mechanism still has never been measured.
- **Classification:** implementation error, second attempt. Not an architectural result.

## Fix and gate change

- Correct support instead of a ban (commit `ee71e32`): layer 0 runs uncheckpointed when value_residual is on (its stashed V1 keeps gradients); later layers checkpoint normally with V1 as a tracked region input — non-reentrant checkpoint semantics make this exact. The stack branch got the same fix (`8bd5523`).
- **New permanent gate:** CPU checkpoint-parity regression test mirroring the native preflight (loss + every parameter gradient, checkpointed vs retained) in both `tests/test_architecture_vres.py` and `tests/test_architecture_stack.py`.
- Pre-launch gate list is now: full CPU suite, protocol envelope, optimizer construction on the candidate config, data-path replica, monitor-active training-loop smoke, checkpoint-parity test. All pass on `ee71e32`.
- Relaunched as **trial-006** (new commit, new ID per rule).

## Cost ledger

Fourth 0-token cycle (0.53 VM-h). The value-residual idea remains unmeasured; two of its three failures were the same underlying mistake (insufficient pre-flight coverage of harness-integration paths), now both covered by tests that run in seconds on the controller.
