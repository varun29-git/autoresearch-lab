# Observations, stop rules and comparison

Implement this before the scientific trials. See the detailed interpretation
and reference requirements in
[LR_BATCH_MONITORING.md, sections 4–6](../amd/pretraining_readiness/LR_BATCH_MONITORING.md).
Freeze definitions, tolerances, held-outs and operational margins in the
external launch packet. No monitoring hook may alter the training trajectory.

## Required observations

| Cadence | Record/check | Why |
| --- | --- | --- |
| Every complete update (`log_interval=1`) | Loss sum, nominal and actual valid-target counts, denominator, token-mean loss, every group LR/role, schedule scale, update/token/cursor counters, phase and allocated hours | Catch masking, accumulation, scheduling and transfer mistakes |
| Every update | Global gradient norm before/after clip, coefficient, required-gradient presence, finite gradients/parameters/moments using fused/chunked complete coverage | Finite sampled loss does not rule out broken state or omitted parameters |
| Every update/time sample | Complete-update duration, data wait, driver/allocator HBM, clocks/power/temperature, device errors | Separate compute, data and learning problems; catch memory cliffs |
| Every 20 updates and exact schedule boundaries | Logical Q/K/V/O/gate/up/down and all Adam groups: weight/gradient RMS, direction-only update, decay, total actual delta, delta/weight RMS, direction cosine and compared-update IDs | Clipping raw gradients does not bound normalized Muon updates |
| Diagnostic points | Per-layer Peri/final/norm and Q/K gains, learned value coefficients, residual/attention/FFN RMS; tied-table full summaries and fixed frequency cohorts | Identify scale drift, ineffective learning and embedding instability |
| Rotating diagnostic samples, plus anomalies | Q/K norms, scaled attention-logit tails, entropy/concentration with declared query/key sample support and masks | Distinguish local/global behavior; sampled maxima are not global maxima |
| Init, warmup exit, pre-decay, anomalies | Actual native FP8 operands/scales/zero and clamping counts, cache versions, gradients and full-update reference deltas | Catch numerical failure and stale weight caches that resemble an LR issue |
| Every 48 updates, plus boundaries | Fixed quick held-out slice; identical evaluator/examples for every arm | Compare learning at matched tokens, without noisy moving evaluation sets |
| Init, warmup exit, pre-decay, final | Frozen larger held-out NLL by domain/language/source and aggregate, with valid counts | Separate early/plateau behavior from cooldown ranking |
| Every checkpoint and periodic watchdog | Payload/receipt hashes, durable copy state, backup queue bytes/age, free disk, checkpoint age, budget/time remaining | Detect local disk exhaustion, failed uploads and expensive lost progress |

A fixed initial **safety** evaluation is distinct from the reserved final
selection holdout. Keep the latter unused until the declared selection step;
do not repeatedly consult it while changing configurations.

For native references, branch from identical complete state and identical
examples at initialization, warmup exit and pre-decay. Preserve weights,
optimizer, schedule, RNG, caches and data cursor; exclude branch updates from
main-run tokens. Compare actual gradients/deltas with independent optimizer
and kernel references and a BF16 precision control. BF16 is not an exact FP8
rounding oracle. Predeclare tolerances from independent controls, not the
observed candidate result. Measure diagnostic overhead and coverage.

## What the existing monitor does not establish

`terminus.stability` samples rank zero and the first microbatch, at most 16
activation tokens, 128 attention keys and 256 parameter entries per tensor.
It skips missing gradients and recomputes quantization diagnostics instead
of observing actual cached native operands. Existing train logs expose
minimum/maximum group LR, and their loss/clip values describe the **last
update**, not the entire preceding log interval. These are useful components,
not receipts for the complete requirements above.

Use absolute deltas and denominator floors for initially near-zero gains and
value coefficients. Separate direction change from `(1 - eta_t * wd)` decay:
changing Muon eta at fixed decay changes shrinkage too. Record cumulative
`log(1 - eta_t * wd)` without claiming it equals the actual weight norm change.
Tied head gradients may update rows absent from the input tokens. FP8 largest
finite codes may occur from normal block scaling; distinguish boundary
occupancy from actual pre-quantization clamping and nonzero underflow. Check
NS5 against the pinned approximate algorithm, not exact SVD orthogonality.
Do not add allocated and reserved PyTorch bytes: reserved contains allocated.

## Actions on failures and warnings

**Fail the current attempt immediately** on nonfinite loss/gradients/master
weights/moments, missing required gradients, invalid loss normalization/data,
stale or incompatible identity, corrupt checkpoints, uncorrectable hardware
errors, breached storage/budget safety limits or a failed mandatory native
gate. A documented intentionally absent gradient is the only exception to
the required-gradient check. Preserve the last valid durable state, error,
update and failing batch identity. Exit nonzero and notify the human.

Do not silently skip data, reset state, lower LR, change optimizer groups,
switch precision, turn off a failed check or continue from a corrupted state.
A code/config fix needs a new identity and a reviewed restart/comparability
decision. Failed attempts still count against the allocation.

**Review alarms**, starting after warmup where applicable:

- Clip coefficient below 0.1 on more than 20 of the last 100 updates.
- A sustained role/logit/FP8 statistic above three times its established
  comparable healthy range, using the same measurement definition.
- Held-out degradation beyond the declared domain margin at consecutive checks.
- Backup lag, low disk/HBM headroom or slower cost forecasts approaching the
  packet's declared limits, before they become hard failures.

These are engineering alarms, not universal LR correctness thresholds. Freeze
the persistence window/ranges and review action before trials; if an alarm
has no safe automated resolution, stop at a valid boundary for review.
Do not invent a healthy range or widen thresholds after seeing a failure.
Avoid copying Kimi K2's attention-logit threshold or full-mask entropy limits
to a different parameterization or a 128-key sample.

## Selection

Use complete healthy trials, matched data/tokens and the same evaluation
implementation. Report train and held-out NLL versus both tokens and allocated
hours, per-domain results, pre-decay and final cooldown values, and stability
diagnostics. Use predeclared practical superiority/noninferiority margins.
Pair uncertainty analysis by document or justified blocks. Document-bootstrap
intervals capture evaluation noise, not training-seed variance or selection
bias. Preserve a separate final holdout.

If the evidence cannot separate candidates, report a tie and an explicitly
provisional transfer choice. Do not force a statistically proved winner from
one seed. R0–R3 is a crossed comparison, not proof of an interior optimum.
Failure after the 238-update warmup implicates that LR/warmup combination;
it does not prove the peak fails with a longer production warmup. A good final
cooldown cannot excuse deteriorating plateau behavior. Do not promote a
candidate solely for throughput or lowest sampled training loss.
