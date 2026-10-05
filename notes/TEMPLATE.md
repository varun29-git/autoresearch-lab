# Architecture trial: <ID>

This note is written **before** the training result. Keep later observations in a separate dated section; do not rewrite the prediction after seeing the score.

## Evidence and applicability

- Terminus observation: exact run/checkpoint/log, measured signal, uncertainty, and why the current architecture may cause it.
- External evidence: primary paper and section/table; implementation URL and pinned commit; model size, data, precision, training budget, metric, and license. Say which differences limit transfer to Terminus.
- If the mechanism is original: derive it below and attach an existing empirical analogue or a reproducible cheap diagnostic. Do not present an unmeasured conjecture as evidence.

## Mathematical specification

- Forward computation: equations with tensor shapes, axes, masks, normalization epsilons, scales, positional ordering, and train/inference differences.
- Backward computation: where gradients flow, any surrogate or stop-gradient, expected update scale, and which optimizer group owns each new parameter.
- Initialization and stability: derive the predicted variance/logit/residual behavior with explicit assumptions and bounds or approximations.
- Cost and transfer: parameter count, training and prefill/decode FLOPs, activation memory, KV-cache bytes per token, FP8 cast/scale points, and expected ternary representability or quantization error. State what changes at dense-to-QAT transition and how checkpoint/optimizer state remains valid.
- If this is a multi-part redesign, explain which parts are coupled and name a cheap control that could separate their effects.

## Prediction and experiment

- Predeclare the expected change in English held-out loss and learning-curve area, optional task signals, position behavior, stability telemetry, training speed and memory at matched token exposure. State a result that would falsify the mechanism.
- List CPU/reference tests and native parity tests needed before training.
- State any complexity, implementation or throughput cost relevant to the fixed GPU budget.
- QAT probe: `skip` with a reason, or `run` with a specific quantization-sensitive mechanism, expected immediate conversion shock and short-continuation signal. A QAT probe requires matched completed dense parents and baseline QAT control; it cannot prove final ternary quality.

## Results (fill after all screen/full/QAT runs on this exact commit)

- Candidate commit and trial ID; actual exposure, GPU-hours, loss/task vector, stability events, failed predictions, and next decision. Preserve the original prediction above. Keep interim results in the external append-only ledger; do not change the candidate commit between its screen, full and QAT stages.
