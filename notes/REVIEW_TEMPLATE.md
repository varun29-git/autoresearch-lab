# Trial diagnostic review: <ID and stage>

Write one copy for every screen, full continuation, QAT probe or failed attempt. Keep the review with the trial notes and append later stages without rewriting earlier judgments. Compare against the baseline at **the same token exposure**. Record exact file paths, token counts, layer/tensor names and values for each material claim.

## Outcome and evidence coverage

- Candidate and baseline commits, stage, tokens, parameters, retries, GPU-hours, checkpoint/evaluation identities.
- Which logs were read: `events.jsonl`, `stability.jsonl`, `incidents.jsonl`, protected evaluation JSON, and `research.loop status`? Note absent, truncated, candidate-only or baseline-only probes. A missing sample is not a healthy value.
- Did native preflight, causality/reference tests and checkpoint integrity pass? If not, separate implementation failure from an architectural result.

## Quality and learning

- Held-out English loss and uncertainty at matched exposure; every point on the validation curve and any late regression. Compare early/middle/late training loss and the loss-curve area. Note train/validation divergence, distribution changes or too few validation points.
- Loss in each measured position bucket. If optional development tasks exist, inspect every task aggregate and the spread of per-item gains and regressions; teacher-forced answer loss is not generated-answer accuracy.
- What improved, what worsened, and does the timing match the proposed mechanism? A lower loss at one noisy checkpoint is not enough on its own.

## Numerical and functional behavior

- List **all incidents and nonfinite probes**, their first occurrence and what happened afterward. Include failed trials.
- Compare gradient norm and clipping by early/middle/late phase. For every affected parameter group or layer, inspect gradient RMS/max, post-clip values and update-to-weight ratio. Look for both stalled updates and runaway updates; include norm and Q/K scale parameters.
- Inspect per-layer attention and FFN output RMS against residual RMS, especially layer 0 and any altered branch. Identify suppressed branches, explosive growth and unexpected cross-layer discontinuities. Report stable changes too when they support the proposed mechanism.
- Inspect Q/K RMS, attention-logit RMS/max, attention probability/entropy and local/global differences. Determine whether sharper or flatter attention has a plausible functional benefit or is a pathology; do not assign a universal safe threshold.
- Inspect FP8 operand relative error, scale range, zero fraction and nonfinite fraction by affected projection/tensor and training phase. Relate changes to loss and optimization, including dense-to-QAT conversion if run.
- Inspect first, last and extreme values of every changed named probe in the status dossier. Explain any material candidate-only or vanished probe. Raw directions are evidence, not automatic labels.

## Cost and transfer

- Compare training tokens/s by phase, peak memory, total GPU-hours including failed attempts, parameter count, measured step cost and architectural complexity. Account for validation/preflight overhead separately when interpreting training speed.
- Revisit proposal estimates for prefill/decode FLOPs and KV-cache bytes per token. Note that actual inference latency is not measured by this runner.
- For a matched QAT probe: immediate conversion loss, recovery curve, new incidents and per-layer/parameter/FP8 changes. State whether 100M tokens are enough to support only a diagnosis or a real ranking.

## Judgment before the next experiment

| Finding | Positive, negative or uncertain? | Exact evidence and location | Mechanism or alternative explanation | Follow-up |
| --- | --- | --- | --- | --- |
| <one material finding per row> | <classification> | <file, tokens, tensor/layer, candidate versus baseline> | <reason> | <continue, reject, test, or monitor> |

State the strongest favorable and adverse evidence, what is still unmeasured, and the decision: reject, continue to full, run matched QAT, retain as a Pareto alternative, or revise the hypothesis. Do not promote a candidate solely because one metric improved; do not reject one solely because a diagnostic moved without showing why that movement is harmful.
