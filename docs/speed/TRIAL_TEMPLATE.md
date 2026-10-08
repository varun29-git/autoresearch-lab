# Speed trial proposal and review

Copy the design portion into a short committed proposal. Keep generated measurements, traces, loss records and receipts outside Git. Commit the review's interpretation and external artifact identifiers without embedding private evaluation records.

## Before coding

- Trial ID, reference/incumbent commit, campaign lock, phase, class E/N and atlas idea (or original question).
- Measured problem: exact kernel/region, shapes, exposed milliseconds, HBM bytes, live allocations and uncertainty. Distinguish observed facts from assumptions.
- Mechanism: equations, rounding/scale boundaries, dependency changes, saved/recomputed tensors, gradients including gains/lambdas, preserved optimizer/input behavior.
- Physics: operation/byte/launch counts before and after, hardware service-time and Amdahl ceilings, register/shared-memory/workspace needs, effect on attainable microbatch.
- Whole-update prediction: low/central/high time saving and memory change; what the trace should show; explicit falsifying outcome.
- Evidence/search: query/date, primary URLs/sections, exact implementation revisions/licenses, strongest competing method/negative evidence. Originality is a hypothesis unless established; do not invent a novelty claim.
- Plan: cheap diagnostic, parity cases, component test, paired full-update count/order/state, numerical limits, quality endpoint/margin, confirmation/soak if justified. No thresholds chosen after results.
- Spend: cap, estimated GPU/VM-on time including cold setup and archive, reserve needed for a valid comparison, stop condition. Production-equivalent local batch and quality batch mapping.
- Risk: stale cache/graph/stream behavior, partial parameter updates, missing grad path, data/target mismatch, head/tied-weight or QAT sensitivity, fallback path.

## Before paying for the trial

- Committed code/proposal, clean checkout, protected checker/reference verified, no unintended architecture/recipe diff.
- CPU/reference tests passed with exact commands; native tests required and unrun clearly listed.
- External supervisor has the human's active campaign authorization, remaining reserved cap, timeout, artifact destination and recovery procedure.

## After the trial

- Exact commits, complete GPU/VM-on charge, external receipt and raw manifest identities; failures and retries included.
- Numerical/optimizer/input checks, separately from speed. Every missing/failed check, not just the aggregate outcome.
- Paired-rate distribution, confidence bound, baseline drift, peak allocated/reserved/device memory, cold amortization and profiler explanation. Did removed work equal the prediction?
- Quality/stability outcomes by phase/domain/checkpoint and statistical uncertainty; unavailable checks listed explicitly.
- Positive and negative signals; plausible confounds; which competing explanation a next cheap test would distinguish.
- Decision: reject, unresolved, remeasure, screen pass, or independently qualified finalist. No production-ready claim from one GPU.
- Updated bottleneck map, remaining 1.5T/2T capacity under fixed QAT and explicit scaling/overhead scenarios, next experiment and why it is worth its cost.
- Composition dependencies, incumbent/fallback reference, rollback/resume concerns and research budget remaining.
