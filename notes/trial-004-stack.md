# Architecture trial: trial-004 — stack: Peri-LN output norms + learnable value residual

Written before training. Campaign `architecture-20261006c`. **Conditional candidate:** screened only if trial-001 and trial-002 each show independent value at screen; otherwise skipped and this note stands as the design record.

## Evidence and applicability

- Both parents have predeclared notes: `notes/trial-001-periln.md` (branch output norms, learnable channel gains) and `notes/trial-002-value-residual.md` (V′ₙ = λ₁V₁ + λ₂Vₙ).
- Interaction analysis (written before either result): the mechanisms touch different tensors. Peri-LN normalizes branch *outputs* (post-o_proj / post-down_proj); value residual changes the *value pathway inside attention*. No shared parameters; no conflicting scale semantics — the attention output norm sits after o_proj, downstream of the value mix. Expected coupling: (a) value residual strengthens layer-0's value pathway while Peri-LN keeps layer-0's attention output engaged — complementary for the layer-0 pathology family; (b) Peri-LN bounds what the mixed-value attention can inject into the residual — makes the value-residual mechanism safe against V₁-driven outlier injection.
- Risk: two small effects may not compose; each parent adds ~0 at screen quality (001: tie with stability win). The stack's screen must beat *both* parents on the scorecard axes that justified them (quality curve + residual control + entropy telemetry) to justify the full slot.

## Mathematical specification

Exactly the union: per layer, `x′ = x + OutNormA(Attn(InNorm(x)))` with attention computed on mixed values `V′ = λ₁V₁ + λ₂Vₙ` (λ free scalars init (0.5,0.5), layer ≥ 1), then `x″ = x′ + OutNormF(FFN(InNorm2(x′)))`. Parameters: +2×768 gains/layer + 2 scalars/layer ≥ 1 (+38,448 total). Cost: union of parents (≤2% throughput). FP8/QAT behavior: union of parents' analyses.

## Prediction and experiment

- Prediction at matched 0.5B: quality ≥ min(parent screens) and curve area ≤ best parent; residual RMS bounded as in trial-001; mid-layer min-entropy no worse than baseline. **Falsifier:** worse than either parent on any axis it inherits from that parent.
- Tests: the union of both parents' architecture test files plus a joint cached/uncached parity test.
- QAT probe: inherits trial-001's quantization-sensitive rationale (branch-scale bounding into FP8/ternary).

## Results

(To be filled if screened.)
