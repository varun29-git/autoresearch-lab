# Dense optimizer promotion, 2026-11-02

The completed H100 proxy selection and full-size confirmation used embedding
AdamW `0.006 * 768 / 2304 = 0.002`. The user authorized integrating this
tested recipe after the readiness audit. AMD config generation and validation
now select `SELECTED_DENSE_OPTIMIZER`, whose only difference from the historical
`production.recipe.SELECTED_OPTIMIZER` is embedding AdamW **0.003 → 0.002**.

Spectral Muon remains **0.017**, using the existing logical-matrix aspect rule;
norm rates, weight decay, momentum/betas, architecture, initialization and
global-batch semantics remain unchanged. Width transfer is already applied;
neither the builder nor optimizer applies another width or GPU-count factor.
The historical H100 recipe is retained for reproducibility.

Regenerate external AMD dense configs. Old 0.003 dense configs are now rejected
by the AMD validator; they are not silently changed on resume. Native AMD
learning and numerical acceptance are still required. The separate historical
QAT starting candidate remains unchanged and must pass transition qualification.
This promotion does not transfer a dense learning result to ternary training.

Campaign results, source/transfer receipts, run configs and checkpoints remain
outside Git. See the current pretraining readiness entry for the other gates.
