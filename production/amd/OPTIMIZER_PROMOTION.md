# Selected dense optimizer

**2026-11-04 user decision:** the dense tied embedding/head AdamW peak is
**0.001**, replacing 0.002 for the intended long pretraining horizon. This is
an explicit user choice; it is not a new measured long-horizon optimum or an
automatic promotion from the short comparison. The completed matched H100
comparison and its learning/stability results remain external to Git.

AMD config generation and validation select `SELECTED_DENSE_OPTIMIZER` with
the already-transferred production rate **0.001**. Its width-768 proxy
equivalent is `0.003`; do not apply another width or GPU-count multiplier.
Spectral Muon remains **0.017**, ordinary norms **0.0003**, learned Q/K norms
**0.0001**, and the architecture, batch and other optimizer settings are preserved.
The existing WSD cooldown to 10% of peak gives embedding AdamW **0.0001**.

Generate fresh external AMD dense configs. The validator rejects older 0.002
and 0.003 dense configs; existing checkpoints are not silently reinterpreted
with a different peak LR. Keep historical campaigns and immutable packets intact.
QAT remains a separate policy with embedding AdamW **0.003**, requiring its own
transition qualification. Native MI325X acceptance and long-horizon learning
monitoring remain required before the month-long launch.

## Historical promotion, 2026-11-02

**2026-11-03:** a separate
[0.002 versus 0.001 learning/stability check](pretraining_readiness/EMBEDDING_LR_001_CHECK.md)
is now available for staging. This does not change the promoted rate below.

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
