# Complete Muon Hyperball candidate on stack 004

This opt-in arm is based on the optimized single-GPU stack and the spectral Muon branch. It implements [Hyperball Optimization](https://arxiv.org/abs/2606.16899) for every logical hidden attention and MLP matrix: Q, K, V, O, gate, up and down. Tied embeddings, normalization gains and scalar parameters retain the existing auxiliary Adam path. It is a dense optimizer candidate, not a measured Terminus quality or throughput win. No paid GPU experiment was launched.

Set `muon_lr_rule` to `"hyperball"` in a **fresh run**. `muon_lr` is then Hyperball's angular LR and must be tuned independently; the spectral Muon numeric LR is not transferable. New `OptimizerConfig` instances default to `"spectral_unclamped"`. The existing `weight_decay` setting still applies to eligible auxiliary Adam parameters, while hidden Hyperball matrices use zero weight decay.

For each logical matrix, initialize `R = ||W_initial||_F`, compute the pinned five-step NS/Nesterov Muon direction `Q`, then apply

```text
W_trial = W - muon_lr(t) * R * Q / ||Q||_F
W_next = R * W_trial / ||W_trial||_F
```

The optimizer stores one radius per logical slice, including separate radii inside packed QKV and gate/up parameters. The batched path uses GPU foreach norms and multiplication, with no `.item()` synchronization. Radius values are in the optimizer checkpoint. Switching an existing spectral or legacy checkpoint to Hyperball is rejected; compare new runs from initialization on matched data and token budgets.

Since `normalize_F(sQ) = normalize_F(Q)` for positive `s`, complete Hyperball cancels the scalar spectral width factor. This is a distinct optimizer arm, not an add-on to the spectral formula. The user-visible model architecture is unchanged. Its throughput overhead and quality on one H100 NVL remain unmeasured.

The selected stack's archived 200M training telemetry showed large sampled O/down weight growth under the **old** Moonshot-style optimizer: median 20.5× and 34.9× respectively by 2.5B tokens. That observation motivates a comparison; it does not prove the old run was harmed or predict Hyperball's effect with the new spectral baseline.

Next build matched **stack-004** 200M configs with `branch_norm="peri"` and `value_residual="learned"`, then compare the two arms with their own LR brackets on the same dense data and completed schedule. The checked-in generic `campaign_200m_dense_fp8_wsd.json` omits those stack fields and uses the archived Moonshot LR rule; it is not a valid comparator as-is. A short prefix can detect numerical failures, but published Hyperball experiments sometimes start worse and overtake during decay. Choose an optimizer before the broad hyperparameter search. Native H100 arithmetic, throughput and matched-token loss are pending.
