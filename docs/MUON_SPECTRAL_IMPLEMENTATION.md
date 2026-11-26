# Spectral Muon on the optimized stack 004

Implemented on branch `hpo/muon-spectral`, based on the completed speed branch at `8a62293`. This changes hidden-matrix optimizer mathematics; it is a new dense tuning recipe, not a same-recipe throughput result. No paid GPU run was launched.

## Exact update

For each logical matrix `W` with shape `(d_out, d_in)`:

```text
M = beta M + (1 - beta) grad
D = (1 - beta) grad + beta M
Q = pinned_NS5(D)
W = (1 - eta * weight_decay) W - eta * sqrt(d_out / d_in) * Q
```

`muon_lr` is `eta`. The spectral path has no `0.2`, no width-growing factor, and no clamping of the aspect ratio to one. This is the `spectral_unclamped` / `unit_rms_norm` convention. Momentum, Nesterov semantics and the pinned five-step Newton–Schulz polynomial remain the same. Decay uses the base LR independently of the shape factor.

The optimized batched path still groups same-shaped logical matrices for NS5 and uses foreach updates. It folds the shape coefficient into the FP32 parameter-add scalar, avoiding an extra BF16 scaling/rounding operation. The reference path implements the same equation. CPU agreement does not establish native H100 agreement or throughput.

| Logical matrix in our width family | `d_out/d_in` | Coefficient on raw `Q`, divided by eta |
| --- | ---: | ---: |
| Q, O | 1 | 1 |
| K, V | 1/3 | 0.577350 |
| Gate, up | 2.5 | 1.581139 |
| Down | 1/2.5 | 0.632456 |

These coefficients are identical at widths 768, 1536 and 2304. Q/K/V and gate/up use their individual matrix dimensions, never the row count of the packed parameter.

## Selecting the new recipe

New `OptimizerConfig` instances and `registry.make_run_config` calls default to `muon_lr_rule="spectral_unclamped"`. In a fresh dense run's optimizer configuration, set this explicitly:

```json
{
  "kind": "muon",
  "muon_lr_rule": "spectral_unclamped",
  "muon_lr": 0.003
}
```

The displayed LR is the existing numeric default, **not a tuned recommendation in this new convention**. Removing the Moonshot factor changes the meaning of that number, including at the small width. Tune eta from initialization; the previous base-width matching suggestion is not applied as a hidden role multiplier.

The implementation leaves embedding/readout parameterization and auxiliary Adam updates unchanged. This implements the selected hidden-Muon rule; it does not by itself establish complete µP or long-horizon transfer for the model. Radius control and Hyperball are not part of this change.

Explicit legacy `sqrt_fan_in` and `sqrt_fan_in_ratio` configurations remain supported for control/reproduction. Existing checked-in Muon campaign configurations explicitly retain their old rule. New configurations that omit the rule use spectral scaling. Keep completed speed/architecture campaigns on their recorded source/configuration rather than substituting this changed optimizer into their locked comparison.

Optimizer state records the update-scaling convention. A checkpoint with no tag is interpreted as the original aspect rule; loading it into spectral Muon raises a scaling-mismatch error rather than silently combining old rates with a new update. Legacy checkpoints still load with the explicit legacy configuration.

## Verification

The local tests cover the raw-NS update equation on GQA Q/K/V and expanding/contracting FFN matrices, width-independent coefficients on square/tall/wide matrices, batched/reference agreement, scheduled and zero LR with independent decay, optimizer-state resume, exact archived-update compatibility, and unchanged auxiliary Adam behavior. Native H100/FP8 training quality and performance remain unrun.
