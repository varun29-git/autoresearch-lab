# Dense embedding LR comparison: 0.002 versus 0.001

The user requested **0.001** as a possible safer large-scale dense embedding
rate. Stage an isolated comparison; the selected AMD dense AdamW rate remains
**0.002**. QAT rates and production resume contracts remain unchanged. The new
entry point is `python3 -m production.embedding_lr_check`; it does not alter the
completed five-arm H100 campaign or provision hardware.

## Transfer and fixed controls

| Geometry | Parameters | Reference embedding LR | Challenger embedding LR | Global tokens/update |
| --- | ---: | ---: | ---: | ---: |
| Width-768 proxy | 194,329,390 | 0.006 | 0.003 | 2,097,152 |
| Width-2304 full size | 1,446,478,126 | 0.002 | 0.001 | 2,097,152 |

Apply the existing AdamW width rule once: proxy `0.003 × 768 / 2304 = 0.001`.
Spectral Muon stays **0.017**; ordinary norm **0.0003**; learned Q/K norm
**0.0001**. Architecture, initialization seed, data order, decay, clipping,
optimizer momentum/betas and schedules match between arms. Only the tied
embedding/head AdamW peak changes. No square-root batch adjustment is needed:
the global batch stays fixed. Eight GPUs do not multiply the LR or the batch.

One H100 can implement the target batch through accumulation. The staged
proxy uses microbatch 8 and accumulation 64; full size uses microbatch 2 and
accumulation 256, both at sequence length 4096 and world size 1. Native memory
fit/preflight is required. An eight-MI325X numerical and full-update acceptance
check remains separate from this learning comparison.

## Bounded comparison

Run fresh matched proxy arms first. Their complete short WSD schedules each
use **953 updates / 1,998,585,856 tokens**, with 238 warmup updates and 191
cosine decay updates to 10% of peak. These schedules reproduce the completed
qualification contract, not a production trillion-token horizon.

If the challenger remains competitive, the optional fresh full-size pair
uses **286 updates / 599,785,472 tokens per arm**, with 29 warmup and 57 decay
updates. Proxy pair total: **3,997,171,712 tokens**. Both pairs combined:
**5,196,742,656 tokens**. Forecast actual hours from measured complete-update
wall times including diagnostics, validation, storage and startup; no new
throughput estimate or paid allocation is implied by staging.

This is a **cold-start** comparison. An embedding-only change from an existing
production checkpoint needs a separately reviewed optimizer/schedule transition;
the current resume contract deliberately rejects altered peak rates.
Historical H results provide context but cannot substitute for a fresh matched
reference after source, runtime or data changes.

## Selection and monitoring

Compare learning at equal consumed tokens during the plateau and after decay,
using the same frozen validation identities and per-domain breakdown. Keep the
existing qualification margins: challenger minus reference NLL at most **0.002
aggregate** and **0.01 per domain** for noninferiority; **0.005** improvement for
clear aggregate superiority. These are operational margins, not guarantees of
trillion-token quality. Resolve uncertainty with paired held-out analysis and,
if necessary, a longer comparison or another seed. Do not promote on a tiny
single-seed advantage.

Review finite weights/gradients/optimizer state, required gradients, actual
group LRs, pre-clip gradient norms, clipping coefficients, embedding update to
weight RMS, sampled attention logits/entropy, Q/K statistics and activation
magnitudes. The runner reuses complete-state finite checks each update, logical
role deltas every 20 updates and at boundaries, and existing sampled stability
probes. Sampled probes do not measure every token/head; frequency-bucketed
embedding-row diagnostics are not implemented by this packet.

The existing severe-clipping alarm triggers after warmup if more than 20 of
the latest 100 updates have a coefficient below 0.1. Inspect the warmup boundary
and full distributions too; clipping at initialization alone does not prove
that a candidate LR is excessive. No universal attention-logit cutoff is
introduced. Halving the LR mechanically shrinks updates; that is not sufficient
evidence of improved learning or stability. Prefer 0.001 if its learning is
noninferior and the complete trajectory is acceptable. Retain 0.002 if reducing
it measurably harms learning without resolving a demonstrated stability issue.

## Execution status and gates

The offline packet generator checks source/file identity, parameter counts,
token horizon, exact global batch and that only embedding LR differs. It leaves
production validation locked. Generated configs, manifests, receipts and results
must stay outside Git. The native runner requires one human-launched H100,
matching runtime and preflight reports, actual data, an open billed-allocation
ledger, measured current/remaining work forecasts and explicit S3 queue/disk
limits. Checkpoints use separate packet/geometry/arm remote prefixes.

The runner checks budget and storage health and verifies the last uploaded
checkpoint on exit. It does **not** start or deallocate a VM or automatically
advance arms. Before a human launch, bind it to verified provider stop/state
hooks and the independent watchdog; process exit alone does not stop billing.
Native execution, memory fit, S3/provider integration, learning comparison and
automatic promotion have not been performed by offline staging.
