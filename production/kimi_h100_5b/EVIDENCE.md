# External campaign packet and handback

Choose one external `QUAL_ROOT` before generation. This is a layout contract,
not an existing generated campaign. Do not put any of these runtime artifacts
or their data/checkpoint payloads in Git:

```text
QUAL_ROOT/
  campaign.json                  # scope, contract, chosen execution, cap
  forecast.json                  # input rates, separate costs and uncertainty
  budget.jsonl                   # append-only allocation/activity accounting
  identity/                      # source/runtime/compiler/GPU/tokenizer hashes
  manifests/                     # exact-horizon views and provenance
  configs/                       # H, R0, R1, R2, R3; later selected full winner
  evaluation/                    # frozen selection, safety, final-set identities
  gates/                         # offline/native/fit/recovery receipts
  probes/                        # isolated reference/fit branches, outside trials
  trials/H/                      # events, metrics, checkpoints, backup receipts
  trials/R0/ ... trials/R3/
  confirmation/                  # exactly one selected recipe by default
  reports/                       # comparison and final handback
  launch/                        # exact tested commands and human launch record
```

Record absolute artifact paths and a durable remote checkpoint destination
without exposing secrets. Capture source commit plus hashes of training and
qualification source, dirty-tree status, pinned dependencies/native libraries,
compilers, GPU SKU/HBM/driver/power settings and actual billing quantity. Pin
resolved config defaults, manifest/data/tokenizer fingerprints, held-out
identities, seed and initial weights hash. Hash references, gate outputs and
the selection packet so a winner cannot be silently substituted later.

## Required receipts before trials

| Gate | Receipt contents |
| --- | --- |
| Offline contract | Actual loaded config and effective optimizer groups; counts; matrix membership; batch/accumulation; exact WSD endpoints; external path checks; source hashes; CPU tests and failures |
| Data/evaluation | Verified shard/tokenizer identities, sample order, provenance, horizon/curriculum construction, no held-out leakage, audited valid-target counts/normalizer; evaluator identities and frozen margins |
| Runtime/native | Actual hardware and runtime; native kernel/complete-update controls and tolerances; cache correctness; missing-gradient/full-state checks; all unrun checks named |
| Execution fit/cost | Full-update samples after warmup, optimizer/eval/checkpoint memory peaks, chosen microbatch/accumulation, cadence/diagnostic overhead, separate proxy and full-size rates |
| Recovery/storage | Atomic completion/hashes, real upload/restore and subsequent-step comparison, safe retention under failed upload, provider network throughput, free-space and lag thresholds |
| Budget/launch | Allocation start/billing unit, consumed/remaining hours, measured forecast including confirmation and emergency stop time, exact bounded commands, explicit human launch record |

A `passed` status requires an actual result. Use `unrun`, `failed`, or
`blocked_by_missing_input` where appropriate; a checklist tick or CPU
arithmetic does not stand in for native evidence. A runtime identity receipt
does not establish kernel or learning correctness.

The launch packet must state how to stop gracefully, what happens on hard
failure/time-limit/interrupt, where the last valid checkpoint lives, and how
to resume without resetting the campaign clock. Candidate changes and new
source code invalidate the relevant receipts. Do not overwrite a prior
attempt or reuse its directory for a different contract.

## Human-readable handback

Keep the handback outside Git and include:

1. **Outcome:** engineering ready / native gates pending / partial campaign /
   completed comparison, with precise limitations; source commits and test results.
2. **Resources:** exact H100 and billing unit, allocated/active/idle hours,
   reserve remaining, cadence-inclusive measured rates and unmeasured assumptions.
3. **Coverage:** each mandatory gate, artifact path/hash, status, remaining
   work and the concrete reason any comparison did not finish.
4. **Candidate table:** ID, proxy Muon/AdamW, seed/initialization identity,
   valid tokens/updates, schedule endpoints, completion/failure, per-domain
   warmup/pre-decay/final NLL, stability alarms, wall hours and checkpoint identity.
5. **Selection:** margins, uncertainty/tie, reserved final-set result,
   provisional choice versus established comparison, and evidence for the choice.
6. **Transfer:** selected proxy rate, `768/2304` multiplier, resulting full-size
   AdamW, unchanged Muon/small-group settings; config diff and fresh model identity.
7. **Confirmation:** full-size fit/numerics, 2M batch, exact 1.0066B horizon,
   learning/stability/recovery results or pending status. No repeated LR sweep.
8. **Next AMD gates:** eight-rank global-batch parity, native FNUZ/attention/Muon,
   measured MI325X fit/speed, RCCL, provider-to-S3 durability, dense-to-QAT
   implementation/control, long-horizon schedule and funded token forecast.

Do not write "production ready" from this handoff alone. Keep the eventual
production schedule/config separate from both completed proxy cooldowns and
the full-size qualification cooldown. Do not copy qualification weights into
the production model or infer trillion-token stability from an early check.
