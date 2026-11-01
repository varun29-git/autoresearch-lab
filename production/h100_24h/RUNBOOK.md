# H100 dense LR qualification: fresh 24-hour allocation

Status: **staged, not launched**. The human starts and stops paid GPU work. The
old 50-hour, 5B-token campaign and its first 6.819572 charged hours remain
historical evidence in `production/kimi_h100_5b/` and its external packet.
This amendment counts **24 hours from the next H100 allocation**. It does not
reset or revise the old ledger.

## Scientific question and fixed controls

Test whether the selected dense learning-rate transfer is sound at the
intended **2,097,152 valid tokens per update**, using the 194,329,390-parameter
24-layer/width-768 proxy, then one fresh 1,446,478,126-parameter width-2304
confirmation. Keep sequence length 4096, model/optimizer semantics, data order,
validation identities, seed 1337, FP8 projection path, five-step spectral
Muon, clip 1.0, and all fixed LR groups identical across arms. H is the
historical-rate anchor; R0 retains the intended production 0.003 embedding
rate after the one-time width transfer.

| Arm | Proxy Muon LR | Proxy tied-table AdamW LR | Full-size transferred AdamW LR |
| --- | ---: | ---: | ---: |
| H | 0.017 | 0.006 | 0.002 |
| R0 | 0.017 | 0.009 | 0.003 |
| R1 | 0.017 | 0.00255 | 0.00085 |
| R2 | 0.0082 | 0.009 | 0.003 |
| R3 | 0.0082 | 0.00255 | 0.00085 |

Other rates: ordinary norms/value 0.0003; Q/K norms 0.0001. Muon momentum
0.95; Adam betas 0.9/0.95; matrix decay 0.02; embedding/norm decay zero.
The H100 proxy uses measured microbatch 8 and accumulation 64; full-size uses
measured-fit microbatch 2 and accumulation 256. These retain the same global
batch. No batch-size trial or QAT is included within 24 hours.

## Complete WSD horizons

| Run | Warmup | Stable | Cosine decay | Total | Boundary updates |
| --- | ---: | ---: | ---: | ---: | --- |
| Each proxy (updates) | 238 | 524 | 191 | 953 | 238, 762, 953 |
| Each proxy (tokens) | 499,122,176 | 1,098,907,648 | 400,556,032 | 1,998,585,856 | |
| Full-size (updates) | 29 | 200 | 57 | 286 | 29, 229, 286 |
| Full-size (tokens) | 60,817,408 | 419,430,400 | 119,537,664 | 599,785,472 | |

H reached update 725 in the old 5B WSD plateau. Its original schedule and
this schedule produce **identical LR on updates 1–725**; the new decay starts
after update 762. The parent checkpoint is immutable. `fork.py` requires its
payload hash, contract, complete update, 725 valid-target and full-state
records, unchanged model/optimizer/microbatch/runtime, identical consumed LR,
and identical training data through the **whole new 2B horizon**. The one
historical operational recovery incident at update 100 is disclosed in the
fork receipt and selection report; a different incident fails the fork.
If any condition fails, **do not credit H's 725 updates or launch the five
fresh arms on this 24-hour forecast**.

All arms have identical initial weight seed, data order and complete WSD
horizon. H is a prospective continuation; its full training history remains
traceable through the parent checkpoint. Comparison uses the same frozen
held-out records at warmup, pre-decay and finish. A 2B-token single-seed
comparison can rank these hypotheses provisionally; the 600M full-size run
checks early transfer and stability. Neither proves a trillion-token optimum,
the eight-MI325X batch, AMD FP8 performance, or QAT quality.

## Time model and decision rule

At the observed recent H wall rate **163,248 proxy tokens/s**, cautious
**35,000 full-size tokens/s**, and **3 hours** for setup, restore, diagnostics,
backup and emergency stopping:

| Work left after the verified H fork | Tokens | Forecast hours |
| --- | ---: | ---: |
| H updates 726–953 plus four fresh proxies | 8,472,494,080 | 14.4165 |
| One fresh full-size confirmation | 599,785,472 | 4.7602 |
| Setup/contingency reserve | | 3.0000 |
| **Total** | | **22.1767** |

Arithmetic headroom is **1.8233 hours**. Five fresh proxies without H credit
forecast **24.7639 hours**, beyond the cap. These are planning rates, not a
guarantee: H's wall rate includes ordinary cadence, while the full-size
35k rate remains to be qualified at the new cadence. The controller measures
each completed proxy from training event wall timestamps, applies a 5%
forecast haircut, and checks all remaining work against the single append-only
24-hour ledger before each arm. The runner monitors cadence-inclusive wall
time every 20 complete updates, reserves future work and 0.5 hours for a
checkpoint/stop, and stops at a complete update if the forecast fails. Actual
setup time consumes the 3-hour allowance. A slower run can exhaust the plan;
the controller then stops without declaring a winner.

## Offline staging before allocation

Run from a clean source checkout; keep generated artifacts outside Git:

```bash
python -m production.h100_24h.prepare24 \
  --parent-root /external/ci1-h100-qualification \
  --output-root /external/ci1-h100-qualification-24h
python -m production.h100_24h.plan
python -m unittest production.h100_24h.test_plan
```

The packet creates exact manifests, five proxy configs, a full-size fit config,
frozen margins, source receipt and **pending** native gates. It never copies
source shards, credentials, checkpoints or evaluation records into Git. The
H payload itself is in the old S3 backup, not in the local packet. The
operator must restore it with `recovery.restore_checkpoint()` into a fresh
directory on the VM, using the SHA-256 from the old step-725 receipt. That
function checks the downloaded payload and sidecar. Keep the old packet and
parent run metadata available on the VM; pass the restored `.pt` path to the
controller.

## Native sequence after the human launches the VM

1. Start the **new** ledger at VM allocation, using
   `python -m production.h100_24h.budget --ledger "$QUAL_ROOT/budget.jsonl" open`.
   Verify one billed H100 VM, GPU/HBM, disk free space and S3 access. The human
   remains responsible for deallocation when the campaign ends or stops.
2. Use this exact clean Git commit. Capture/verify `production.runtime`; run
   native `terminus.preflight` for `configs/H.json` and
   `configs/fit_full_R0.json`. Recheck effective FP8 operands, attention,
   packed Muon updates, full-state finiteness and at least 15% HBM headroom.
3. Restore the old H step-725 payload and receipt from the old S3 prefix.
   Confirm the old checkpoint SHA-256 against the new packet's
   `parent_H_checkpoint_sha256`. Leave the parent run/manifest unchanged.
4. Verify all shard/tokenizer bytes for **both** new manifests with
   `production.amd.staging`: use `configs/H.json` with
   `manifests/proxy_2b_m8.json`, and `configs/fit_full_R0.json` with
   `manifests/confirmation_600m_m2.json`. Write separate staging receipts.
5. Confirm the full-size m2 fit, optimizer-state peak and cadence-inclusive
   rate on the intended runtime. A short probe is operational evidence, not
   candidate progress. Verify checkpoint upload/restore and ensure the new
   backup destination is separate from the original campaign.
6. Reforecast from the actual observations. Launch the single controller
   process below only if the native gates pass. It forks H, runs H→R0→R1→R2→R3,
   selects or records a tie under frozen margins, transfers the rates once,
   and runs one fresh full-size confirmation. An incomplete/failed arm or
   budget alarm stops the sequence. The controller does not provision or
   deallocate a VM.

```bash
python -m production.h100_24h.auto \
  --qual-root "$QUAL_ROOT" --data-root "$DATA_ROOT" \
  --parent-run-dir "$PARENT_ROOT/trials/H" \
  --parent-manifest "$PARENT_ROOT/manifests/proxy_5b_m8.json" \
  --parent-checkpoint "$RESTORED_H/step_00000725.pt" \
  --runtime-report "$QUAL_ROOT/identity/runtime.json" \
  --staging-proxy "$QUAL_ROOT/identity/staging_proxy.json" \
  --staging-full "$QUAL_ROOT/identity/staging_full.json" \
  --preflight-proxy "$QUAL_ROOT/gates/preflight_proxy.json" \
  --preflight-full "$QUAL_ROOT/gates/preflight_full.json" \
  --backup-destination "$NEW_BACKUP_URI"
```

The external `reports/live_forecast.json`, `qualification.jsonl`, training
events, checkpoint upload markers, `fork_receipt.json`, selection and transfer
records are the handback evidence. Inspect loss, held-out NLL by domain,
valid-target count, nonfinite state, gradient clipping, logical Muon/AdamW
role updates, FP8 cache freshness, HBM, data waits, checkpoint/S3 lag and
the allocation ledger. A healthy short-run result is still provisional; the
AMD production readiness gates remain separate.
