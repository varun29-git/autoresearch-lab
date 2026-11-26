# Campaign contract and execution order

This is a qualification campaign, not the production run. Freeze the external
campaign packet and its source commit before launching trials. Changes to the
scientific controls require an explicit amendment and comparability review.

## 1. Fixed model, optimizer and batch

Derive the proxy from [production/recipe.py](../recipe.py), changing only
`dim=768`, `ffn_dim=1920`, `n_heads=6`, `n_kv_heads=2`. Verify the count
**194,329,390** using the real model on the meta device. Full-size confirmation
uses the reviewed width 2304 / FFN 5760 / heads 18/6 and count **1,446,478,126**.
Both retain 24 layers, context 4096, head dimension 128, 512-token local
attention and global layers 6/12/18/24. Preserve all other model defaults,
including initialization, tied head, gains and value residuals.

Dense DeepGEMM FP8 projections and FA3 remain selected for H100. Residuals/head
are BF16; latent weights, gradients and optimizer state remain FP32. FP8
ternary/QAT is not part of this campaign. The selected model's
`gradient_checkpointing` default is **false**. Initially measure fit without
checkpointing. Prefer reducing microbatch and increasing accumulation at
fixed global batch if needed. If checkpointing is necessary, qualify its
parity and freeze that execution choice across proxy comparisons; record the
reviewed execution delta instead of silently changing it halfway through.

Run in the deterministic order **H, R0, R1, R2, R3**. H establishes the
historical-rate anchor under the new contract; the complete crossed matrix is
still required for the planned comparison. Order does not authorize pruning
a healthy arm before its 5B horizon.

| ID | Proxy Muon eta | Proxy tied-table AdamW | Transferred full-size AdamW |
| --- | ---: | ---: | ---: |
| H | 0.017 | 0.006 | 0.002 |
| R0 | 0.017 | 0.009 | 0.003 |
| R1 | 0.017 | 0.00255 | 0.00085 |
| R2 | 0.0082 | 0.009 | 0.003 |
| R3 | 0.0082 | 0.00255 | 0.00085 |

Transfer is `adamw_lr_full = adamw_lr_proxy * 768 / 2304`, exactly once.
Muon eta, norm and Q/K rates stay unchanged. Do not add square-root batch
scaling or a factor of eight. Preserve `spectral_unclamped` and the pinned
five-step Newton–Schulz/Nesterov implementation on each logical matrix,
including separate Q/K/V and gate/up slices of packed tensors.

Fixed in all five arms: norm/Peri/final/value LR **0.0003**; Q/K norm LR
**0.0001**; matrix decay **0.02**; embedding/norm/QK/value decay **0**;
momentum **0.95**; Adam betas **(0.9, 0.95)**; epsilon **1e-8**; clip **1.0**;
seed **1337**; no role-specific LR overrides, compilation, CUDA graphs or
FP8 head. Assert effective optimizer groups, not just input JSON: defaults
and packed logical shapes are part of the contract.

| Hardware/execution example | Microbatch sequences/rank | Accumulation | Global tokens/update |
| --- | ---: | ---: | ---: |
| One H100, initial proxy fit candidate | 4 | 128 | 2,097,152 |
| One H100, larger measured proxy microbatch | 8 / 16 | 64 / 32 | 2,097,152 |
| One H100, smaller full-size fit candidate | 1 / 2 | 512 / 256 | 2,097,152 |
| Eight MI325X, later measured fit candidates | 8 / 16 / 32 | 8 / 4 / 2 | 2,097,152 |

These are arithmetic examples, not measured fit or speed claims. Measure
complete updates with optimizer state and evaluation/checkpoint peaks.
Freeze proxy microbatch/accumulation after fitting; choose full-size execution
separately without changing its global batch. No need to fill all HBM for its
own sake. Require at least 15% usable driver HBM free at measured peaks.

**Token accounting is a launch gate.** The current trainer computes
`update_tokens` from manifest sequence slots and divides summed loss by that
number. Loss kernels ignore targets marked `-100`. Audit the actual frozen
data path: every claimed valid token must contribute exactly once to loss,
including packed boundaries, padding and partial tails. If ignored targets
make valid counts differ from the nominal batch, reject the launch packet;
fix/test normalization and explicitly reconcile schedule/throughput units
before proceeding. Never relabel nominal slots as measured valid targets.

## 2. Complete token-aligned schedules

The proxy manifest's usable training horizon must be **4,999,610,368 tokens**.
Do not use a larger manifest and merely stop at update 2384: the trainer
derives its WSD endpoint from the manifest, so that would move the cooldown.

| Stage | Warmup | Stable | Cosine decay | Total |
| --- | ---: | ---: | ---: | ---: |
| Each 5B proxy: updates | 238 | 1669 | 477 | 2384 |
| Each 5B proxy: tokens | 499,122,176 | 3,500,146,688 | 1,000,341,504 | 4,999,610,368 |
| Full-size confirmation: updates | 48 | 336 | 96 | 480 |
| Full-size confirmation: tokens | 100,663,296 | 704,643,072 | 201,326,592 | 1,006,632,960 |

Use `schedule=wsd`, `decay_style=cosine`, `final_lr_fraction=0.1`, fresh
`phase_start_tokens=None`, and no QAT warmup. The trainer evaluates the LR
at the upcoming update's ending token count; test this convention explicitly.
Proxy full validation/checkpoints are required at updates **238, 1907, 2384**;
full-size at **48, 384, 480**. Pre-decay proxy tokens are **3,999,268,864**.
Log a fixed initial evaluation as a sanity baseline as well.

Quick fixed held-out checks every **48 updates** (~100.66M tokens) and logical
role diagnostics every **20 updates** are starting cadences. Boundary checks
are additional; a simple interval does not hit update 1907. Checkpoint every
**100 updates** plus these boundaries as an initial operational choice; price
its actual cost and freeze the cadence/retention before the matrix. A dedicated
boundary trigger is preferred; segmented invocations must retain the complete
manifest, unchanged schedule and exact optimizer/RNG/data-cursor resume.

Optional 1M-batch proxy, **deferred by default**: same winning recipe, same
5B token horizon and absolute token boundaries, 4768 updates with segments
476/3338/954. No silent LR scaling. It is a proxy batch comparison, not proof
of the full model's critical batch. Further seeds, matched-decay controls,
norm controls and longer confirmations are separately priced follow-ups.

## 3. Offline work Kimi must finish before native trials

| Work item | Required implementation and acceptance |
| --- | --- |
| Candidate builder/validator | Separate qualification path; allow exactly the reviewed proxy geometry/rates and selected full-size winner. Reject unknown trial IDs, changed fixed settings, wrong world size, batch/horizon mismatch, stale runtime/data receipts, in-repo or reused output paths. Preserve production locks. Fully resolve dataclass defaults in external configs. |
| Data horizon views | Generate external manifests without rewriting shards. Retain tokenizer, shard hashes, source offsets, mixture and provenance. Use identical ordered data within the proxy matrix. If a phase curriculum exists, truncate its budgets consistently or reject unsupported input; never flatten it silently. Ensure enough disjoint frozen held-out data. |
| Bounded runner | Reuse `terminus.train` update semantics. Enforce planned absolute update/token endpoints, explicit boundary evaluation, per-update logs, a campaign ledger and stage limits. Pin trainer AND qualification-source hashes. Fail closed on missing native gates. Do not count probes as candidate progress. |
| Monitoring and references | Implement MONITORING.md gaps: all group LRs, required gradients, finite full state, logical-role deltas, actual native FP8 operands/cache checks and isolated same-state update references. Prove observation leaves RNG/state/data order unchanged. |
| Durable recovery | Atomic checkpoint completion, payload/receipt hashes, bounded asynchronous upload, durable completion marker, free-space/upload-age watchdog and verified restore. Resume only the same config/runtime/world-size/data contract; no moments/cursor reset. Test corrupt and interrupted uploads and retention safety. |
| Budget controls | One ledger covering allocation intervals and all subprocesses/retries. Forecast remaining work before each trial. Bound probes, guard shutdown/checkpoint time, and expose stop/alarm status to the human. An individual trainer time limit is insufficient. |
| Selection report | Fixed validation identities/margins, complete-trial coverage, per-domain and aggregate pre-/post-decay NLL, learning/stability/cost comparisons, explicit ties/incomplete outcomes and a traceable single width transfer. |

Use meaningful CPU tests for rejected contracts, batch and loss normalization,
schedule boundaries, packed optimizer grouping, state-preserving telemetry,
exact restart, failed backup/retention and clock/ledger expiry. Then keep
native tests separately marked unrun until measured. Avoid modifying the
training algorithm to accommodate a test harness.

Existing building blocks and their limits:

| Entry point | Appropriate use |
| --- | --- |
| `production.build_dense_config` | Baseline builder to study; does not generate this candidate matrix |
| `production.inspect_run` | Baseline CPU validation; locks model and optimizer, so not a candidate validator |
| `production.runtime` | Capture/verify real H100 runtime, pinned dependencies and compiler identity |
| `terminus.preflight` | CPU parameter count and native reduced-model/kernel checks; not a complete learning/optimizer soak |
| `terminus.train` | Existing training, cadence, preflight and resume contract to reuse |
| `scripts/train.sh` | Lower-level invocation, lacking the new campaign-specific guards |
| `production/train.sh` | Human production wrapper; rejects proxy geometry |
| `production.amd.train` | AMD-specific guarded runner to study, not to execute on H100 |
| `terminus.stability` | Sampled telemetry to extend; not full coverage or an all-rank watchdog |

After Kimi has generated actual candidate configs outside Git, this existing
CPU command checks their parameter count without allocating weights:

```bash
python -m terminus.preflight --config "$QUAL_ROOT/configs/H.json" --count-only
```

`QUAL_ROOT` is the external campaign root chosen by the operator, not a
directory in this repo. Do not run this example before that config exists.
The planning helper in README is runnable immediately. **No final sweep
command exists in this folder yet:** Kimi must supply the actual tested
qualification entry point, flags and gate receipts in the launch packet.

## 4. Native sequence on the human-allocated H100

1. Confirm exactly one visible H100, SKU/HBM, driver/runtime, power/clock state,
   storage and network paths. Record the actual billed VM/GPU quantity; exposing
   one device in a billed eight-GPU VM is not one GPU-hour per clock hour.
   Reuse a verified pinned runtime when available. Price any build/install time.
2. Capture and verify `production.runtime` identity. Run native preflight on
   the actual arithmetic and representative logical shapes. Test loss/masking,
   gradients and complete optimizer deltas against independent references.
   Verify FP8 cache refresh after updates and correct packed Muon scaling.
3. Fit the proxy and full model with bounded full-update probes. Measure
   warm complete-update speed, cadence overhead, data stalls and peak HBM
   through first optimizer state creation, validation and checkpointing.
   Small warmup probes are not full-size LR searches. Freeze execution choices.
4. Complete a real checkpoint/upload/restore drill, comparing subsequent state,
   tokens, data cursor and losses against uninterrupted execution within
   independently established native tolerances. Keep the drill outside trials.
5. Use those measured rates/costs to reforecast all five proxies plus the short
   confirmation and remaining checks. If it cannot fit, give the human the
   concrete remaining options; do not launch a knowingly unaffordable matrix.
6. Run H, R0–R3 from identical initial proxy state/data. Use unique output
   directories. Apply hard stops, boundary probes and evaluations. A failed
   arm stays failed; repairs define a new identified run. Preserve evidence.
7. Select a credible complete recipe under the declared rules, or report a
   tie/insufficient evidence. For an unresolved tie, identify a provisional
   choice for confirmation explicitly; do not claim a proven ranking.
8. Apply transfer once, generate/validate the full-size winner packet, and
   confirm that its measured forecast still fits. Start from fresh full-size
   initialization, with the intended batch and its own 1.0066B-token WSD.
9. Produce EVIDENCE.md's handback, including remaining AMD and QAT gates.
   Notify the human when compute is idle or the campaign needs stopping;
   process exit does not necessarily stop billing. Do not delete the VM or
   checkpoints as a side effect of this handoff.

## 5. Cap, forecasts and limits of the result

At proxy 200k / full-size 40k tokens/s: five proxies **34.7195 h**, short
confirmation **6.9905 h**, setup/check/contingency **8 h**, total **49.7100 h**.
The same assumptions with the sixth proxy cost **56.6539 h**. Five proxies
with a 2.0133B full-size confirmation cost **56.7005 h** including reserve.
Neither extension is included in the 50-hour plan. With 200k proxy and an
eight-hour reserve, the short full-size confirmation needs about **38.4k
tokens/s** or better for the default plan to fit.

Forecast using cadence-inclusive rates or add separately measured overhead,
without double-counting. The ledger charges allocated time regardless of
whether GPUs are busy. Maintain remaining emergency checkpoint/stop time;
do not wait for hour 50 to notice that a checkpoint takes time. Actual extra
setup/probe cost reduces available training time. Never reset elapsed time
on resume. The eight-hour reserve is an estimate, not permission to skip
uncompleted native gates if it is exhausted.

This campaign can select among tested proxy recipes and check early full-size
transfer. It cannot certify a trillion-token plateau, all batch optima, AMD
throughput or dense-to-ternary quality. Keep the eventual production warmup,
decay, dense/QAT split, eight-rank fit/collectives, FNUZ numerics, S3 provider
network and long-run recovery/quality gates in
[pretraining readiness](../amd/pretraining_readiness/README.md).
