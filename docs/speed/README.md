# Stack-004 speed research framework

Entry point for Kimi K3: [speed_program.md](../../speed_program.md). This is a fresh speed campaign built from the winning architecture and its source, without adopting the later HPO/kernel planning work. The existing model, trainer and architecture campaign are preserved.

The user confirmed on 2026-10-08: **one research GPU, 32 production GPUs, 6,000 production H100-hours including QAT; experiments budgeted separately**. Exact production H100 SKU/interconnect, research cap, production global batch, QAT allocation and production checkpoints are not yet supplied. Those are explicit launch-time inputs, not guessed experimental controls.

## Contents and current readiness

| Artifact | Purpose |
| --- | --- |
| [Physics](PHYSICS.md) | Token/hour arithmetic, shape-specific mixed-precision compute bound, memory/network bounds and feasibility scenarios |
| [Experiment atlas](EXPERIMENTS.md) | Implementation hypotheses, derivations, falsifiers, priorities and interactions |
| [Measurement](MEASUREMENT.md) | Native instrumentation boundary, numerical/quality gates, statistical decisions, external receipt schemas and production acceptance |
| [Sources](SOURCES.md) | Verified primary material, pinned implementation revisions, applicability and limitations |
| [Proposal/review template](TRIAL_TEMPLATE.md) | Concrete instructions before and after an experiment |
| [Offline tools](../../speed_research) | Budget calculator, winner-source verifier, campaign lock, paired screen analysis, append-only receipts and reusable block timer |

**Implemented and locally testable:** offline math/provenance/locking/analysis/receipt tools and block timing API. **Requires the native research campaign:** integrating the timer into the pinned trainer, validating that integration, hardware profiling, kernel candidates, quality experiments and speed measurements. **Requires production hardware later:** NCCL validation and 32-GPU throughput. No GPU campaign has been run in constructing this framework. The CLI intentionally has no cloud/training launcher. It is not a replacement for a billing supervisor or an archive verifier.

## Baseline facts, not assumptions

The [architecture report](../../notes/FINAL_REPORT_ARCHITECTURE_20261006.md) identifies `8bd552314470586f589d94bdaa6d7e8bc3c0c101` as the winner. At 200M, its recorded 2.5B-token English loss was 2.65363 versus 2.68445, with 143.5K versus 152.4K tokens/s on an H100 NVL. The stack retained the advantage after the matched 500M-token QAT probe. These are **archived proxy results**, not fresh measurements or 1.5B speed estimates. The report's prediction that output-norm fusion will reclaim most of the overhead is a hypothesis to test, not established causation. In particular, its 11 GiB additional peak does not prove the norms alone own that memory.

The winning branch preserves 25 layers and adds Peri-LN branch-output RMSNorms and learned layer-0 value residuals. The complete [target model](../../speed_research/target_model.json) makes a natural width transfer explicit: width 2304, FFN 5760, 18Q/6KV heads of dimension 128, the same 25 layers, 20 local/5 global layers, context 4096 and local window 1024. It has **1,500,456,496 trainable parameters**. This target fixture is derived from the winner (3× width and heads, same head dimension and ratios); it is not a claim of a completed production architecture/quality validation. If the owner supplies a different already-frozen production geometry, version the fixture and recompute all bounds before launch; do not search geometry to improve speed.

All model defaults are explicitly materialized from the winning dataclass. No gain, initialization scale, padding ID or residual mode is silently inherited from the current planning branch. `anchor.json` contains SHA-256 hashes of the winning candidate and all tracked `terminus/` files. The reference has dense FP8 DeepGEMM, FA3 BF16 attention, learned QK RMSNorm, fused projections/SwiGLU, cached FP8 weights and streaming Triton cross-entropy already enabled.

## Local commands (no GPU work)

Run from the repository root; the tools use only Python's standard library:

```bash
python3 -m speed_research --help
python3 -m speed_research verify-anchor
python3 -m speed_research check-config --config speed_research/target_model.json --phase dense
python3 -m speed_research plan
python3 -m speed_research plan --hardware h100_nvl
python3 -m unittest tests.test_speed_research -v
```

Explicit scenario, **not a measured result**:

```bash
python3 -m speed_research plan --dense-tps 130000 --qat-tps 110000 \
  --qat-fraction 0.4 --scaling-efficiency 0.9 --unavailable-fraction 0.1
```

Do not call `verify-anchor --checkout` on the planning branch expecting success: that check is supposed to reject its non-winner model. It checks the research worktree before candidate changes.

## Create the future research checkout

The owner starts the campaign separately. On the inexpensive controller, use a unique tag and the **final framework commit** (obtain it with `git rev-parse HEAD` after reviewing this milestone):

```bash
git worktree add -b speed/<unique-tag> ../terminus-speed-<unique-tag> \
  8bd552314470586f589d94bdaa6d7e8bc3c0c101
```

In that new worktree, restore **only** `speed_program.md`, `speed_research/`, `docs/speed/` and `tests/test_speed_research.py` from the reviewed framework commit using `git restore --source <framework-commit> -- <those-paths>`. Commit the additive framework. Do not merge the current planning branch: it includes post-winner work. Do not copy the current `terminus/`, `candidate.json`, HPO docs or existing performance configs. Run `verify-anchor --checkout`, then instantiate the target on `torch.device("meta")` and compare the actual manifest/count before first native use. Save that manifest outside Git as a protected reference artifact.

Bootstrap instrumentation is itself a reviewed baseline change: integrate the block timer into the reference trainer without changing update arithmetic, prove equivalence, and establish an instrumented-reference commit. This is experiment S0, not a speed improvement. The subsequent candidate implementations may edit model execution, FP8/pointwise/attention/loss kernels, optimizer execution, loader/prefetch and scheduling **implementations** under the frozen mathematical contract. Protected checker/evaluator/reference copies stay outside those worktrees.

## Freeze a campaign

Copy `speed_research/campaign.template.json` outside Git. Fill every null, set `status` to `ready`, and use complete SHA-256 identities and a versioned quality protocol. The template's 74 GB peak limit leaves 6 GB headroom on an 80 GB board and deliberately catches designs that fit the research NVL's 94 GB but fail production. Replace it before locking if confirmed production capacity warrants a different reserve. The speed thresholds are initial policy choices: calibrate baseline noise in S0, then freeze; never loosen them for a disappointing candidate.

```bash
python3 -m speed_research lock --campaign /external/speed/campaign.json \
  --output /external/speed/campaign.lock.json
python3 -m speed_research analyze --lock /external/speed/campaign.lock.json \
  --evidence /external/speed/trial-001/comparison.json
python3 -m speed_research append --ledger /external/speed/ledger.jsonl \
  --receipt /external/speed/trial-001/receipt.json
```

The tool rejects incomplete configurations, changed protected checker files, mismatched control hashes, non-native screens, unbalanced/unmatched measurements, insufficient blocks, counter manipulation, reported trial overspending and corrupted/duplicate ledger records. `check-config` permits a small explicit execution-option surface while rejecting layer/context/gain/precision-recipe shortcuts; the actual native computation still requires independent review. New scheduling/tile controls may live in a separate committed execution configuration, leaving the model contract fixed. Class N requires an explicitly versioned numerical contract/checker before its altered recipe can qualify. It returns only speed-screen decisions; it cannot attest to the truth of supplied measurements or pass quality/production gates automatically. A trusted outer supervisor must verify raw artifacts, enforce research/VM-on caps and archive each receipt. A locally recomputable hash is not an authorization token.

## Relationship to Karpathy autoresearch

We inspected the actual [upstream program](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md), not just the name. Its loop is: fixed five-minute training trial, agent edit, immutable evaluator, measure validation bits/byte, keep/discard, repeat. We retain short experiments, a clear editable surface, protected measurements and an agent-readable research program. Here architecture and useful token exposure are fixed, quality is a constraint, speed is the objective, and multiple evidence stages prevent a cheap microbenchmark becoming a false winner. Failed commits/receipts are retained instead of upstream's destructive reset pattern. The loop is bounded by the separately authorized research budget.
