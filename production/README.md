# CI-1 production handoff

**Before an eight-MI325X training campaign:** read the
[`amd/pretraining_readiness/` entry point](amd/pretraining_readiness/README.md).
It records the unresolved batch, QAT and native-hardware gates for the
5,760 GPU-hour allocation.

**2026-10-28 qualification policy:** the latest direction reuses completed HPO
and spends up to 50 hours on one Azure H100 on a
[focused full-model batch-transfer check](amd/pretraining_readiness/AZURE_H100_BRACKETS.md),
using the existing rates and a conditional matrix-decay comparison.
The [broader eight-MI325X plan](amd/pretraining_readiness/FULL_QUALIFICATION_PLAN.md)
retains the longer learning, QAT and recovery gates and an optional larger
qualification envelope. The recipe below is the code's reviewed
baseline; it is not automatically the winner of the new campaign.

This folder is the entry point for the selected Terminus Stack 4 implementation. The standalone `CI-1` checkout contains the full model, optimizer, FP8 kernels, streaming data loader, DDP trainer, checkpointing, tests, scripts, completed research history, and `manuscript/terminus_draft.md`. It was created from `paper/technical-draft-v1` at `fa84773`, which includes the final 200M HPO winner at `36d3de9`, the spectral-Muon change, and the E3/E4/E9 speed stack. Historical research files remain available, but the 30-layer/96K configs and pre-spectral 25-layer configs in `configs/` are not the selected production recipe.

The selected settings now live in [`recipe.py`](recipe.py); config generation
does not import the archived HPO campaign. Follow
[`PARALLELISM.md`](PARALLELISM.md) for the initial DDP strategy, CPU run inspection,
startup scaling issue and cluster acceptance procedure. It also identifies the
runtime, recovery, evaluation and QAT work beyond data and parallelization.
See [`OPERATIONS.md`](OPERATIONS.md) for the implemented runtime, checkpoint,
evaluation, QAT transition and packed ternary CPU inference commands, plus the
remaining hardware checks. For the AMD allocation, the recommended dense/QAT
split and starting-rate candidates are now documented in
[`amd/QAT_ALLOCATION.md`](amd/QAT_ALLOCATION.md). Native AMD QAT execution and
quality transfer remain unqualified.

For the proposed **eight-MI325X, 5,760 GPU-hour** allocation, start with
[`amd/README.md`](amd/README.md). It contains the new gfx942 candidates, pinned
build recipe, native test gates, measurement tools, research and optimization
plan. AMD execution and throughput remain unmeasured. The H100 history and
6,000-hour calculations below describe the original allocation.
The eight-GPU batch, schedule and execution starting choices are now selected
in [`amd/CONFIG_SELECTION.md`](amd/CONFIG_SELECTION.md); use
`python3 -m production.amd.prepare` to generate them from an external manifest.

## Selected dense starting point

The model has 24 layers; width 2304; FFN width 5760; 18 query and 6 KV heads; a tied 65,536-token embedding/readout; 4096-token context; 512-token local attention in 20 layers and global attention at layers 6, 12, 18 and 24; Peri-LN; learned value residual; and 1,446,478,126 trainable parameters. The operator-reviewed change is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md). The new geometry has not been benchmarked on H100. It retains the original implementation: FA3 attention, DeepGEMM FP8 linear operations, packed QKV/MLP and pointwise kernels, BF16 residuals, and streaming loss. The dense optimizer is spectral Muon (`eta × sqrt(d_out/d_in)` on each logical matrix, without Moonshot's width-growing factor) plus AdamW for the tied embedding and scalar/norm groups. See `docs/MUON_SPECTRAL_IMPLEMENTATION.md` for the exact update.

The *starting* rates are Muon 0.017; tied-embedding AdamW 0.003; ordinary norm 0.0003; learned Q/K norm 0.0001; hidden-matrix weight decay 0.02; tied-embedding weight decay 0. The Muon and small-group settings come from the completed 200M campaign. In the 700M width bridge at 500M tokens, embedding LR 0.003 had lower English cross-entropy than 0.00424 or 0.006 under matched settings. This supports 0.003 as a production starting point, not a proven width law or a full µP guarantee. The 700M records remain external to Git. The dense stage has no QAT. Transition tooling is available; the AMD mixed-phase recommendation is 60/40 by tokens, with actual phase budgets dependent on both measured rates and native QAT qualification.

## Make a runnable config

The operator must first choose the production token manifest, global batch, warmup, and decay horizon. The builder reads the manifest's per-rank microbatch and token budget, derives gradient accumulation for the chosen world size, and refuses a batch or schedule that does not fit. It writes the run config outside Git. Its checkpoint/validation intervals are starting operational defaults in *updates*, to review for the actual run.

```bash
python -m production.build_dense_config \
  --manifest /external/production/manifest.json \
  --world-size 32 \
  --global-batch-tokens <chosen_multiple_of_32_x_manifest_batch_x_4096> \
  --warmup-tokens <chosen_warmup> \
  --decay-tokens <chosen_decay> \
  --output /external/production/dense_config.json
```

`<...>` marks decisions still to make; do not paste the example as a launch command. The manifest controls total dense tokens and data mixture. Check its tokenizer and shards against the model and data contract. `python -m terminus.preflight --config /external/production/dense_config.json --count-only` checks the parameter count without allocating the model. The native CUDA preflight produces the report consumed by `scripts/train.sh`. The production wrapper `production/train.sh` additionally requires a captured `RUNTIME_REPORT`. The launcher uses `CONFIG`, `MANIFEST`, `DATA_ROOT`, `OUTPUT_DIR`, `PREFLIGHT_REPORT`, and `NPROC_PER_NODE`; multi-node runs also set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. The human controls any paid GPU launch.

## What still needs production validation

The historical single-H100 NVL dense rate was about 41,000 tokens/s on the optimized 25-layer stack (three complete paired 20-update receipts; see `docs/speed/TRANSFER_32GPU.md`). At that rate, 6,000 GPU-hours would yield about 886B dense tokens before QAT, communication, checkpoints, and downtime. The 1.5T minimum needs an average of 69,444 tokens/s/GPU across *all* production GPU-hours, and the 2T target needs 92,593. These are arithmetic requirements, not achieved rates. The 32-GPU H100 NVL/PCIe topology, global batch, communication overlap, sustained throughput, and dense/QAT time split still need measured decisions. A single-GPU winner does not establish the production target.

Do not start a 1.5T run from an old config. First validate native kernels and count, then measure full 32-rank updates at the actual batch and data path, inspect slowest-rank and checkpoint costs, and calculate the dense plus QAT token forecast within the 6,000-hour aggregate budget. Save run manifests, generated configs, preflight reports, token shards, and checkpoints outside Git. This handoff prepared code and configuration tooling; it has not launched cloud GPUs or validated a 32-GPU run.
