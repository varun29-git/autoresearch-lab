# CI-1 production handoff

**2026-11-08 QAT finishing schedule:** the selected main QAT allocation has
100B recovery/stable content tokens followed by 100B finishing content tokens.
Cosine LR decay covers the full finishing window, ending at 0.1 of QAT peak;
this supersedes the older 20%-cooldown candidate. Resolve actual scored-target
boundaries from the packed receipts before generating the training config.
See [QAT allocation](amd/QAT_ALLOCATION.md). Packing quotas are unchanged and
native QAT/packed-loader acceptance remains pending.

**2026-11-08 dataset-child mixing:** the [packed-data contract](PACKING.md)
now includes a CPU reader that interleaves individual dataset children inside
each language/source, rather than consuming their physical creation order.
Independent descriptor indexing can run ahead of the warmup allocation ledger.
Future fast packers emit descriptor offsets and a completed-allocation mixture
candidate. Older completed warmup reports resolve their separate audit ledger
with the same inherited URI, size and checksum checks before metadata finalization.
Native segment-aware trainer integration remains pending.

**2026-11-07 CPU packing optimization:** the [parallel preparation worker](PREPARATION_QUEUE.md)
adds a compact RAM allocation ledger, bounded source prefetch, native JSON
processing, multiprocess packing and overlapped verified uploads. CPU tests
compare complete packed payload and segment hashes with the reference worker.
Measured performance and deployment receipts remain external; this change
does not establish native training-loader acceptance.

**2026-11-07 phase handoff:** the [CPU preparation queue](PREPARATION_QUEUE.md)
adds hashed allocation-ledger handoff, protected finishing records and bounded
CPU dispatch. Physical candidate data still requires native loader acceptance.

**2026-11-07 CPU packing implementation:** the [document packer](DOCUMENT_PACKER.md)
now creates candidate uint16 shards with segment metadata, deterministic
best-fit packing, contiguous long-document windows, heldout exclusions and
source quotas. Native training-loader integration remains pending.

**2026-11-07 packing preparation:** the [document-packing integration contract](PACKING.md)
records the loader, AMD attention, target-count and graph changes required by
the external multilingual data plan. Current continuous streams do not yet
implement those document boundaries. The plan preserves the selected tokenizer,
source mix and optimizer policy; CPU and native acceptance remain separate.

**2026-11-07 tokenizer decision:** fresh source conversion can use the qualified
[65K atom revision and native C++ encoder](TOKENIZER_V2.md), retaining 27 reserved
IDs and the existing model geometry. Original HF/custom bins cannot be treated
as compatible based on vocabulary alone. The external conversion campaign must
finish source/holdout reconciliation and packing before training acceptance.
Historic checkpoints remain pinned to their original artifacts.

**2026-11-06 whole-program budget:** include posttraining, recovery and
finalization in the same allocation using the CPU-only
[program budget and unique-data cap](amd/pretraining_readiness/PROGRAM_BUDGET.md).
The current source-specific phase/mix decision packet is external. The tool
funds remaining QAT and named reserves before extending dense; it does not
implement SFT/DPO or automatically launch/switch paid phases.

**2026-11-04 dense LR decision:** the user selected tied embedding/head AdamW
**0.001** for AMD production, with spectral Muon **0.017** and other groups
preserved. Regenerate fresh external configs using the
[current optimizer policy](amd/OPTIMIZER_PROMOTION.md). Historical H100 trial
packets and QAT rates are preserved; native and long-horizon qualification
remain open.

**2026-11-03 tokenizer update:** an
[encoder-preserving stock decoder candidate](DECODER_REPAIR.md) and contextual
streaming reference are available. Generate the candidate outside Git; keep
corpus/checkpoint hashes pinned to their original encoder artifact. Corpus
source acceptance, ambiguity-policy review and actual output-stack qualification
remain open. Decoder repair preserves existing encoding; incompatible source
execution paths need [source-based conversion](SOURCE_RECOVERY.md) before
they can share a production manifest.

**2026-11-02 readiness update:** the single-H100 qualification campaign is
complete; selection/transfer/full-size confirmation receipts are external.
The AMD baseline now promotes the confirmed dense embedding LR and includes
CPU-tested checkpoint/budget/recovery tools. Consistent tokenizer/data provenance,
native qualification and actual provider controls remain open. Start with the
[current readiness entry](amd/pretraining_readiness/README.md).
The AMD ternary port is implemented, with native checks still pending.

**Historical 2026-11-01 H100 qualification plan:** The next single-H100 allocation is
limited to 24 hours. The [new staged 24-hour campaign](h100_24h/README.md)
shortens the five matched proxy arms to about 2B tokens each, credits H's
first 725 updates only after a verified future-only schedule fork, and keeps
one 600M-token full-size confirmation. Its automatic controller advances only
healthy complete arms and reforecasts the whole remaining budget. The earlier
5B/50-hour Kimi handoff below is historical and remains unchanged.

**Before an eight-MI325X training campaign:** read the
[`amd/pretraining_readiness/` entry point](amd/pretraining_readiness/README.md).
It records the unresolved batch, QAT and native-hardware gates for the
5,760 GPU-hour allocation.

**2026-11-01 production policy:** close the
[data, evaluation and numerical quality gates](amd/pretraining_readiness/QUALITY_GATES.md),
then use the [adaptive compute plan](amd/pretraining_readiness/ADAPTIVE_COMPUTE_PLAN.md)
to fund a measured token horizon inside the fixed allocation. Extend the WSD
plateau while reserving complete finishing phases. The current trainer still
uses a fixed manifest horizon; [prospective stopped-checkpoint transitions](amd/HORIZON.md)
now preserve the consumed schedule/data prefix. Automatic multi-phase orchestration
and native qualification remain open. The focused five-billion-token proxy comparisons
below remain fixed and comparable.

**2026-10-28 qualification policy:** the latest direction reuses completed HPO
and spends up to 50 hours on one Azure H100 on a
[proxy LR/batch checks followed by winner transfer](amd/pretraining_readiness/LR_BATCH_MONITORING.md),
comparing existing and lower published-rate hypotheses at the same global
batch for **5B tokens per proxy trial**. Only the transferred winner receives
a short full-size confirmation. The current plan keeps the 50-hour cap and
defers the optional extra proxy batch comparison unless measured time permits.
The [Kimi execution handoff](kimi_h100_5b/README.md) packages the current
five-billion-token proxy matrix, exact schedules, implementation tasks,
monitoring, evidence requirements and CPU budget checker. Start Kimi with
[`KIMI_START.md`](kimi_h100_5b/KIMI_START.md).
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

The model has 24 layers; width 2304; FFN width 5760; 18 query and 6 KV heads; a tied 65,536-token embedding/readout; 4096-token context; 512-token local attention in 20 layers and global attention at layers 6, 12, 18 and 24; Peri-LN; learned value residual; and 1,446,478,126 trainable parameters. The operator-reviewed change is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md). The new geometry has completed a short single-H100 qualification; its receipts are external. This does not certify MI325X throughput or long-run quality. It retains the original implementation: FA3 attention, DeepGEMM FP8 linear operations, packed QKV/MLP and pointwise kernels, BF16 residuals, and streaming loss. The dense optimizer is spectral Muon (`eta × sqrt(d_out/d_in)` on each logical matrix, without Moonshot's width-growing factor) plus AdamW for the tied embedding and scalar/norm groups. See `docs/MUON_SPECTRAL_IMPLEMENTATION.md` for the exact update.

The historical H100 starting rates are Muon 0.017; tied-embedding AdamW 0.003; ordinary norm 0.0003; learned Q/K norm 0.0001; hidden-matrix weight decay 0.02; tied-embedding weight decay 0. The Muon and small-group settings come from the completed 200M campaign. In the 700M width bridge at 500M tokens, embedding LR 0.003 had lower English cross-entropy than 0.00424 or 0.006 under matched settings. The later H100 selection/transfer confirmed embedding **0.002**; the user subsequently chose **0.001** for AMD long pretraining under the [current optimizer policy](amd/OPTIMIZER_PROMOTION.md). The historical recipe remains reproducible. Campaign records remain external to Git. The dense stage has no QAT. The current AMD policy [reserves a qualified fixed QAT budget and extends dense when affordable](amd/HORIZON.md), using measured phase rates and native QAT qualification.

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
