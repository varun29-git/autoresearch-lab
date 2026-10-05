# Running the stage-one architecture campaign

This is an **execution runbook**, not a record of a launched run. The owner supplies the GPU machine, prepared data and benchmark files, and starts the Kimi K3 session. The repository never creates a VM. Keep campaign configuration, token shards, task examples, checkpoints and results outside Git.

## 1. Stage external inputs

Use the prepared ModelTunnelv2 65K-tokenizer stream and a manifest with the ten sources `en, hi, mr, kn, te, ta, gu, bn, code, math`, fixed train/validation splits, source provenance and a single fixed source mixture. The manifest's source paths must resolve under `data_root`, and its tokenizer path must point to the frozen JSON artifact with SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. The training view must contain enough usable source tokens for a 2.5B-token trial. The [existing token-set build receipt](TOKENSET_LIVE_BUILD_20261001.md) describes the S3-packed set; it is a source lead, not a local file in this repo. Do not retokenize or copy full data into Git.

Prepare a **development-only** task manifest following [the example](../research/task_manifest.example.json). It needs at least one answer-only task for each domain. Each entry declares `domain`, `task`, `path`, SHA-256, `role=development`, `answer_only=true`, dataset `source` and `split`. Each JSONL row has a stable optional `id`, a `prompt`, and the short verified answer as `reference` (optional `choices`/`correct_choice` are recorded for a later accuracy adapter). Do not put generated rationales in `reference`. Hashes and all ten domains are checked before the campaign locks. Keep an untouched final task suite separately; it never enters the Kimi loop. Document the train/task contamination check in the prepared-data provenance.

The current repository does **not** include a complete seven-language development task suite. The existing ScalingBench material is English and the recorded MILU route is gated. If this input is unavailable, `prepare.py` must fail and the paid search should wait rather than silently choose an English-only architecture.

Copy [campaign.example.json](../research/campaign.example.json) to an external path and replace every placeholder with absolute paths. `results_root` should be persistent storage with enough space for all trial logs and checkpoints. Check the ten-domain score weights, 190M–210M parameter envelope, 0.5B/2.5B token horizons and 12 GPU-hour cap in [policy.json](../research/policy.json) **before** creating a campaign tag. Changing them later requires a new tag and lock.

## 2. Prepare on the intended native GPU machine

Use the pinned H100/SM90 PyTorch, Triton, DeepGEMM and FA3 environment documented in the existing [H100 runbook](../experiments/terminus_lean_150m_3b/AZURE_H100.md). The architecture baseline has learned Q/K RMSNorm and may be slower than the old 150M fused path; measure its actual rate before committing to an overnight trial count. An A10G run is a different arithmetic/backend experiment.

Open the private repo in Kimi Code with its K3 model. The first message can be: **“Read `program.md` and execute the prepared Terminus architecture campaign at `/absolute/path/to/campaign.json`. Research and choose architecture ideas yourself. Stay within the campaign budget and report a shortlist, not a final 1.5B claim.”** No Kimi API key or provider config belongs in this repo. A Kimi CLI launcher has not been validated here; the program is agent-interface agnostic.

Create the fresh campaign branch, then prepare:

```bash
git switch -c autoresearch/architecture-20261005
python prepare.py --campaign /absolute/path/to/campaign.json
```

Preparation performs data/task hash and schema checks, validates every shard and the fixed tokenizer, checks model parameter count, and writes an immutable `campaign_lock.json` under `results_root/tag`. This can take time because full shard hashes are read. It uses CPU only and does not train. If it fails, fix the external input or make a new campaign version; do not edit the validator to pass.

## 3. Establish baseline, then let Kimi iterate

```bash
python train.py --campaign /absolute/path/to/campaign.json --trial-id baseline --stage screen
python train.py --campaign /absolute/path/to/campaign.json --trial-id baseline --stage full
python -m research.loop status --campaign /absolute/path/to/campaign.json
```

For each researched, committed architecture candidate, Kimi runs:

```bash
python train.py --campaign /absolute/path/to/campaign.json --trial-id trial-001 --stage screen
python -m research.loop status --campaign /absolute/path/to/campaign.json
python train.py --campaign /absolute/path/to/campaign.json --trial-id trial-001 --stage full
```

The full stage resumes from the exact screen checkpoint and the same full-horizon WSD schedule. The `trial-001` branch must remain checked out at its recorded commit for continuation. A failed or timed-out trial remains in `ledger.jsonl` and may resume if a valid checkpoint exists; do not reuse the trial ID for different source. The runner rejects changes to protected code, task/data hashes, or the parameter envelope. It performs native numerical preflight before first training of each candidate. Its stdout is captured in per-step log files; `status` prints primary score, per-domain changes and a paired-batch interval. Inspect `run/events.jsonl`, `run/stability.jsonl`, `run/incidents.jsonl`, full-stage task records, and the actual commit diff before promoting a candidate.

The runner's 12-hour cap counts preflight, training and evaluation wall time on one GPU. Its per-trial timeout sends SIGTERM so Terminus can commit an interruption checkpoint. A result is **completed** only after the target tokens and protected checkpoint evaluation exist. Kimi must stop at the cap, the human's stop instruction, or a real readiness blocker. Results are append-only and branches are retained.

## 4. Verify remote archive before stopping the paid machine

The local result tree includes every trial's metadata, config, data view, preflight receipt, logs, training events, stability/incident records, checkpoint files and hashes, evaluations, and ledger. Generate a whole-tree SHA-256 receipt:

```bash
python -m research.artifacts create --root /persistent/results/architecture-20261005 --receipt /persistent/results/architecture-20261005/artifact_receipt.json
```

Upload the tree using the owner's approved S3 or Blob destination. Download it independently to another directory (or a separate verification machine) and run:

```bash
python -m research.artifacts verify --root /verification/download/architecture-20261005 --receipt /persistent/results/architecture-20261005/artifact_receipt.json
```

**Only after verification succeeds for the full tree, including checkpoints and every log, may the VM be deallocated.** An upload command's exit status or object-size listing alone does not satisfy this gate. If verification fails, retain the machine and repair the archive. This repository deliberately does not include an automatic cloud shutdown command.

## 5. Decision boundary

Kimi returns a reviewable shortlist with candidate commit hashes, source citations, completed token exposure, all ten held-out losses, task/position diagnostics, stability events, actual GPU-hours, and reasons to keep or discard. The stage-one output is a set of architecture hypotheses. Stage two must test larger-width transfer, LR/batch/μP interactions and matched dense-to-ternary QAT before the final model architecture is fixed. Stage three may then search kernels under numerical parity and full-update throughput measurement.
