# Running the stage-one architecture campaign

This is an **execution runbook**, not a record of a launched run. Kimi runs on a cheap controller and starts the existing Azure H100 only for one committed trial at a time. The VM deallocates after its trial archive is independently verified in Blob. The owner still supplies prepared English proxy data. Keep campaign configuration, token shards, optional task examples, checkpoints and results outside Git. See the [Azure experiment cycle](AZURE_EXPERIMENT_CYCLE.md) for the exact cloud contract and recovery behavior.

## 1. Stage external inputs

Use an English-only view of the prepared ModelTunnelv2 65K-tokenizer stream: fixed `en` train/validation shards, `source_mix={"en":1.0}` (or another predeclared English-source mixture), and a versioned provenance scope with `languages=["en"]` and `domains=[]`. The source paths must resolve under `data_root`; the tokenizer must match SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. The view needs about 2.5B usable English tokens for a dense trial or 2.6B if a conditional QAT probe is selected. The [existing token-set build receipt](TOKENSET_LIVE_BUILD_20261001.md) is a source lead; it is not a local file in this repo. Do not retokenize or copy full data into Git.

An English **development-only** task manifest is optional for stage one. If supplied, follow [the example](../research/task_manifest.example.json): pin source/revision and SHA-256, use short verified answer-only references, and check train/task contamination. The present evaluator measures teacher-forced answer loss; choice accuracy and code execution are not wired into this architecture loop. Keep untouched final tasks outside the loop.

Set `task_manifest` to `null` if no pinned English development suite is ready. Preparation still requires real, held-out English data. The seven-language task suite, code/math tasks and MILU are left for finalist transfer checks, not silently represented by this English-only score.

Copy [campaign.example.json](../research/campaign.example.json) to an external path and replace the data/result placeholders with absolute paths. `results_root` should be persistent storage with enough space for all logs and checkpoints. Review the English proxy, [scorecard](ARCHITECTURE_SCORECARD.md), 190M–210M parameter envelope, 0.5B/2.5B token horizons and 12 GPU-hour cap in [policy.json](../research/policy.json) **before** creating a campaign tag. Changing them later requires a new tag and lock.

## 2. Prepare the inexpensive controller and staged VM disk

Use the pinned H100/SM90 PyTorch, Triton, DeepGEMM and FA3 environment documented in the existing [H100 runbook](../experiments/terminus_lean_150m_3b/AZURE_H100.md). The architecture baseline has learned Q/K RMSNorm and may be slower than the old 150M fused path; measure its actual rate before committing to an overnight trial count. An A10G run is a different arithmetic/backend experiment.

Open the private repo in Kimi Code with its K3 model **on the controller**. The first message can be: **“Read `program.md` and execute the prepared Terminus architecture campaign through `research.azure_cycle`. Independently search and verify the papers and implementations each hypothesis or trial result calls for; research and commit architecture ideas while the H100 is deallocated. Stay within the VM-on budget and report a shortlist.”** Give the agent web/search access on the controller for this work. No Kimi API key or provider config belongs in this repo. A Kimi CLI launcher has not been validated here; the program is agent-interface agnostic.

Create the fresh campaign branch and commit it before paying for a VM boot. Fill [the Azure config example](../research/azure_cycle.example.json) outside Git with the actual managed-disk mount and campaign path. The source bundle sent to the VM is the committed controller HEAD; the VM does not need GitHub credentials. Inspect the existing VM without starting it:

```bash
git switch -c autoresearch/architecture-20261005
python -m research.azure_cycle inspect --config /absolute/path/to/azure_cycle.json
```

The first baseline cycle runs `prepare.py` on the VM's staged managed disk. Preparation verifies the English-only scope, shard hashes, tokenizer, optional task files and model parameter count, then writes the immutable campaign lock. If it fails, the controller deallocates before any trial begins. Fix the external input or make a new campaign version; do not edit the validator to pass.

## 3. Screen the baseline, then let Kimi iterate

```bash
python -m research.azure_cycle run --config /absolute/path/to/azure_cycle.json --trial-id baseline --stage screen
```

For each researched, committed architecture candidate, Kimi runs:

```bash
python -m research.azure_cycle run --config /absolute/path/to/azure_cycle.json --trial-id trial-001 --stage screen
```

Only after a candidate screen warrants a longer comparison, return to the baseline commit and run another Azure cycle with `--trial-id baseline --stage full`; then return to the candidate commit and run its `--stage full`. If all screens are unpromising, the baseline need not consume its remaining 2B tokens. A full stage resumes from the exact screen checkpoint and WSD schedule. The candidate branch must remain at its recorded commit. A failed or timed-out trial remains in `ledger.jsonl` and may resume if a valid checkpoint exists; do not reuse its ID for different source. The runner rejects protected-code, data/task-hash or parameter-envelope drift. Native numerical preflight runs before a candidate's first training. The worker saves `diagnostic_status_<stage>.json`, raw training/stability/incident records and sampled GPU telemetry into Blob; the controller downloads the analysis files after deallocation. Complete [the diagnostic review](../notes/REVIEW_TEMPLATE.md) before the next trial.

The runner's 12-hour cap counts preflight, training and evaluation; the controller separately counts VM-on time including boot, upload and verification, and reserves archive time. Its per-trial timeout sends SIGTERM so Terminus can commit an interruption checkpoint. A result is **completed** only after the target tokens and protected checkpoint evaluation exist. Kimi must stop at the cap, the human's stop instruction, or a real readiness blocker. Results are append-only and branches are retained.

Most architecture trials end with dense FP8. Before each candidate Kimi writes a [mathematical and empirical proposal](../notes/TEMPLATE.md), including whether later ternary conversion could change the result. For a candidate with a specific quantization-sensitive mechanism and completed matched dense full runs, the optional diagnostic is:

```bash
# On the baseline commit, then repeat on the candidate's exact commit and trial ID.
python -m research.azure_cycle run --config /absolute/path/to/azure_cycle.json --trial-id baseline --stage qat_probe --qat-rationale "Specific quantization mechanism, evidence, and predicted conversion signal"
python -m research.azure_cycle run --config /absolute/path/to/azure_cycle.json --trial-id trial-001 --stage qat_probe --qat-rationale "Specific quantization mechanism, evidence, and predicted conversion signal"
```

The baseline probe must complete first. Both probes continue from their 2.5B-token dense checkpoints through about 100M matched extra tokens with the same source mixture and FP8 activations. The protected conversion carries compatible weights, optimizer state, RNG and data cursor; the probe saves its transition receipt, immediate zero-update evaluation, QAT logs and final evaluation. `status` reports both absolute QAT quality and the relative conversion/QAT penalty. This short, fixed recipe detects sensitivity but is not the final dense-to-ternary tuning campaign. If the candidate has no specific QAT-sensitive hypothesis, skip the probe and preserve the budget.

## 4. Verified shutdown and offline analysis

The worker creates a SHA-256 receipt for every file in that trial directory, uploads to its unique Blob prefix, downloads every file independently, verifies the entire tree and then requests VM deallocation. The controller confirms the receipt and downloads the analysis files. To fetch the checkpoint binaries too:

```bash
python -m research.azure_cycle fetch --config /absolute/path/to/azure_cycle.json --cycle-id <id> --include-checkpoints
```

If upload or verification fails, the worker **does not deallocate**; preserve the VM and repair the archive from the persistent disk. If the controller loses contact during shutdown, check the Azure power state and receipt before reconciling the local cycle ledger. Do not start a second trial while the first cycle is unresolved.

## 5. Decision boundary

Kimi returns a Pareto shortlist with candidate commits, source citations, mathematical proposal notes, English loss and learning curves, numerical incidents and sampled scale behavior, training speed, memory, parameter count, GPU-hours, and QAT-probe results or reasons for skipping. The stage-one output is a set of architecture hypotheses. Stage two must check Indic/code/math and larger-width transfer, LR/batch/μP interactions and longer matched dense-to-ternary QAT before fixing the final architecture. Stage three may search kernels under numerical parity and full-update throughput measurement.
