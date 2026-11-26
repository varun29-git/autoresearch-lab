# Reproducibility guide

This repository contains source, configurations, campaign protocols, and summarized evidence. It deliberately excludes corpora, token shards, checkpoints, generated trial ledgers, credentials, private infrastructure configuration, and third-party paper PDFs. Exact campaign reproduction therefore requires the external inputs identified below; a fresh clone can reproduce the CPU checks, not the GPU results by itself.

## 1. Retained source snapshots

Historical reports name the original campaign commits. Consolidation retained the research history but not every original object ID, so use this map to find the available snapshots. The right-hand commits are the retained source/review points; the mapping does not assert that unavailable original objects can be fetched from this repository.

| Campaign record | Original ID in reports | Retained snapshot |
| --- | --- | --- |
| Architecture baseline | `948663d` | `0ffda55` (baseline review; code derives from the fixed campaign baseline) |
| Peri-LN trial | `14d62aa` | `c012d88` |
| Value-residual trial | `ee71e32` | `07a72f3` (includes the checkpointing correction) |
| Selected Peri-LN + value-residual stack | `8bd5523` | `9e3bba8` (stack with final parity correction) |
| E3 implementation | — | `c048957` |
| E4 implementation and rounding correction | — | `207bc1d`, `0eb3210` |
| E3/E9 composition | — | `ba9c2e5` |
| Combined E3/E4/E9 research base | `1e8c0a4` | `babe242` (retained combined-stack research/documentation point) |
| Speed source named by early HPO notes | `8a62293` | `a66616b` (retained completed speed tip) |
| Spectral-Muon source named by early HPO notes | `8153a78` | `8ce81c9` |
| Final HPO candidate | — | `5e5f4ab` |
| Architecture reports and final campaign record | — | `5c0ca02` |

The final release also contains later speed experiments. Use the campaign snapshot, not the release tip, when auditing a historical measurement. Do not silently substitute spectral Muon into the architecture or speed campaigns: those used their recorded legacy optimizer rules. HPO used `spectral_unclamped` explicitly.

## 2. Evidence and external inputs

| Campaign | Checked in | Required for exact rerun or independent reanalysis |
| --- | --- | --- |
| Architecture | Trial proposals, reviews, final report, source snapshots, tokenizer identity, campaign policy and runner | English train/validation manifest and shards, exact H100 runtime, optional task manifest, external checkpoints, events/stability/evaluation records, and archive receipts |
| E3 and combined kernels | Trial summary, measurement protocol, source snapshots, three paired rates and archived soak summary | Locked real-data stream and manifest, exact H100 NVL runtime, native numerical receipts, paired `block.json` files, and the missing full 80-update JSON if sustained-log reanalysis is required |
| HPO | Fixed reference, final candidate, runner/ranker, research note, selection commit message | Manifest with SHA-256 `3c70bef91d09b2c31194887191a8058899d199f3bb4ebbed41c8691b3b521575`, its 415 train and 3 validation shards, exact runtime, candidate result files, curves, checkpoints, and complete ranked ledger |
| Older Lean dense/QAT context | Analysis reports and selected measured summaries | Original run records, checkpoints, data views, and any linked generated receipts not present in Git |

The architecture tokenizer is `tokenizer/tokenizer-65k.json`, SHA-256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`. It is a third-party derived asset under the terms recorded in `tokenizer/tokenizer-65k-freeze.md` and `NOTICE`; the Apache-2.0 project license does not replace those terms.

The architecture report's external Blob paths, HPO source locations, and older cloud run paths identify provenance, not downloadable public artifacts. No availability or redistribution right is implied. Private account names, subscription IDs, local home paths, and SSH details have been removed from public examples.

## 3. CPU verification from a fresh checkout

Use Python 3.11 to match CI. The Muon extra installs the pinned external optimizer dependency; it is not vendored.

```bash
python -m pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[muon]'
python -m compileall -q terminus tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
python -m terminus.preflight --config configs/1.5b_candidate.json --count-only
bash -n scripts/train.sh
git diff --check
```

These commands perform local compilation, unit/protocol tests, configuration and parameter accounting, shell parsing, and whitespace validation. They do not allocate a VM, launch training, validate CUDA/DeepGEMM/FA3 arithmetic, or reproduce a campaign metric. The old `configs/1.5b_candidate.json` used by the count-only CI gate is a compatibility fixture, not the selected Stack 4 speed or HPO recipe.

## 4. Auditing a campaign without starting GPU work

1. Read [RESULTS.md](RESULTS.md) first and identify the campaign boundary.
2. Check out the retained source snapshot from the table in a separate working copy when source inspection is needed. Do not mutate the release branch merely to inspect a report.
3. Compare the locked configuration and source hashes in the external receipt with that snapshot. A branch name or a short hash alone is not an immutable runtime identity.
4. For architecture, follow the measurement and scoring contracts in `program.md`, `research/policy.json`, and the final report. Treat `program.md` as a completed protocol, not authorization for another run.
5. For speed, use `docs/speed/MEASUREMENT.md` and `scripts/s0_block.py` to understand the complete-update boundary. Compare only equal model, token stream, update size, microbatch/accumulation, and warm-up/measurement policy.
6. For HPO, validate an externally supplied result set before ranking it:

   ```bash
   python -m hpo rank --updates 1272 --results-root /external/hpo-results
   ```

   The ranker checks the reference and manifest identities, but an absent external ledger means no local ranking is possible.

Native reruns require the exact external data and runtime plus a separate explicit execution decision. This guide does not launch cloud resources or authorize paid work.

## 5. Paper and source references

`RESEARCH_LIBRARY.md` links canonical arXiv pages. Other reports link first-party papers, model configurations, and source revisions where available. Downloaded paper PDFs are not included and no `papers/` path is required. External literature motivates hypotheses; it is not Terminus measurement evidence. Check the license and immutable revision of any external code or dataset before reuse.

## 6. Known reproducibility limits

- Original campaign IDs listed as unavailable do not resolve in the consolidated object database; the retained-snapshot map is the public audit path.
- The complete 80-update combined-kernel JSON and full HPO ranking table are not checked in.
- Some older evidence documents link generated JSON receipts that were intentionally not imported. Their summaries remain historical evidence, not self-contained raw data.
- One seed and one English proxy dominate the architecture and HPO comparisons. No repeated-seed uncertainty, width transfer, multilingual quality, or production-cluster scaling can be reconstructed from these records.
