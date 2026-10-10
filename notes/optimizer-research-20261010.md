# Optimizer research for the dense 200M Stack 4 HPO (campaign stack4-hpo-200m-en-v2)

Date: 2026-10-10. Branch: hpo/autoresearch-v1. Researcher: Kimi.
This note backs the four required research choices in `hpo/candidate.json`
(Muon LR, tied-embedding AdamW LR, matrix weight decay, embedding weight
decay) and lays out the adaptive trial plan. Fixed by campaign:
`spectral_unclamped` Muon rule, model, data, 393,216-token update, WSD with
250M-token warmup, 1,272-update (500,170,752-token) endpoint, scorer.

## 1. LR-scale conversions (critical)

Published "Muon LR" numbers live on incompatible scales:

| Convention | Update rule | Per-element update RMS |
| --- | --- | --- |
| Keller (KellerJordan/Muon `muon.py`) | `lr * sqrt(max(1, d_out/d_in)) * NS5` | `lr / sqrt(fan_in)` |
| Moonlight (arXiv 2502.16982, Eq. 4/7) | `eta * 0.2 * sqrt(max(A,B)) * NS5` | `0.2 * eta` |
| This campaign (`spectral_unclamped`) | `lr * sqrt(d_out/d_in) * NS5` | square 768x768: `lr/sqrt(768)` = 0.0361*lr |

For our square Q/O matrices the spectral rule equals the Keller rule
(clamp inactive), so our `muon_lr` is directly on the Keller scale.
Conversion from Moonlight-scale eta to our scale at width 768:
`lr_here = eta * 0.2 * sqrt(768) = 5.543 * eta`.

Role-dependent RMS under the fixed spectral rule (width 768/FFN 1920):
Q, O, gate, up: `lr/sqrt(768)`; K, V: `lr/sqrt(2304)`; down: `lr/sqrt(4800)`.
This asymmetry is part of the fixed rule, not a tunable.

## 2. Source anchors

### Muon matrix LR (converted to our Keller-compatible scale at width 768)

| Source | Published value | Converted | Context |
| --- | --- | --- | --- |
| Moonlight 16B-A3B (2502.16982 §3.3) | eta 4.2e-4, wd 0.1, 16.8M-token batch, 5.7T tokens, cosine | 2.3e-3 | huge batch/model; long decayed run |
| Moonlight dense scaling-law 0.4-1.5B (Table 2) | eta 8.3e-4 to 9.5e-4, batch 0.79-2.1M tokens, 9-39B tokens | 4.6e-3 to 5.3e-3 | dense, closest production analogue |
| Kimi K2 (2507.20534 §2.5) | eta 2e-4, wd 0.1, 67M-token batch, 15.5T tokens, WSD | 1.1e-3 | giant batch; MoE; MuonClip |
| GLM-4.5 (2508.06471 §2.4) | eta 2.5e-4, wd 0.1, 16->64M-token batch, cosine, RMS 0.2 | 1.4e-3 | giant batch; MoE |
| modded-nanogpt record (05/25/25, 124M, width 768) | lr 0.05 Keller-scale, momentum 0.95, wd 0, batch 393,216 tokens | 0.05 | same batch/width as ours; no-warmup trapezoid speedrun, 0.7B tokens |
| KellerJordan/Muon README default | lr 0.02 | 0.02 | generic default |
| On-model 250-update screen (terminus-optimizer-evidence, results.tsv) | spectral_unclamped lr 0.017 beat 0.003 by ~1.05 CE; best of screen (CE 4.17 at 98.3M tokens) | 0.017 | only on-model evidence; mid-warmup screen, not an endpoint |

Short-horizon, constant-peak-LR trials tolerate higher LR than
multi-trillion-token decayed production runs. The optimum for our 500M-token
endpoint should sit between the dense-production band (~5e-3) and the
speedrun band (~0.05), near the on-model screen value.

### Tied embedding/readout AdamW LR

- Moonlight (§2.2): AdamW handles RMSNorm/LM head/embedding and **shares eta
  and lambda with Muon by design** ("Muon can directly reuse the learning
  rate and weight decay tuned for AdamW"). Applied to our scale:
  `adamw_lr = muon_lr / 5.543`. At muon_lr 0.017 -> 3.1e-3.
- RMS-matching arithmetic agrees: AdamW update RMS ~0.2*lr (Moonlight §2.2,
  App. A: 0.2-0.4); 0.2*3e-3 = 6.0e-4 matches the Muon square-matrix RMS
  0.017/sqrt(768) = 6.1e-4.
- KellerJordan/Muon README: AdamW 3e-4 for embed/head/gains (contrast
  candidate, 10x lower).
- modded-nanogpt: embed Adam lr 0.6 / head 0.22 — untied tables, wd 0,
  soft-capped head, value embeddings folded in; not transferable to a tied
  table (parameterization differs).
- muP (Tensor Programs V, Table 8 / §3): with a tied table, Adam LR Theta(1)
  in width (exponent 0) with a 1/fan_in output multiplier; our model lacks
  that multiplier, which is exactly why the 700M/1.5B bridge tests exponents
  0, 0.5, 1 via `hpo.transfer`.

### Weight decay

- Matrices: 0.1 is the uniform production choice (Moonlight wd 0.1 all
  stages; K2 "0.1 throughout"; GLM-4.5 0.1; nanoGPT 0.1). Moonlight added wd
  because weight/output RMS grew past bf16 range without it; their App. D:
  decay on RMSNorm gamma is stability-critical (our norm groups run wd 0 at
  small LR — monitor, fixed field). Keller README uses 0.01 (short speedrun
  regime). Note `LR x wd` accounting: at muon_lr 0.017, wd 0.1 gives
  1.7e-3/step shrinkage; over 636 post-warmup peak steps, cumulative pure
  decay factor ~0.34. A contrast at 0.02 (Keller-scale) is owed.
- Embedding: nanoGPT decays the tied table at 0.1 ("all weight tensors in
  matmuls + embeddings decay"); Moonlight's released code decays
  embed/head with the shared wd (paper silent, treat as implicit 0.1).
  Counter-consideration: decoupled decay hits all 65,536 rows every step
  while rare-token rows get rare gradient (at 393k tokens/batch the average
  row appears ~6x). Contrast 0.0 is owed.

### Norm and Q/K-gain LRs (required explicit choices per b493de9)

- KellerJordan/Muon README puts hidden gains/biases in the AdamW group at
  3e-4 (same rate as embed/head there). modded-nanogpt runs scalars at 0.04
  (no-warmup speedrun regime; not transferable). Moonlight shares one eta
  across Muon and AdamW groups, which under my tied-table choice would imply
  3e-3 for gains — hot for 1-D gains with wd 0.
- Choice: `norm_lr` 3e-4 (Keller convention; 10x below the tied-table rate),
  `qk_norm_lr` 1e-4 (extra conservative: Q/K gains directly scale attention
  logits; Moonlight App. D flags norm-gamma behavior as stability-critical;
  matches archived Terminus family practice).
- momentum 0.95 (Moonlight: no gain from tuning; speedrun and Keller default
  agree), betas (0.9, 0.95), eps 1e-8, grad clip 1.0 stay at reference.
  Revisit only if trials show instability or update/weight-ratio anomalies.

### The NYU critique of the Moonlight rule (added post-launch, user pointer)

Qiu, Chen, Phan, Lei & Wilson (NYU), "Hyperparameter Transfer Enables
Consistent Gains of Matrix-Preconditioned Optimizers Across Scales", arXiv
2512.05620 (NeurIPS 2025). Appendix G shows Moonlight's `0.2*sqrt(max(A,B))`
rule ("Muon-Kimi") **fails to transfer the optimal LR across width**, with or
without an added Adam-style 1/d_in factor; only the `eta*sqrt(d_out/d_in)`
spectral rule — this campaign's fixed `spectral_unclamped` — transfers, with
the base LR Θ(1) in width. That independently supports (a) the campaign rule
choice, (b) `hpo.transfer`'s plan to hold the Muon base rate constant across
the 700M/1.5B width bridge, and (c) Adam for the tied embedding/readout
(their recommendation matches our optimizer split). Their §3.4 also argues
for decoupled weight decay λ = Θ(1/D) over the LR-coupled AdamW convention,
warning that LR-coupled decay "can be highly suboptimal for other optimizers
(e.g., Muon)" — this reinforces the planned matrix-wd contrast (trial 5) and
the LR×wd accounting. Caveat: the critique is about **width transfer**; at a
single width either rule can be re-tuned, and Dion3 (2608.11612) reports the
Moonlight factor working well at larger scale. Counter-evidence noted, not
hidden.

### Missing-in-literature (recorded honestly)

K2: momentum value, Nesterov flag, NS steps, param split, embed LR. GLM-4.5:
non-Muon group settings, warmup length. Qwen3, Gemma 3: no optimizer
disclosures at all. MiniCPM4: optimizer never named in the paper — the
"MiniCPM4 uses Muon" claim is unsourced. MiniCPM: Adam at muP base LR 0.01,
batch 3.93M tokens, WSD with exponential decay over ~10% of tokens, wd not
used.

## 3. Candidate 1 (committed as hpo/candidate.json)

| Field | Value | One-line reason |
| --- | --- | --- |
| `muon_lr` | 0.017 | On-model screen winner; between dense-production band (5e-3) and same-batch speedrun (0.05) on Keller scale |
| `adamw_lr` | 0.003 | Moonlight shared-eta rule at muon_lr 0.017 (0.017/5.543); matches embedding update RMS to matrix update RMS |
| `norm_lr` | 0.0003 | Keller README gains/biases AdamW rate; 10x below tied-table rate |
| `qk_norm_lr` | 0.0001 | Conservative for logit-scaling Q/K gains; archived family practice |
| `weight_decay` | 0.1 | Universal production Muon/AdamW value (Moonlight/K2/GLM/nanoGPT); contrast owed |
| `embedding_weight_decay` | 0.1 | nanoGPT tied-table and Moonlight-code convention; 0.0 contrast owed |

## 4. Adaptive trial plan (8 allocated GPU-hours, one H100)

Every trial: 1,272 updates = 500,170,752 tokens; rank only endpoint CE.
Archived throughput 143.5k tokens/s -> ~58 min + startup/validation;
real runtime measured on trial 1 gates all later starts. Planned sequence
(later choices adapt to measured curves):

1. `researched_candidate_01`: 0.017 / 3e-3 / 0.1 / 0.1 (anchor above).
2. Muon LR high bracket: 0.034 (screen favored the higher direction).
3. Muon LR low bracket: 0.0085, if needed to bound the optimum.
4. Embedding AdamW contrast at best muon_lr: 3e-4 vs 3e-3.
5. Matrix wd contrast at best LRs: 0.02 (matched-LR; LR x wd accounting).
6. Embedding wd contrast: 0.0 vs 0.1.
7. Winner refinement / per-role rate (e.g. muon_down_lr) only if time fits a
   complete run.

Start a trial only if its measured full runtime fits the remaining cap;
otherwise rank, report, and request deallocation. Negative controls for the
later width bridge come from stable losers on opposite sides of the
matrix-LR optimum, so the bracket trials double as bridge material.

## 5. GPU staging protocol (when the human starts the H100)

VM: `H100_GROUP/h100` (Standard_NCC40ads_H100_v5, Central US), SSH
`azureuserboss` with `~/.ssh/id_rsa`. Record the allocation start time;
everything (startup, validation, idle) counts against 8 h.

1. Confirm power state (read-only):
   `az vm get-instance-view -g H100_GROUP -n h100 --query instanceView.statuses -o tsv`
2. Stage the exact controller manifest and verify:
   `scp /Users/varundaiya/terminus-optimizer-evidence/stack004-manifest.json azureuserboss@<vm>:/srv/terminus-10b/stack004-manifest.json`
   then on the VM: sha256sum must equal
   `3c70bef91d09b2c31194887191a8058899d199f3bb4ebbed41c8691b3b521575`.
   (If it is already staged with that hash, do not re-copy.)
3. Verify readable shards under `/srv/terminus-10b/data/packed` (count 418
   train/val .bin files, spot-check one sha256 against the manifest).
4. Sync this exact commit to `/home/azureuserboss/terminus-autoresearch` via
   `git bundle` (VM has no GitHub credential), checkout the branch commit.
5. Create `/srv/terminus-10b/hpo-results` (outside Git).
6. Run trials from the repo root with the pinned runtime env
   (PYTHONPATH=/home/azureuserboss/terminus-optimization-20260910/runtime/fa3-site,
   CUDA_HOME=/usr/local/cuda-13.2, DG_JIT_NVCC_COMPILER, DG_JIT_CACHE_DIR per
   the architecture-cycle config), remote python
   `/home/azureuserboss/terminus-results/runtime/venv/bin/python`:
   `python -m hpo run --trial <name> --updates 1272 --candidate hpo/candidate.json
     --manifest /srv/terminus-10b/stack004-manifest.json
     --data-root /srv/terminus-10b/data/packed --results-root /srv/terminus-10b/hpo-results`
   under nohup with a log; poll; on completion pull `result.json` / ledger
   rows and the events curve for analysis.
7. `python -m hpo rank --updates 1272 --results-root /srv/terminus-10b/hpo-results`
   after each trial; keep/revert; commit each new candidate before running.
8. Track allocated hours from VM start; before each new trial confirm a
   complete run fits; when it cannot, rank, pull all results to the
   controller, and ask the human to deallocate.

## 6. Width-transfer plan (700M/1.5B) — prepared, NOT run under this cap

Bridge views are prepared once, outside Git, from the verified manifest
(both keep the 393,216-token update and the same ordered examples):

- Width 1536 (700M): `python -m hpo.bridge --manifest <verified manifest>
  --width 1536 --microbatch 4 --output-dir /srv/terminus-10b/hpo-bridge`
- Width 2304 (1.5B): same with `--width 2304 --microbatch 2`

Then at each width, from the 200M winner candidate file:

- `python -m hpo.transfer --candidate winner_200m.json --width <w>
  --embedding-exponent {0, 0.5, 1} --output ...` — three matched-shrinkage
  tied-embedding AdamW hypotheses (LR scaled `(768/d)^alpha`, decay
  inversely, so per-step `LR x wd` is identical).
- Plus the closest credible 200M runner-up and two stable 200M losers on
  opposite sides of the matrix-LR optimum, all under the best-supported
  exponent, at matched 750/1,272-update milestones.
- Muon spectral and small-parameter rates stay at the 200M values initially;
  if the matrix-LR optimum moves with width, fit a role-specific width
  exponent instead of declaring transfer success. The 1.5B check also needs
  the production data mixture and a declared global-batch bridge before any
  production claim. Until those run under a separate GPU budget, the 200M
  result is an English proxy recipe only.
