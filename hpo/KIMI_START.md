# Paste this prompt to Kimi for the dense 200M HPO campaign

Work in `/Users/varundaiya/terminus-autoresearch-hpo` (branch
`hpo/autoresearch-v1`). Read `AGENTS.md`, `hpo/program.md`, and
`hpo/MODEL_PROFILE.md`. This is the fixed 25-layer Stack 4 dense model on
the selected E3/E4/E9 single-GPU speed implementation. The optimizer rule
for this campaign is **spectral_unclamped Muon**; keep it fixed. The printed
rates and decay in `hpo/reference_200m.json` are parseable placeholders,
not research conclusions. Do not run Hyperball, ternary QAT, architecture
changes, data changes, or a new speed campaign here.

The human supplies and starts one H100 NVL/PCIe GPU. Once it is available,
you are authorized to run consecutive HPO experiments without asking again,
up to **8 allocated GPU-hours total** including startup, validation, and
idle allocated time. The human controls VM provisioning and final shutdown;
request deallocation promptly when another complete experiment cannot fit.
Do not start a paid VM yourself. If the GPU is already allocated, avoid
leaving it idle while doing extended literature research.

First do your **own source-backed research** on actual production Muon and
AdamW practices: Moonlight/Kimi, MiniCPM, MiniCPM4, Qwen3, Gemma 3/4, and
other relevant first-party papers or released training implementations.
Extract effective group LRs, width/update scaling, matrix and embedding
weight decay, batch, warmup, schedule, and omissions. Distinguish the
published rule from what this model implements. Choose the initial Muon
LR, tied-embedding AdamW LR, matrix decay, and embedding decay yourself;
fill the four `null` fields in `hpo/candidate.json`. Make a compact source
note and a short reason for each candidate. You may tune packed QKV,
attention O, gate/up, down, input/Peri/final norms, Q/K gains, and learned
value mix separately. Weight decay acts every optimizer update; the WSD
learning-rate decay does not start until roughly 2B tokens.

The exact locked manifest exists on the controller at
`/Users/varundaiya/terminus-optimizer-evidence/stack004-manifest.json`.
Stage that exact file on the GPU host's persistent disk and verify its
SHA-256 is
`3c70bef91d09b2c31194887191a8058899d199f3bb4ebbed41c8691b3b521575`.
The prepared shards resolve under `/srv/terminus-10b/data/packed`. Do not
use the older architecture manifest. Put results and checkpoints on an
external persistent path such as `/srv/terminus-10b/hpo-results`, outside
Git. Confirm the manifest and a readable sample shard before paid training;
do not burn GPU time on a prolonged preflight or profiler run.

Run the Karpathy-style loop in `hpo/program.md`: state one hypothesis,
change and commit one candidate, run a uniquely named trial, inspect its
matched-token English held-out cross-entropy and stability, then keep or
revert and select the next experiment. Use the runner's exact syntax:

```bash
python -m hpo run --trial researched_candidate_01 --updates 1272 \
  --candidate hpo/candidate.json \
  --manifest /srv/terminus-10b/stack004-manifest.json \
  --data-root /srv/terminus-10b/data/packed \
  --results-root /srv/terminus-10b/hpo-results

python -m hpo rank --updates 1272 \
  --results-root /srv/terminus-10b/hpo-results
```

If the staged manifest uses another path, substitute its **verified** path.
Every valid 200M candidate must complete 1,272 updates = 500,170,752
tokens. Only the final endpoint ranks; earlier validations guide the next
hypothesis. Optimize held-out CE, then consider stability and throughput.
Test optimizer-group rates and weight decay with high-information adaptive
experiments, not a Cartesian sweep. Account for `LR × weight_decay` when
comparing decay. Before starting another trial, make sure its measured full
runtime fits the remaining allocated time. Keep the GPU running useful
trials while it is allocated; stop at the cap or when no complete trial fits.
Record source/config hashes, CE curves, tokens/s, incidents, and allocated
GPU time. Keep all shards, checkpoints, and trial results out of Git.

This eight-hour allocation is **only the 200M dense search**. The 700M and
1.5B width bridge needs its own GPU budget later. After selecting the 200M
winner, prepare the bridge plan using `python -m hpo.bridge` and
`python -m hpo.transfer`, but do not run those larger trials under this cap.
At each larger width, test the winner with tied-embedding AdamW LR exponents
0, 0.5, and 1, keeping per-step `LR × embedding_weight_decay` matched;
then test the close 200M runner-up and two stable 200M losers on opposite
sides of the matrix-LR optimum. Leave Muon spectral and small-parameter
rates constant initially. Report the 200M result as an English proxy recipe
until those transfer and production-batch checks are measured.
