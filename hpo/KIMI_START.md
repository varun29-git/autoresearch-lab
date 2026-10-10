# Paste after the optimizer result and HPO budget are set

Work in `/Users/varundaiya/terminus-autoresearch-hpo`. Read `hpo/program.md`.
This worktree contains yesterday's selected E3/E4/E9 speed implementation
and a runnable 200M Stack 4 dense HPO loop. The separate optimizer
comparison has selected **<SPECTRAL_OR_HYPERBALL>**. Set that rule once in
`hpo/reference_200m.json` and, if Hyperball won, merge its tested optimizer
code and verify the existing local tests before the first GPU trial.

You are authorized to use the human-provided single H100 NVL for at most
**<TOTAL_H100_HOURS>** GPU-hours. The human controls VM provisioning. The
exact archived manifest is at **<MANIFEST_ON_VM>**, prepared shards at
`/srv/terminus-10b/data/packed`, and external results at **<RESULTS_ROOT>**.
Do not use the older architecture campaign manifest: this HPO runner checks
the exact archived Stack 4 manifest SHA-256.

Run the Karpathy-style loop in `hpo/program.md`: one candidate, one named
trial, inspect its matched-token English held-out cross-entropy and
throughput, then keep or revert and choose the next candidate. Use 125
updates only for stability, 750/1250 for initial ranking, and complete
the 2.5B-token schedule for only strong finalists if the authorized budget
allows. Start with selected Muon's LR, then tied embedding LR, norm/QK
gain rates, decay and momentum. Do not copy Moonshot's numerical LR as
the expected optimum of spectral Muon or Hyperball. Do not run QAT or
change architecture, data, scorer, or batch inside this 200M campaign.

Keep the GPU doing useful experiments while authorized. Do not ask between
trials. Stop when the evidence is informative or the total GPU-hour cap is
reached. Report CE curves at equal tokens, code/config hashes, tokens/s,
incidents, GPU-hours, and the remaining width/horizon uncertainty. A final
transfer claim additionally requires the width and production-batch
checks in `hpo/program.md`. At both larger widths, compare four 200M
recipes at matched post-warmup tokens: the winner, a close runner-up, and
two stable but clearly worse settings on opposite sides of the matrix-LR
optimum. A former loser winning at larger width means the 200M ranking
did not transfer. Do not call the English 200M winner a 1.5B production
recipe without these checks.
