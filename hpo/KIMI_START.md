# Paste after the optimizer result is set

Work in `/Users/varundaiya/terminus-autoresearch-hpo`. Read `hpo/program.md`.
This worktree contains yesterday's selected E3/E4/E9 speed implementation
and a runnable 200M Stack 4 dense HPO loop. The separate optimizer
comparison has selected **<SPECTRAL_OR_HYPERBALL>**. Set that rule once in
`hpo/reference_200m.json` and, if Hyperball won, merge its tested optimizer
code and verify the existing local tests before the first GPU trial.

You are authorized to use the human-provided single H100 NVL for at most
**8 allocated GPU-hours total** for this 200M dense HPO campaign, counting
startup, validation, and idle allocated time. The human controls VM
provisioning. The exact archived manifest is at **<MANIFEST_ON_VM>**, prepared shards at
`/srv/terminus-10b/data/packed`, and external results at **<RESULTS_ROOT>**.
Do not use the older architecture campaign manifest: this HPO runner checks
the exact archived Stack 4 manifest SHA-256.

Before GPU allocation, read the factual `hpo/MODEL_PROFILE.md` and do your
**own deep research** using first-party papers and released training code:
Moonlight/Kimi, MiniCPM, MiniCPM4, Qwen3, Gemma 3/4, and other relevant
implementations you find. Determine which published rates and weight decay
apply to which optimizer groups and how their scaling, batch, and schedule
differ from ours. Decide the initial Muon LR, tied-table AdamW LR, and matrix
and embedding weight decay yourself. Record sources, calculations, and why
you chose the first candidates; do not treat reference-file values as tuned.
Weight decay acts on every optimizer update during these 500M-token trials;
the later WSD learning-rate decay is a different mechanism. Include matched
comparisons of matrix and tied-embedding weight decay in the research loop
when time permits, and report plainly if either remains untested.

Run the Karpathy-style loop in `hpo/program.md`: one candidate, one named
trial, inspect its matched-token English held-out cross-entropy and
throughput, then keep or revert and choose the next candidate. **Every valid
200M candidate runs 1,272 updates (500,170,752 tokens); rank only there.**
Intermediate validations inform the next hypothesis, not a shortened trial.
Work the actual optimizer groups: packed QKV, O, gate/up, down, the tied
embedding/readout Adam group, input and Peri-LN norms, final norm, Q/K gains,
and learned value mix. Use the eight-hour cap to choose high-information
experiments adaptively, not a Cartesian sweep. Investigate rates and decay
rather than inheriting a published or placeholder number. Do not run QAT or
change architecture, data, scorer, or batch inside this 200M campaign.

Keep the GPU doing useful experiments while authorized. Do not ask between
trials. Before starting another configuration, confirm that a full 500M-token
run fits the remaining allocated GPU time based on measured runtime; then
deallocate when it does not. Report CE curves at equal tokens, code/config
hashes, tokens/s, incidents, allocated GPU-hours, and the remaining
width/horizon uncertainty. A final transfer claim additionally requires the width and production-batch
checks in `hpo/program.md`. At both larger widths, compare four 200M
recipes at matched post-warmup tokens: the winner, a close runner-up, and
two stable but clearly worse settings on opposite sides of the matrix-LR
optimum. A former loser winning at larger width means the 200M ranking
did not transfer. Do not call the English 200M winner a 1.5B production
recipe without these checks.
