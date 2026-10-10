# Terminus autoresearch working rules

This is the independent, private Terminus architecture-research repository. It contains a clean snapshot of `varun29-git/terminus-modeltunnelv2` at `353ecc32e8c5d7bd1d1eb29a461962519c1553de`. Do not work in the older `terminus-flagship` checkout. Preserve the existing model and training implementation while adding the architecture-search harness.

The human launches and stops paid GPU campaigns. Building or validating this repository locally does not authorize starting a cloud VM or training sweep. This worktree's dense 200M HPO campaign follows `hpo/program.md` and has an eight-allocated-GPU-hour cap, separate from the completed speed campaign. Kimi must complete its own optimizer research; this campaign has selected spectral Muon. Once the human launches the campaign, Kimi can run consecutive HPO trials within that cap without repeated permission requests. No source corpora, token shards, evaluation records, checkpoints, credentials, or generated trial results belong in Git.

For an active architecture campaign, follow `program.md`. Its editable surface is the model implementation and candidate architecture file. The campaign policy, scorer, training/evaluation harness, tokenizer, and data manifest are fixed during a campaign; a proposed change to those requires a new campaign version. Keep experimental commits and append-only receipts. Never rewrite published history.

Make small reviewable commits, run relevant tests, and push completed milestones. Document unrun native-GPU checks honestly. Do not change the fixed HPO reference, prepared stream, scorer, model, or source code after the first HPO trial under one campaign ID; start a new campaign ID for a changed contract.
