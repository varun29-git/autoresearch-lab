# Terminus autoresearch working rules

This is the independent, private Terminus architecture-research repository. It contains a clean snapshot of `varun29-git/terminus-modeltunnelv2` at `353ecc32e8c5d7bd1d1eb29a461962519c1553de`. Do not work in the older `terminus-flagship` checkout. Preserve the existing model and training implementation while adding the architecture-search harness.

The human authorizes paid GPU campaigns. Building or validating this repository locally does not authorize starting a cloud VM or training sweep. For the stack-004 speed campaign, one explicit authorization delegates VM start, deallocation, and restart to Kimi within its three GPU-hour research cap; do not ask again for each experiment. No source corpora, token shards, evaluation records, checkpoints, credentials, or generated trial results belong in Git.

For an active architecture campaign, follow `program.md`. Its editable surface is the model implementation and candidate architecture file. The campaign policy, scorer, training/evaluation harness, tokenizer, and data manifest are fixed during a campaign; a proposed change to those requires a new campaign version. Keep experimental commits and append-only receipts. Never rewrite published history.

Make small reviewable commits, run relevant tests, and push completed milestones. Document unrun native-GPU checks honestly.
