# Terminus Stack 4 hyperparameter autoresearch

This worktree prepares the dense HPO campaign on the completed E3/E4/E9
single-GPU speed implementation. The selected optimizer is still pending
the separate Hyperball comparison. Start with [the HPO README](hpo/README.md),
[the operating program](hpo/program.md), and [the Kimi prompt](hpo/KIMI_START.md).

The 200M English proxy loop and the width-bridge generator live in `hpo/`.
The source includes the completed speed implementation at `8a62293` and
the direct spectral-Muon update at `8153a78`; optional later numerical
speed candidates are disabled in the HPO reference. The architecture is
still the 25-layer Peri-LN plus learned-value-residual Stack 4 model.

The old [speed program](speed_program.md) and [architecture program](program.md)
remain for provenance and do not govern this HPO branch. The earlier five-hour
speed cap does not authorize a new HPO campaign. No token shards, checkpoints,
trial results, credentials, or generated evaluation records belong in Git.
