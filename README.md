# Terminus Stack 4 hyperparameter autoresearch

This worktree prepares the dense HPO campaign on the completed E3/E4/E9
single-GPU speed implementation. The selected optimizer is still pending
the separate Hyperball comparison. Start with [the HPO README](hpo/README.md),
[the operating program](hpo/program.md), and [the Kimi prompt](hpo/KIMI_START.md).

The 200M English proxy loop and the width-bridge generator live in `hpo/`.
Every valid 200M candidate trains for 500,170,752 tokens under a total
eight-allocated-GPU-hour cap. Kimi begins with its own optimizer research from
[the factual model profile](hpo/MODEL_PROFILE.md), then chooses rates and
weight decay; the reference config's numbers are placeholders.
The source includes the completed speed implementation at `8a62293` and
the direct spectral-Muon update at `8153a78`; optional later numerical
speed candidates are disabled in the HPO reference. The architecture is
still the 25-layer Peri-LN plus learned-value-residual Stack 4 model.

The old [speed program](speed_program.md) and [architecture program](program.md)
remain for provenance and do not govern this HPO branch. The earlier five-hour
speed cap is unrelated to this HPO campaign. No token shards, checkpoints,
trial results, credentials, or generated evaluation records belong in Git.
