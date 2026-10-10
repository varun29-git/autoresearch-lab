# Stack 4 dense hyperparameter autoresearch

This is a separate HPO worktree based on the completed single-GPU speed source
`8a62293` and the spectral-Muon implementation `8153a78`. Its 200M reference
uses the selected E3 packed learned Q/K path (`fused_qkv=true`), E4 value mix,
and E9 batched Muon. Optional numerical speed candidates such as an FP8
vocabulary head are **not** enabled. The architecture remains 25-layer Stack 4.

The optimizer-selection result is pending. `reference_200m.json` currently
chooses spectral Muon with weight decay; if Hyperball wins, select that
optimizer implementation and change the reference **before** the first HPO
trial. The optimizer rule is fixed across an HPO campaign. Do not compare
HPO trials with different optimizer rules as if they were one LR sweep.

`python -m hpo run` executes one candidate against the fixed English stream,
and `python -m hpo rank` compares candidates at one matched-token milestone.
Candidates only override LRs, decay, momentum/betas/epsilon, or clipping.
All results and checkpoints go outside Git. Read [program.md](program.md)
before starting a paid run; [KIMI_START.md](KIMI_START.md) is the pasteable
agent prompt once the optimizer winner, GPU-hour cap, and data paths are known.

The exact archived Stack 4 manifest is staged on the controller at
`/Users/varundaiya/terminus-optimizer-evidence/stack004-manifest.json`.
Its 415 English train and 3 validation shard paths resolve under the Azure
persistent-disk directory `/srv/terminus-10b/data/packed`. Source shards are
in `s3://quark-research-repository-storage/data/modeltunnelv2_tokensets_20261001/packed/`.
The older `/srv/terminus-10b/architecture/manifest.en.json` is **not** the
same file as the locked HPO reference manifest.

No native H100 HPO run or width-transfer test has been performed in this
worktree. The 1.5B production optimizer recipe cannot be declared from the
200M English screen alone.
