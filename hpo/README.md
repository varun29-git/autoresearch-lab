# Stack 4 dense hyperparameter autoresearch

This completed campaign used the E3 packed learned Q/K path, E4 value mix, E9 batched Muon, and the separate `spectral_unclamped` Muon rule. The model remained the 25-layer Stack 4 architecture at 200M scale. Optional numerical speed candidates such as an FP8 vocabulary head were not enabled.

Each ranked candidate ran exactly 1,272 complete updates, **500,170,752 tokens**, against the fixed English stream. Only the endpoint result ranked candidates. The final selection commit `5e5f4ab` records 12 ranked trials and endpoint CE **3.0456** for the values in [`candidate.json`](candidate.json):

- spectral-Muon LR 0.017;
- tied-table AdamW LR 0.006;
- norm LR 0.0003 and Q/K-norm LR 0.0001;
- matrix weight decay 0.02;
- embedding weight decay 0.0.

This is a 200M English-proxy result, not a validated 1.5B optimizer recipe. No 700M or 1.5B width-transfer trial was completed. See [the consolidated evidence guide](../docs/RESULTS.md#4-hpo-result) for the bracket observations and limitations.

## Auditing and reproduction

[`reference_200m.json`](reference_200m.json) fixes the model, optimizer convention, schedule, campaign stages, and manifest SHA-256. Its original placeholder optimizer numbers are not the selected candidate; use `candidate.json` for the final values. [MODEL_PROFILE.md](MODEL_PROFILE.md) and [the optimizer research note](../notes/optimizer-research-20261010.md) record the starting assumptions and literature context.

The raw manifest, 415 English train shards, 3 validation shards, candidate result files, checkpoints, and complete ranked ledger remain external. The exact manifest SHA-256 is `3c70bef91d09b2c31194887191a8058899d199f3bb4ebbed41c8691b3b521575`. The architecture campaign manifest is not interchangeable with this HPO manifest. A complete 12-row results table is not checked in.

With an externally supplied compatible result set, ranking is local and does not start training:

```bash
python -m hpo rank --updates 1272 --results-root /external/hpo-results
```

`python -m hpo.bridge` prepares a width-specific reference from external inputs, and `python -m hpo.transfer` generates tied-embedding AdamW LR exponents 0, 0.5, and 1 at matched `LR × weight_decay`. Those generators do not establish transfer. [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) maps early HPO notes' unavailable source IDs to retained snapshots.

[`program.md`](program.md) and [`KIMI_START.md`](KIMI_START.md) are historical campaign prompts, not authorization for new paid work. All results, data, checkpoints, credentials, and infrastructure configuration remain outside Git.
