# Historical HPO campaign prompt

The 200M HPO campaign is complete. This file records its operating constraints and is not an instruction to launch new work. Current results and audit steps are in [`README.md`](README.md) and [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

The historical prompt fixed the 25-layer Stack 4 dense model, E3/E4/E9 implementation, English stream, and `spectral_unclamped` Muon rule. It prohibited architecture, data, Hyperball, QAT, and unrelated speed changes. Candidates could tune optimizer-group rates and decay through adaptive, complete 1,272-update trials. Only endpoint English held-out CE ranked candidates. Every trial, including startup and validation, counted against the campaign's eight-allocated-GPU-hour cap.

The prompt required independent source-backed optimizer research, one committed hypothesis per trial, source/config hashes, stability and throughput records, and external storage for shards, checkpoints, and results. The 700M/1.5B bridge was explicitly outside that budget and remained unrun.

Personal checkout paths, SSH and cloud details, and the original paid-run authorization have been removed. The locked manifest SHA-256 remains `3c70bef91d09b2c31194887191a8058899d199f3bb4ebbed41c8691b3b521575`; the manifest and result ledger must be supplied externally for reproduction. Any new GPU campaign requires a separate explicit human execution decision.
