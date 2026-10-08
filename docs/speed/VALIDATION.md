# Local validation of the framework

Validated on 2026-10-08. These are repository/tool checks, not GPU experiment results.

- `python3 -m unittest tests.test_speed_research -v`: **26 tests passed**. Covers aggregate GPU-hour math, phase-weighted time, causal mask counts, proxy/target parameter counts, mixed-precision hardware bounds, incomplete/tampered contracts, forbidden architecture changes, candidate execution options, paired timing decisions, token/clock/billing manipulation, production memory, failure records, ledger corruption and timing boundaries.
- `python3 -m speed_research verify-anchor`: all **43** pinned candidate/`terminus/` source files matched their winning Git objects. This verifies the historical source, not the current planning checkout's model.
- `check-config` accepted the complete dense target fixture. Default and explicit dense/QAT/scaling/overhead planning scenarios executed locally.
- The **actual winning model source**, exported to an isolated temporary directory from `8bd552314470586f589d94bdaa6d7e8bc3c0c101`, was imported with local PyTorch 2.14.1 and instantiated on the **meta device** at the proposed target shape. It reported exactly **1,500,456,496 parameters, 25 layers, 50 Peri-LN output norms and 48 learned value-residual scalar parameters**. This agrees with the independent analytical count; it does not exercise numerical computation or allocate training weights.
- Local Markdown links and whitespace were checked. Existing model/training/campaign implementations were not edited. No generated training/evaluation records were added to Git.

The host has **no CUDA GPU available**. Native SM90 forward/backward/parity, full-update timer integration, real data/quality trials, memory/throughput profiling, compiled/graph paths, QAT transition/soak, checkpoint/resume and NCCL/32-GPU acceptance are **unrun**. No paid GPU work, VM launch or training sweep was performed. The first native task for Kimi is S0 as specified in [MEASUREMENT.md](MEASUREMENT.md).
