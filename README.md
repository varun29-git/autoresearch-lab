# Terminus autoresearch

This standalone Terminus research repository adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to the maintained Terminus ModelTunnelv2 code. It combines the completed architecture campaign with the subsequent dense speed campaign; `program.md` governs architecture research and `speed_program.md` governs kernel and execution research.

**Architecture campaign `architecture-20261006c` — COMPLETE (2026-10-06 → 2026-10-07).** The stack architecture (Peri-LN branch output norms + ResFormer value residual) beat the baseline by −1.15% at a matched 2.5B-token dense-FP8 horizon and by −0.55% after ternary-weight conversion + 500M-token QAT, while bounding the residual stream the baseline lets migrate. Full evidence: [notes/FINAL_REPORT_ARCHITECTURE_20261006.md](notes/FINAL_REPORT_ARCHITECTURE_20261006.md) and [notes/qat500-probe-comparison.md](notes/qat500-probe-comparison.md).

**Dense speed research** starts from the selected stack with E3 packed Q/K, E4 fused value mix and E9 batched Muon. The model remains 25 layers and 1,500,456,496 parameters. The measured E3/E4/E9 results and later experiments are documented under [docs/speed](docs/speed/README.md).

- [KIMI_START.md](KIMI_START.md) is the copyable one-time paid-run prompt.
- [PENDING_5H.md](docs/speed/PENDING_5H.md) lists every unfinished experiment, its status and the first falsifiable test.
- [EXPERIMENTS.md](docs/speed/EXPERIMENTS.md) is the original hypothesis atlas; [PHYSICS.md](docs/speed/PHYSICS.md) derives the one-GPU hardware bounds.
- [MEASUREMENT.md](docs/speed/MEASUREMENT.md) defines complete-update timing. [TRANSFER_32GPU.md](docs/speed/TRANSFER_32GPU.md) is archived production context, outside this loop.
- [Speed README](docs/speed/README.md) contains the controller commands and external config location.

The new research cap is **five cumulative H100 VM-on hours**, enforced by `research.speed_session` with a fresh external ledger. Kimi's sole objective here is to maximize validated dense full-update throughput on **one H100 NVL**. The previous best sustained dense log reported 41,075 valid tokens/s. Kimi must attack the largest exposed update costs, compare with single-GPU physical bounds and report the remaining one-GPU bottlenecks. QAT and 32-GPU production planning are outside this campaign. This folder's preparation did not start a VM or paid training.

The original architecture research design and source baseline remain at [SOURCE_BASELINE.md](SOURCE_BASELINE.md) and [docs/AUTORESEARCH_DESIGN.md](docs/AUTORESEARCH_DESIGN.md). No corpora, token shards, evaluation records, checkpoints, credentials or generated speed-trial results belong in Git.
