# CI-1 production handoff rules

This standalone repository is the Terminus Stack 4 production handoff. Start with `production/README.md`. The selected dense model is 25 layers, width 2304, 65,536 vocabulary, Peri-LN plus learned value residual, and spectral Muon on the optimized FP8/DeepGEMM implementation. Preserve the model and optimizer semantics unless a change is explicitly reviewed against the selected baseline. Older `program.md`, `speed_program.md`, `hpo/program.md`, and their configs document completed or separate research campaigns; they do not govern this production branch.

The human launches and stops paid GPU campaigns. Local code, configuration generation, review, and CPU tests do not authorize starting a cloud VM or training sweep. No source corpora, token shards, evaluation records, checkpoints, credentials, or generated trial results belong in Git. Keep production manifests, data, run configs, reports, and checkpoints outside this repository. Do not claim 32-GPU throughput, token-budget feasibility, dense-to-QAT transfer, or production quality before measuring them on the intended H100 NVL/PCIe cluster.

Make small reviewable commits, run relevant tests, and push completed milestones. Document unrun native-GPU checks honestly. Never rewrite published history.
