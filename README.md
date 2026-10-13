# Terminus CI-1

This independent folder contains the full Terminus Stack 4 codebase, optimized dense training path, tests, research history, and [technical-paper draft](manuscript/terminus_draft.md). Use the [production handoff](production/README.md) to prepare a 1.5B dense run. The older architecture, speed, and HPO campaigns remain available for provenance; their run configurations are not production launch configurations.

The handoff does not start a paid GPU run or assert that the 32-GPU throughput target has been reached. Data manifests, token shards, checkpoints, credentials, and trial results remain outside Git.
