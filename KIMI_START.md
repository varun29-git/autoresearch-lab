# Historical speed-campaign prompt

The dense-speed campaign described here is complete. This file is retained only to explain the original operating constraints; it is not a prompt to execute and does not authorize cloud or GPU work. Read [the speed archive](docs/speed/README.md), [results](docs/RESULTS.md), and [reproducibility limits](docs/REPRODUCIBILITY.md) instead.

The historical campaign optimized complete dense-update throughput for the 25-layer, 1.5B Stack 4 shape on one H100 NVL. It required exact token accounting, preserved gradients and optimizer updates, paired measurements, and matched-token quality gates for numerical changes. QAT, 32-GPU scaling, and production-budget work were outside its scope. Its external session ledger, private working paths, infrastructure identifiers, and paid-run authorization have been removed from the public document.

The archived protocol remains in `speed_program.md`; experiment status and measured evidence are indexed in `docs/speed/README.md`. Any new native run requires a separate explicit human execution decision and newly supplied external inputs.
