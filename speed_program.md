# Kimi K3: fast stack-004 speed loop

The job is to speed up the **same** 25-layer, 1.5B-parameter stack-004 model so 32 H100s can train at least 1.5T, ideally 2T, tokens within 6,000 production GPU-hours including QAT. Research uses one separately budgeted H100. The human launches and stops paid campaigns.

Start from the winning stack code in this worktree. Keep the layers, width, context, vocabulary, Peri-LN, value residual, optimizer mathematics and unique training-token count. No model shrinking or skipped work. The [physics sheet](docs/speed/PHYSICS.md) gives the target rates; the [idea atlas](docs/speed/EXPERIMENTS.md) is a menu, not a required queue. You may invent better ideas when measured bottlenecks suggest them.

## The loop

1. On the CPU, inspect the last **complete optimizer-update** timing. Choose one promising change, state the reason in a few sentences, implement it, run focused correctness tests and commit it. Keep failed commits and append a short result to the external trial log.
2. On the authorized H100, run the same 1.5B shape and data path for reference and candidate. Warm each once or twice; time **2–5 complete updates** with CUDA synchronization at the block boundaries. Aim for tens of seconds of measured work. Record valid tokens, elapsed time, memory, phase, commit and cold setup time. A short profiler capture is optional if the result needs explanation.
3. If the gain is clear and large, keep investigating it. If the difference is small or noisy, rerun a short matched comparison. Spend longer on gradient/quality checks and sustained training **only for finalists**. Dense and QAT must both be checked before a production recommendation.
4. Archive the results, verify they are readable, and turn the research VM off before returning to CPU analysis. Do not leave it running while reading papers, writing code or deciding the next idea. Follow the human's VM procedure if the automatic worker has not been proven end to end.
5. Repeat until the human stops the campaign, its separate research budget is reached, or further experiments are not worth their cost.

The first throughput screen does **not** require six paired minute-long runs, a full checkpoint, or the broad native production preflight. The short benchmark deliberately bypasses that preflight; the ordinary production trainer still requires it. Never count a forward-only kernel speedup as training throughput. A CPU measurement is not H100 evidence.

You may prepare a few independent candidate implementations offline before one GPU session if startup cost makes that worthwhile. Do not code the entire idea list in advance: GPU findings choose the next batch. No cloud VM or sweep may be started from this file alone.

Record failures honestly. The simple default is one reference/candidate comparison and a short external log entry. The detailed campaign lock and statistical analyzer under `speed_research/` are optional tools for later confirmation, not gates for an initial speed trial. Before recommending production use, show numerical and quality parity, repeat the promising speed result, and measure 32-GPU scaling when those GPUs become available.
