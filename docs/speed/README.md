# Stack-004 speed research

The operating instructions are in [speed_program.md](../../speed_program.md). Kimi's branch is `speed/stack-004-20261008`, based on the winning architecture commit `8bd552314470586f589d94bdaa6d7e8bc3c0c101`.

The user has one research H100 for a **three cumulative GPU-hour** optimization run. Production later uses 32 H100s and 6,000 aggregate GPU-hours including QAT. After one explicit authorization, Kimi starts the VM and runs a continuous series of experiments, making its own choices until the objective is credibly achieved or the research cap is exhausted. It does not deallocate after every short comparison.

Prepare a private config outside Git using [the example](../../research/speed_session.example.json), with a results directory on a persistent disk. Then, once the human authorizes the paid campaign:

```bash
python -m research.speed_session open --config /external/speed-session.json
python -m research.speed_session status --config /external/speed-session.json
python -m research.speed_session stage --config /external/speed-session.json
python -m research.speed_session exec --config /external/speed-session.json -- nvidia-smi -L
python -m research.speed_session close --config /external/speed-session.json --reason budget
```

`open` starts the VM once and installs a VM-side cutoff timer; `status` includes prior allocations and live time; `stage` sends a committed candidate; `exec` runs a remote command. Kimi calls `scripts/s0_block.py` through `exec` using the staged 1.5B config and manifest, writing results outside Git. It can stage and test successive commits without restarting the VM. The cutoff timer aims to deallocate before the three-hour cap even if Kimi disconnects. Azure shutdown latency is variable, so the ledger is a conservative estimate, not an exact billing invoice. The older [S0 Azure worker](../../research/azure_speed_worker.py) is one-shot and must not drive this continuous session.

Keep small raw results on persistent storage and copy them off during the session. A restart costs boot, SSH, possible JIT, and restaging time; use it only when saved idle VM time exceeds that cost. The first screen uses a few complete optimizer updates with no full checkpoint or broad production preflight. [Measurement notes](MEASUREMENT.md) explain the timing; [physics](PHYSICS.md) gives the required rate and ceilings; [experiment ideas](EXPERIMENTS.md) are a menu, not a queue.

Local checks, which do not start a VM:

```bash
python3 -m unittest tests.test_speed_session tests.test_s0_block tests.test_speed_s0 tests.test_speed_research
```

No native H100 run or VM cutoff has been validated end to end in this checkout. The 1.5B target has 25 layers, width 2304, FFN 5760, 18 query and 6 KV heads, 4K context and 65,536 vocabulary entries: 1,500,456,496 trainable parameters.
