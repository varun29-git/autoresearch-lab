# Stack-004 speed research

Start with [speed_program.md](../../speed_program.md). Kimi's working branch is `speed/stack-004-20261008`, based on winning architecture commit `8bd552314470586f589d94bdaa6d7e8bc3c0c101`. The architecture campaign in `program.md` is complete. The user has one research H100 and will later use 32 H100s for production; the 6,000 production GPU-hours include QAT and exclude research.

The loop is **CPU idea and code → short H100 comparison → save results and deallocate → CPU analysis**. A first screen needs a warmup and 2–5 full optimizer updates per arm, not six long pairs. Longer numerical, quality and sustained speed checks belong to candidates that show a real gain. VM boot, kernel compilation and archive time are separately billed overhead, so a short measured block is not a guarantee that the whole VM cycle lasts one minute.

The current [S0 timer](../../scripts/s0_block.py) measures complete updates from the real trainer. It can omit the broad production preflight and the large checkpoint for quick research timing. The [S0 worker](../../research/azure_speed_worker.py) requests deallocation after its archive is verified. **That worker has not been validated on an H100 end to end; do not assume automatic shutdown until the human's controller confirms it.** The human launches/stops paid campaigns. If the worker is not connected, the human must deallocate the VM after the results are safely copied.

[Physics](PHYSICS.md) gives the required throughput and optimistic hardware limit. [Experiment ideas](EXPERIMENTS.md) explain possible levers and falsifying tests; Kimi should profile first and is free to pursue other ideas. [Measurement notes](MEASUREMENT.md) explain the short benchmark and what a later production claim needs. [Sources](SOURCES.md) records external evidence. All model/training records and GPU traces stay outside Git.

Offline checks from this repository:

```bash
python3 -m speed_research plan
python3 -m unittest tests.test_s0_block tests.test_speed_s0 tests.test_speed_research
```

For a directly launched research GPU, `scripts/s0_block.py --help` shows the short benchmark inputs: model config, existing staged manifest/data root, output paths, warmup and measured update counts. The Azure S0 worker currently accepts the older campaign-lock layout and uses the same short block. Assemble those inputs on the CPU before turning on the GPU. Use the detailed lock/analyzer only when a later comparison needs that level of audit.

The proposed target has 25 layers, width 2304, FFN 5760, 18 query and 6 KV heads, 4K context and 65,536 vocabulary entries: 1,500,456,496 trainable parameters. The archived 200M architecture win is evidence for the stack, not a 1.5B training speed measurement. No native H100 speed result is claimed by this documentation.
