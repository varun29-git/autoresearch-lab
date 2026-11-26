# Dense parallelization and operating procedure

Use the selected recipe in `recipe.py`. This is a proposed starting strategy and
an acceptance procedure, not evidence of a 32-GPU result. The operator still
needs to record the actual H100 SKU/memory, GPUs per node, NVLink pairs, NICs,
inter-node fabric, storage layout and chosen global batch outside Git.

## Initial strategy

Start with replicated data parallelism: DP=32, TP=PP=CP=1. Each rank retains the
whole model and optimizer. `terminus.train` already uses DDP gradient bucket
views and `no_sync()` on every accumulation microstep except the last. DDP
averages FP32 gradients; loss scaling by `world_size / update_tokens` supplies
the global token mean, including partial final updates and empty tail ranks.
Clipping happens after reduction, then spectral Muon/auxiliary AdamW runs on
each rank. Preserve logical Q/K/V and gate/up matrix slices, tied embedding
updates and cache invalidation after the optimizer step.

Microbatch 4 at context 4096 is the first memory candidate. The original 25-layer research run
allocated 55.5 GB at that batch; production NCCL buffers, allocator reservations,
runtime workspaces and optimizer temporaries still need measurement. Microbatch
8 allocated 87.4 GB and is not a suitable default for an 80 GB H100. NVLink
between a pair of NVL cards does not pool their memory for this DDP trainer.

For microbatch `m`, accumulation `a` and world size `W`, the global update is
`B = W × m × 4096 × a` valid tokens. At W=32 and m=4:

| Global batch example | Local accumulation | Tokens/rank/update |
| --- | ---: | ---: |
| 2,097,152 | 4 | 65,536 |
| 4,194,304 | 8 | 131,072 |

These are examples, not newly selected hyperparameters. Preserve the reviewed
global batch when changing microbatch or world size. Do not compare these local
profiles with the archived mb4 × accum32 rate as if the workloads were equal.

The selected 24-layer model's 1,446,478,126 FP32 gradients occupy 5,785,912,504 bytes/rank. A ring model
moves `2 × (W−1)/W × gradient_bytes`, or 11.21 GB/rank/update at W=32. At an
effective *per-rank* 50 GB/s this is 224 ms of service before latency/contention,
with an unmeasured exposed portion. A node's NIC bandwidth is shared and cannot
be assigned in full to every rank. NCCL may choose another algorithm. Measure
bucket readiness and the last reduction's completion on the actual topology;
DDP can overlap reductions only with work after each bucket becomes ready.

## Data and startup

Stage frozen shards and tokenizer on durable storage or node-local NVMe, with
identical manifest-relative paths and contents on every node. The current
loader gives ranks disjoint sequence indices from the same deterministic
global stream and supports a saved microstep cursor. It expects all manifest
files to be accessible; supplying different rank-local manifests fails the
cross-rank contract check. An ownership-aware shard cache is a separate change.

There is a scale blocker to resolve before a full-corpus launch:
`validate_manifest` hashes every shard and scans all token IDs independently
on every rank, after distributed initialization. For a 3 TB corpus, 32 ranks
logically inspect about 96 TB per pass, even if filesystem caching avoids some
physical reads. This also risks exceeding the five-minute collective timeout
when ranks finish at different times. Do not disable integrity checking as a
workaround. Design and test immutable, content-addressed staging receipts,
one full verification per node/staged snapshot, bounded startup identity
checks, and distributed error propagation before changing this path. Define
how replacing or corrupting a staged shard invalidates its receipt.

Run the CPU inspector before GPU allocation:

```bash
python -m production.inspect_run \
  --config /external/production/dense_config.json \
  --manifest /external/production/manifest.json \
  --world-size 32 \
  --expected-global-batch-tokens <reviewed_global_batch> \
  --output /external/production/offline_inspection.json
```

Adding `--data-root /external/production/data` reads and validates the full
dataset on CPU. Without it, the report explicitly leaves data unchecked. This
report is not the CUDA preflight report consumed by the trainer. It also does
not bind the generic launcher to the inspected world size: the operator must
check `NNODES × NPROC_PER_NODE` and the startup contract against the plan.

## Cluster acceptance sequence

Only the human launches or stops paid GPU work. Keep configurations, identities,
traces and measurement receipts outside Git.

1. Freeze the runtime image: Python, PyTorch, CUDA runtime **and JIT compiler**,
   driver, NCCL, FA3, DeepGEMM, Triton and pinned Muon. Record native source/binary
   hashes. `pyproject.toml` is not a complete reproducible CUDA image; the old
   corpus-prep bootstrap is not a production trainer bootstrap.
2. Inventory `nvidia-smi topo -m`, GPU memory/power limits, NIC affinity and
   storage. Check point-to-point and collective bandwidth at the actual gradient
   payload. Do not assume an SXM/NVSwitch topology for NVL/PCIe hardware.
3. Run the native preflight on the frozen runtime, then a real full-size dense
   update with the selected spectral optimizer. The existing preflight certifies
   kernels and a reduced model; it does not validate a complete production
   optimizer trajectory. Compare losses, gradients, optimizer state and updated
   weights against a matched single-rank global update with declared tolerances.
4. Progress through within-pair, within-node, cross-node and all-32-rank runs.
   Keep the global update and stream matched for numerical comparisons. For
   performance comparisons, state both global and per-rank token counts.
5. Warm JIT/cache/optimizer state, then time complete updates on every rank.
   Report total valid tokens divided by elapsed time, slowest-rank latency,
   p50/p95, peak allocated **and reserved** memory, data waits, communication
   exposure and checkpoint/evaluation time. Collect detailed profiles in
   separate windows; the current rank-zero training log is not a per-rank trace.
6. Exercise scheduled checkpoint plus validation, graceful stop, abrupt worker
   loss and recovery from the last committed receipt. Compare resumed model,
   optimizer, RNG and next-data cursor with uninterrupted execution. The current
   checkpoint transaction is synchronous on rank zero and all ranks wait;
   measure its latency and storage pressure. Choose retention, remote backup,
   free-space alerts, lost-work allowance and operator escalation thresholds.
7. Soak the chosen topology/batch long enough to include repeated operating
   cadences. Account for setup, idle allocation, validation, checkpoints,
   restarts and the separate QAT phase in the 6,000 aggregate GPU-hour forecast.

Use `scripts/train.sh` with reviewed CONFIG, MANIFEST, DATA_ROOT, OUTPUT_DIR,
PREFLIGHT_REPORT and NPROC_PER_NODE. For multiple nodes, also set NNODES,
NODE_RANK, MASTER_ADDR and MASTER_PORT consistently. Use the same source and
runtime on every rank. Exact resume fixes world size, data, model, optimizer,
schedule, source and runtime identity. Elastic world-size changes and arbitrary
code changes during recovery are not supported.

## Escalate only after measuring

| Measured limitation | Candidate response | Required additional review |
| --- | --- | --- |
| Replicated state prevents a useful microbatch | ZeRO-1 style optimizer ownership | Keep each logical Muon matrix intact, synchronize updates, handle tied weights and sharded checkpoint recovery; no adapter exists today |
| Exposed all-reduce dominates | Tune DDP buckets/overlap and topology placement | The trainer currently uses PyTorch's bucket-size default; compare full updates and numerical behavior |
| Activation memory dominates | Selective activation recomputation | Validate custom FP8 caches/autograd and measure extra work before adopting |
| A rank cannot hold the model after simpler options | TP/PP/FSDP design | Custom packed layers, spectral scaling, value residual, tied head and checkpoint contracts require explicit adaptation |

Do not install a generic distributed Muon or flatten/shard its matrices and
assume equivalence: the Newton–Schulz direction and spectral aspect ratio are
defined on each complete logical matrix. Do not change gradient reduction to
BF16, enable an FP8 head, or alter wgrad quantization as an execution-only fix.
Those numerical changes need comparison against the selected baseline.

## Other production acceptance work

Beyond data preparation and parallelization, require a clean-machine runtime
rebuild; full-size spectral-Muon stability and multilingual/domain evaluation;
held-out task/version/contamination contracts; checkpoint retention and recovery
operations; and a separately reviewed dense-to-QAT recipe, transfer-quality
measurement, deployment/export checks and time allocation. The existing
transition and evaluation modules are useful building blocks, not evidence that
these production gates have passed. Hyperparameter selection need not be reopened
to specify these operational and quality gates.
