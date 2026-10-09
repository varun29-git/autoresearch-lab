# 32-GPU transfer model for the dense stack-004 update (2026-10-09)

Built from measured single-GPU values on the research H100 NVL (see
trials/e2-e3-e4-e9-dense-round.md). Projections are scenarios, not measured
32-GPU performance. Production hardware/topology and global batch are owner
inputs; the missing values are named at the end.

## Measured single-GPU anchors (dense, real stream)

- Best in-envelope update: E3+E9+E4 at mb4 x accum32, 524,288 tokens/update:
  12.50 s/update (41,936 tok/s), 54.3 GB torch peak. Pareto alternative outside
  the 74 GB envelope: mb8 x accum16, 12.34 s/update (42,471 tok/s), 87.4 GB.
- Phase split per update (mb4, accum32): forward 5.09 s, backward 8.66 s,
  optimizer 0.28 s (0.10-0.15 s with E9 batching), clip 0.01 s, data ~0.
- Per microstep (M=16384): fwd 159 ms, bwd 271 ms. GPU-side op profile:
  wgrad 67 ms, fwd+dgrad GEMM 100 ms, attention 48 ms, FP8 quantize 41 ms,
  loss head 31 ms, norms/packed-QKV/value-mix ~45 ms.
- Persistent training state ~18.6 GB (FP32 params/grads/Muon momentum/Adam);
  activation peak drives the 54.3 GB at mb4.

## Production per-rank work

With global batch B tokens/update and G=32 ranks (DDP, fixed global batch),
per-rank tokens = B/32. The measured per-token cost at M=16384 (mb4) holds for
per-rank microbatches of 4 sequences; smaller per-rank microbatches lose GEMM
efficiency (mb2 measured 3% slower per token).

| Global batch B | per-rank tokens | per-rank microsteps (mb4) | compute/rank | optimizer/rank (E9) | fixed-cost share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 524,288 | 16,384 | 1 | 0.391 s | ~0.12 s | 24% |
| 2,097,152 | 65,536 | 4 | 1.56 s | ~0.12 s | 7.4% |
| 4,194,304 | 131,072 | 8 | 3.13 s | ~0.12 s | 3.8% |

This is why E9 is kept despite being a wash in the research profile: at low
per-rank accumulation the unbatched Muon loop (0.28 s) would dominate overhead.

## Communication

FP32 gradients, 1.5005 B params: ring all-reduce moves
2(G-1)/G x 4P = 11.63 GB per rank per update (BF16 gradients would halve bytes
but change the numerical contract - not assumed).

Gradient readiness (measured structure, not yet a bucket trace): backward runs
8.66 s per 32-microstep update in reverse layer order; the tied
embedding/classifier gradient and the layer-0 V1 fan-in complete only at the
end of backward. Bucketed overlap can hide early-layer buckets under the rest
of backward; the head/embedding bucket and layer-0/V1 buckets form an exposed
tail of at most (embedding wgrad ~31 ms + first-layer times) per microstep
chain - with accumulation, readiness repeats per microstep, so only the final
microstep's tail is exposed per update.

Efficiency scenarios (t_comm = 11.63 GB / effective per-rank bandwidth;
exposed = t_comm x (1 - overlap)):

| effective BW/rank | t_comm | overlap 70% | overlap 90% |
| ---: | ---: | ---: | ---: |
| 50 GB/s (poor cross-node) | 233 ms | eta .94 at B=2M | eta .98 |
| 100 GB/s | 116 ms | eta .97 | eta .99 |
| 200 GB/s | 58 ms | eta .985 | eta .995 |
| 350 GB/s (NVLink-class) | 33 ms | eta .99 | ~1.0 |

(eta at B=2,097,152: compute 1.56 s + exposed comm + 0.12 s optimizer; lower
overlap or smaller B degrades it; the table assumes the optimizer overlaps
nothing and adds fully.)

## What must be measured on 32 GPUs later (S5 acceptance)

1. Hardware: exact SKU (SXM/NVL), power cap, NVLink/NVSwitch domain size, NIC
   count/effective per-rank cross-node bandwidth, driver/NCCL versions.
2. NCCL all-reduce service time for the real 11.63 GB payload at G=32, and the
   bucketed readiness/overlap trace against the backward timeline (validate the
   70-90% overlap assumption; measure the head/V1 tail).
3. Per-rank sustained rate at the chosen B with real data sharding, logging,
   evaluation and checkpoint cadence; slowest-rank wall time, not the mean.
4. One known-update numerical equivalence vs the single-GPU contract
   (gradients bitwise-comparable before reduction order), resume correctness.
5. QAT phase rates under the same measurement discipline before any production
   token claim; dense rates here do not cover the QAT fraction.

## Missing owner inputs

production_hardware, production_topology, production global batch B, QAT
token fraction, production checkpoint cadence. The campaign lock stays a draft
until these arrive; today's screens used the locked research update profile
(524,288 tokens) with label `production-local mapping: B/32 at mb4`.
