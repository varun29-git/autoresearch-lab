# Dense round 2026-10-09: E3 kept, E4/E9 kept, E2 rejected (post-GPU)

Reference instrumented baseline `5bc1f2d`/framework `25c24b2`; measurements on the research H100 NVL at the locked 524,288-token update, dense phase, real staged English stream.

## Measured outcomes (tokens/s, peak torch-allocated)

| Arm | tok/s | s/update | peak GB | decision |
| --- | ---: | ---: | ---: | --- |
| S0 reference mb2 x accum64 (5 upd) | 35,720 | 14.68 | 40.1 | baseline |
| mb4 x accum32 (4 upd) | 36,788 | 14.25 | 56.8 | best in-envelope profile |
| mb8 x accum16 (4 upd) | 37,856 | 13.85 | 90.0 | exceeds 74 GB envelope |
| incumbent 20-upd confirm mb4 | 36,527 | 14.35 | 56.8 | reference |
| E3 fused_qkv+gain grads (6 upd) | 41,441 | 12.65 | 54.3 | keep |
| E3 20-upd confirm | 40,187 | 13.05 | 54.3 | confirmed +10.0% |
| E2 residual-norm fusion + E3 (8 upd) | 40,947 | 12.80 | 54.2 | reject (gain-reduction re-read cancels fusion) |
| E9 batched Muon + E3 (8 upd) | 40,752 | 12.87 | 54.3 | keep (wash here; matters at 32-GPU local accumulation) |
| E3+E9+E4 value-mix (8 upd) | 41,936 | 12.50 | 55.5 | keep |
| quant_triton probe | 35,538 | — | 40.1 | reject (wash) |
| fp8_wgrad_splits 4 / 8 | 26,613 / 25,237 | — | 40.1 | reject (large regression) |
| cuda_graphs probe mb2 | 33,243 | — | 24.4 | reject (not launch-bound; replay overhead) |
| loss_chunk 2048 / 8192 | 41,345 / 40,377 | — | — | reject (keep 4096) |

## Numerical gates (all native H100)

- E3: packed QK norm+RoPE with learned gains matches eager reference (fwd exact, dx exact, gain grads within 2e-2/2e-4 after shared-channel reduction); prior no-gain packed test unchanged.
- E9: batched Muon step bitwise identical to the reference loop over 3 updates (params and optimizer state); CPU: bitwise vs archived separate-matrix Muon.
- E4: fused value-mix bitwise vs eager including the aten scalar-to-BF16 coefficient rounding (discovered: 0-dim FP32 coefficient casts to BF16 before multiply).
- E2: unit equivalence passed; rejected on speed, not numerics.

## Attribution after E3 (per microstep, M=16384, event-profile; ideal from roofline)

wgrad 67 ms (ideal 26.5), fwd+dgrad GEMM 100 ms (ideal 53), attention 48 ms (17.8), FP8 quantize 41 ms (~20), loss head 31 ms (17.8), packed-QKV 21 ms, norms 17 ms. DeepGEMM wgrad at ~2.5x its ideal service time is the largest single unresolved cost; the split-K Triton variant is strictly worse. Attention backward (FA3, local-window majority) runs ~3x its forward. Host/launch overhead is negligible (CUDA graphs regressed; data next() 0.7 ms/update; optimizer 0.28 s/update at accum 32).

## 32-GPU note

At production-local accumulation the per-update fixed costs (Muon 0.28 s, clip, FP8 weight re-cache) amortize over 32x fewer tokens per rank; E9's batched Newton-Schulz exists for that profile even though it is a wash at accum 32 on one GPU.
