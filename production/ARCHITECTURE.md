# Reviewed production architecture change

On 2026-10-17 the operator selected 24 layers with five local layers followed
by one global layer, using a 512-token local window. This replaces the original
25-layer, four-local/one-global, 1024-window production baseline.

| Setting | Original baseline | Selected recipe |
| --- | ---: | ---: |
| Layers | 25 | 24 |
| Global layer numbers (one based) | 5, 10, 15, 20, 25 | 6, 12, 18, 24 |
| Local layers | 20 | 20 |
| Local attention window, including current token | 1024 | 512 |
| Sequence length and global context | 4096 | 4096 |
| Trainable parameters | 1,500,456,496 | 1,446,478,126 |

Only `n_layers`, `global_attention_every`, and `local_attention_window` change.
Width 2304, FFN 5760, 18 query/6 KV heads, 128 head dimension, tied 65,536-token
embedding, Peri-LN, learned Q/K gains and learned value residual are retained.
Residual initialization remains 0.14078571632817444. Spectral Muon and all
optimizer settings, including starting learning rates, are retained. The
operator chose to reuse these starting rates without a separate GPU sweep.

This removes one complete block and one full-context mixing stage. The first
global layer moves from layer 5 to layer 6. A 25-layer checkpoint cannot be
resumed or silently converted into this architecture; start a fresh run.

Valid causal attention pairs decrease 36.8%, block projection arithmetic 4%,
and parameter count 3.6%. The mixed FP8/BF16 tensor-core accounting in
`speed_research/physics.py` gives 7.8% less ideal service time per token
(excluding replay, memory traffic, pointwise operations and operating overhead).
This is not a throughput measurement or a quality equivalence claim.

Historical speed and HPO evidence still describes the original architecture.
The new architecture's sustained native H100 performance, long-run stability,
task quality and dense-to-QAT transfer remain unmeasured. CPU tests validate
configuration, exact parameter count and the existing configurable model path.
