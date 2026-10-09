# Experiment atlas: preserve the model, change its execution

These are **hypotheses**, not a required queue or a reason to keep the H100 running while reading. The current five-hour run is **dense-only**; QAT-specific ideas below are reserved for a later campaign. Start with the short full-update baseline in [speed_program.md](../../speed_program.md). Class E preserves the mathematical computation/optimizer and the frozen precision recipe; class N changes numerical representation and needs a later quality comparison. Sources are in [SOURCES.md](SOURCES.md); all repository observations below refer to the winner commit.

## S0 — Establish the cost map and prove the instruments

**First question:** which part of the full update explains the 800B projection? Time a few real updates at the target width/depth/context and record memory. Use one short profiler capture only if the bottleneck is unclear. More detailed tensor, kernel and input-path analysis should follow a measured question. The old 200M speed does not predict the 1.5B speed.

## E1 — Tensor ownership and selective rematerialization

**Observation:** the winner stores `attn.values_used`, keeps layer 0 uncheckpointed for the V₁ gradient, and retains packed MLP activations for backward. The proxy peak grew by ~11 GiB. **Hypothesis:** unnecessary ownership or broad checkpoint regions force a poor batch/memory tradeoff at width 2304.

Audit saved tensors, aliases, module attributes and cache lifetimes by storage ID. Distinguish the single live V₁ source from 24 mixed tensors and any stale graph from a prior update. Make V₁ an explicit autograd input/output internally while preserving the monitored public tensor output. Remove ownership only after the last consumer; an attribute assignment is not inherently a leak. Preserve gradient fan-in to layer-0 V projection.

Measure a small Pareto grid: microbatch 1/2/4/... until the production peak limit; no checkpointing, selected pointwise recompute, selected attention/MLP recompute. Keep global update tokens and token ordering unchanged. Choose the **fastest complete update**, not the largest microbatch. Consider selective layer-0 checkpointing only after explicit source plumbing proves recomputation correct; its current exclusion is a correctness safeguard.

**Falsifier:** freeing references does not reduce live storage, or more recomputation costs more time than the better batch saves. **Tests:** two consecutive updates, accumulation >1, repeated forward before backward, checkpoint recompute, monitoring on/off, inference cache if touched, layer-0 gradient parity and restored state. **Ceiling:** memory savings alone count as zero speed gain until an execution change realizes them.

## E2 — Fuse Peri-LN, residual add and the following normalization

The preserved equations are `a = x + g_o ⊙ u / sqrt(mean(u²)+eps)` and `z = g_i ⊙ a / sqrt(mean(a²)+eps)`. A naive path writes/reads normalized branch output and residual several times. A fused kernel can load `x,u`, compute the first reduction, form `a`, compute its reduction, and emit the residual plus the next normalized operand; save minimal statistics or cheaply recompute them. FP8 packing of `z` can be an additional epilogue if it follows the exact existing scale/block rules.

The derivative of RMSNorm must retain both its direct term and the projection term:

`dx = r·(dy⊙g) − x·r³·mean((dy⊙g)⊙x)`; `dg = sum_rows(dy⊙x·r)`.

Preserve the reference BF16 cast after normalization and residual addition; fusing arithmetic can silently remove a rounding boundary. The Peri-LN gain is channel-wise, not scalar. Do not move the norm before the linear or into its weights: the input-dependent denominator makes that non-equivalent. At attention/FFN boundaries, verify which normalized output is actually used and which residual is needed by later backward.

**Cheap experiment:** isolated forward/backward at M×2304 with adversarial magnitude and gain vectors, then one whole block. **Signals:** fewer HBM passes and launches, equal per-gain gradients; check occupancy and scratch for gain reductions. **Falsifier:** spills/low occupancy or reduction scratch erase the saved traffic. Large fused GEMM+norm is a separate E16 experiment; row RMS needs all output-channel tiles and cannot be wished into a local epilogue.

## E3 — Fuse learned Q/K normalization, RoPE and layout conversion

**Observation:** `fused_qkv` is rejected when learned Q/K gains are enabled; the existing packed kernel lacks the gain gradients. The winner therefore uses separate learned RMSNorm and rotary paths despite a packed QKV projection. **Hypothesis:** a correct learned-gain packed kernel removes intermediate reads/copies and kernel launches without removing those gains.

For each head, `q' = RoPE(g_q⊙q / rms(q))` and similarly for k. Use the exact 128-dimensional reduction, epsilon, gain sharing across all heads/rows, rotation convention and scale. Values bypass these operations and feed E4. Emit layouts consumed directly by FA3, including noncontiguous views where supported. Backward applies inverse rotation, RMS derivative and reduction into the shared 128-element gain vectors, then packs dQ/dK/dV into the projection gradient layout.

**Test:** learned nonunit gains, nonzero positions, long/local boundaries, noncontiguous packed strides, Q≠KV heads, arbitrary upstream gradients. Compare every gain and Q/K weight gradient, not only forward outputs. **Falsifier:** it only wins against an unfused toy while current input/FA3 kernels already consume the views efficiently; reduction overhead consumes the gain. Never enable the old boolean without implementing and validating the missing derivative.

## E4 — Value residual as an IO and backward-reduction problem

At layer `l≥1`, `V'_l = a_l V_0 + b_l V_l`; `V_0` is raw layer-0 values before attention. Keep both free scalar coefficients; no sigmoid, normalization or clamping. Gradients include:

`dV_l = b_l dV'_l`, `dV_0 += a_l dV'_l`,
`da_l = <dV'_l,V_0>`, `db_l = <dV'_l,V_l>`.

**Near-term E:** one fused value-mix/layout kernel with explicit backward reductions, combining the mix with E3's otherwise required packed-QKV read. Eliminate needless contiguous copies; preserve BF16 multiplication/add rounding unless declared in the numerical budget. Stage reductions rather than large nondeterministic atomics.

**Higher-risk E:** fuse mixed-V formation into attention's V-tile loads so `V'_l` need not be written to HBM. The backward kernel must return source and local gradients plus both coefficient gradients, with correct GQA reduction. It may need to reread V₀/Vl for `da/db`; include that traffic. Consume V₀ read-only across layers, use a stable accumulation plan for its fan-in, and prove layer-0 backward is ready only after all consumers.

Attention linearity gives `A(aV₀+bVl)=aAV₀+bAVl`, but doing **two PV GEMMs** is not free and would usually lose. It is a useful analytical/reference identity, not the default optimization. V₀ cannot be detached, shared between examples, or reused across updates. A 6 MiB source at microbatch 1 does not by itself explain an 11 GiB peak delta.

**Falsifier:** mix buffers are not materialized on the critical path, or reading two inputs inside FA3 hurts more than the write it saves. **Tests:** coefficients zero/negative/>1, all local/global layers, layer-0 gradient sum versus independent reference, checkpointing and multiple updates.

## E5 — Treat FP8 operands as a lifetime-managed representation

**Observation:** the winner already caches quantized weights per update and uses split quantizers; DeepGEMM leaves casts/transposes/scales to its caller. **Hypothesis:** activations/gradients are converted or transposed more often than their actual consumer graph requires.

Build an operand graph: for each X, W and dY list logical axes, storage layout, scale granularity, rounding, forward/dgrad/wgrad consumers and validity interval. FP8 rowwise and columnwise quantization are generally **not equivalent**; transposing codes does not create the codes that quantizing the transpose would produce. Weight quantization may legitimately share the code transpose under the existing blockwise scheme; prove it from the implementation rather than generalizing to activations.

Try producing both required orientations and scale layouts in one read, or fusing exact quantization into RMSNorm/SwiGLU/dgrad output. Cache only where reuse exceeds retained-memory cost. Weight codes remain valid across accumulation microsteps until `optimizer.step`; version or clear caches on every update, checkpoint load, state transition, weight mutation and graph replay. Do not cache stale amax/scale metadata across updates.

**Cheap diagnostic:** counters/trace of casts and bytes per update, cache hit/miss ledger. **Falsifier:** casts overlap with GEMMs already or fused dual outputs reduce occupancy enough to lose. **Tests:** ragged M, 128-block boundaries, zero/large values, tensor version changes, graph replay twice, FP8 backward orientations and exact scale metadata.

## E6 — Shape-specific GEMM dispatch, especially wgrad

For `Y=XWᵀ`: forward is `(M,K)×(K,N)`, dgrad `(M,N)×(N,K)`, wgrad `(N,M)×(M,K)`. Do not tune all three with one algorithm. At small M, weight-gradient output dimensions and reduction K have very different occupancy/traffic constraints. Benchmark actual target shapes with exact scale/transposition costs and an initialized, realistic allocator.

Candidates: existing pinned DeepGEMM, suitable cuBLASLt/CUTLASS paths under the same numerical contract, tile/scheduling choices and split-K wgrad with FP32 partial accumulation. Sweep a small reasoned set of tiles/stages/SM allocations using roofline and wave count, not thousands of paid blind trials. Save dispatch by shape/phase/hardware; tuning costs belong in research, startup cache misses in production overhead.

Split-K creates more parallelism at the cost of partial buffers/reduction. Estimate `splits × N × K × accumulator_bytes` and reduction traffic. Persistent kernels help only when work distribution/reuse justifies them. A current upstream library may change APIs, scale formats and CUDA requirements; swapping it is a versioned environment trial, not an uncontrolled package update.

**Falsifier:** kernel-only advantage disappears after operand preparation or increases full-update memory. **Tests:** wgrad accumulation >1, mean/sum normalization, signs/large dynamic ranges, different output strides and weight-update parity. Preserve the production FP8 backward mode.

## E7 — Deferred or grouped wgrad across accumulation microsteps

**Potentially high impact, memory dependent.** At fixed weights during accumulation, `dW = Σ_i dY_iᵀX_i = concat(dY_i)ᵀ concat(X_i)` in real arithmetic. The reference performs a wgrad per microstep. If those are small/launch-bound, defer compatible wgrads and issue a larger or grouped operation; dgrad still runs in time to unblock preceding layers.

The cost is retaining X/dY operands until the deferred wgrad, scheduling around all backward dependencies, and altered FP32 accumulation order. Keep original quantization groups/scales; concatenating and re-quantizing changes the numerical experiment. 128-aligned token boundaries can help reuse the original quantized operands. Group only where the library's layout contract supports it. Microbatch interleaving may improve reuse but must not advance any parameter before the whole update/global clip is ready.

**First gate:** offline memory liveness simulation with measured shape sizes and actual accumulation count. Then implement one dominant MLP projection and compare complete gradients/updates to the accumulation reference. Limit the deferred group to a bounded number of microsteps when full deferral exceeds memory.

**Falsifier:** larger operand lifetimes force MLP checkpointing or reduce batch enough to lose; wgrad is already efficient; dgrad dominates. **Numerical gate:** if accumulation differences exceed the frozen class-E envelope, move to class N rather than relaxing it. This is not gradient staleness or skipped optimizer updates.

## E8 — Exact vocabulary loss: optimize what remains

**Observation:** `StreamingLinearCrossEntropy` already avoids repeated forward recomputation and accumulates weight gradients directly in FP32; the loss derivative is formed chunk by chunk in forward. Blindly replacing it with “fused loss” may regress speed. The remaining ~0.906 GFLOPs/token at BF16 and chunk-level logits/softmax traffic deserve direct measurement.

Compare current chunk sizes and an exact tiled linear-cross-entropy implementation that avoids a full `M_chunk×V` logits buffer. Online log-sum-exp needs stable maxima/sums; backward needs exact probabilities (often requiring logits recomputation) and all vocabulary gradients. Account for recomputed GEMMs, FP32 classifier-gradient accumulation and tied input-embedding contributions. A kernel that saves logits may trade memory for compute, not necessarily win both.

Apple CCE is a relevant implementation, but its default gradient filtering is an approximation. An exact comparison must disable **both classifier and embedding** gradient filtering (`cce_exact` at the inspected revision), preserve ignore index, sum reduction, pre-shifted targets and upstream loss scaling. Do not shift tokens twice. Precomputing gradients in forward changes profiler attribution; count the entire loss/autograd path.

**Falsifier:** current streaming chunks already avoid the bottleneck; recomputing vocabulary logits costs more than saved memory. **Tests:** tied weight receives both embedding and classifier gradients, duplicate target IDs, ignored labels, all-ignored reference cases, ragged last chunk, tiny upstream multipliers, gradient accumulation and optimizer update. Vocab parallelism is a later multi-GPU hypothesis, not a single-GPU win.

## E9 — Muon/Adam execution without changing optimization

**Observation:** the reference `LogicalMuonWithAuxAdam` loops Python over groups and separate Q/K/V and gate/up logical matrices. It uses Newton–Schulz-style Muon updates plus Adam for embedding/gains. At 1.5B, one update's optimizer time may be significant, especially if the production-local accumulation is small.

Profile momentum updates, norm reductions, polynomial matmuls, casts, weight decay and writes. Batch **independent same-shaped logical matrices across layers**, or use grouped kernels with bounded workspace. The polynomial must use the pinned iteration count, coefficients, normalization epsilon, momentum/Nesterov semantics and transpose rule. Rectangular-matrix orientation matters. Do not apply one matrix orthogonalization to the physically packed QKV/gate-up tensor: the winner intentionally treats logical slices separately.

Fuse Adam pointwise passes and small norm/coefficient updates; remove per-parameter host scalar synchronization only while preserving finite checks and error propagation. Keep FP32 master/state precision, step counters, decoupled decay, LR scaling and ordering. Weight cache regeneration may overlap other *independent* updates after global clipping, but next-step forward must observe all final weights and valid codes.

**Cheap gate:** random and saved-gradient one-step comparisons for every logical group and state; then repeated updates under a changing LR. **Falsifier:** Muon matmuls, not Python/pointwise overhead, dominate and grouping cannot improve utilization; workspace reduces attainable batch. Fewer NS iterations, approximate orthogonalization, quantized momentum or different groups are class N or a different optimizer campaign, not E.

## E10 — Remove host bubbles; selective compile and graph replay

**Observation:** the trainer reads loss/stop summary to host, performs clipping/error checks and clears caches each update. Existing CUDA graphs support microsteps on one GPU and reject checkpointing; compilation/stability monitoring have compatibility guards. Do not assume enabling all flags together works.

First remove avoidable dispatch/layout Python overhead with selective compilation of stable regions. Then consider static-buffer graph capture of the legal update region or microstep sequence, respecting optimizer tensor state, LR/step counters, FP8 cache lifetime, input buffer reuse, asynchronous copies and accumulation. Run a separate untimed telemetry pass when graph capture cannot include hooks; final training must retain the agreed safety/evaluation cadence.

Keep finite-gradient/loss failures observable before corrupting weights. Postponing all finite checks until after an epoch is a safety/semantics change, not host optimization. A capture-time input, LR or cached weight must not be replayed stale. Use a graph for each fixed shape/phase; never count replayed inputs as new unique stream progress unless the loader actually supplies the next data.

**Falsifier:** host gaps are hidden behind GPU work, capture overhead dominates a short run, or static buffers inflate peak. **Tests:** at least two weight-changing replays, variable LR, dense→QAT switch, resume, last partial accumulation, stop requests and injected nonfinite errors. Multi-GPU NCCL graph capture remains unverified until tested there.

## E11 — CPU/data/checkpoint pipeline as paid-time engineering

Inputs may be small in bytes but expensive in Python, page faults or synchronization. Use the same frozen token stream via contiguous mmap reads, pinned double buffers and asynchronous H2D. Do not “optimize” by benchmarking one repeated GPU-resident batch as production throughput. Check data-cursor/target identities across microbatch repartitioning, rank sharding and resume.

Checkpointing can move serialization/compression/upload off the critical GPU path only with an **immutable snapshot** of model, optimizer, RNG, scaler/FP8 state and cursor. A background thread reading live tensors while the optimizer updates them creates a corrupt checkpoint. Account for snapshot copies, memory pressure, NVMe/network bandwidth and durability delay. Rate-limit upload so it does not compete with production input or collectives.

**Falsifier:** data already overlaps completely or checkpoint cost is a negligible fraction of the 187.5-hour run. **Gate:** exact uninterrupted/resumed updates and cursor, partial-write recovery, hash-verified remote archive; elapsed/end-to-end cost including snapshot creation. Keep profiler and research telemetry off the production critical path only if the frozen diagnostic coverage is preserved.

## E12 — Attention execution specialized for the existing hybrid mask

The model is already on FA3, so “switch to FlashAttention” is no gain. It has twenty 1024-window causal layers and five 4096-global layers, GQA 18:6, head dimension 128. Compare actual local/global kernels and their backward paths. Confirm the current kernel does not compute substantially more masked tiles than necessary. Tune tile sizes, work partition, head grouping and layouts for this mix without changing any allowed query/key pair.

Try a local-window specialization only if the profile shows FA3's generality leaves exposed overhead. Reuse K/V across the three query heads within a KV group where it improves bandwidth, but recognize higher register pressure. Backward has GQA reductions; account for them. Handle diagonal/window edges exactly; tests must cross 1023/1024/1025 positions and query offsets rather than use only short all-global examples.

**Falsifier:** existing FA3 already handles the window efficiently, or specialized backward loses enough to erase forward gains. **Upper bound:** attention's exposed fraction, not all transformer FLOPs. A faster long-sequence global benchmark says little about this local-majority workload.

## N1 — FP8 head and/or attention, with a real error budget

This is the largest obvious remaining datatype opportunity, but not a guaranteed net gain. The head and attention are BF16 in the winner; hardware's FP8 peak cannot be claimed for them already. Test them separately, then together only if both survive. Inspect available support: the pinned FA3 README advertises FP8 **forward**, not a complete validated FP8 training backward.

For the tied head, preserving a FP32 master is straightforward; forward FP8 operands, dgrad FP8/BF16 choices and wgrad FP32 accumulation need separate investigation. Vocab logits/softmax retain stable FP32 reductions. Outliers and small logit differences can alter rare-token gradients even with small average logit error. For attention, Q/K outliers, softmax probability quantization and dS/dQ/dK backward sensitivities differ; forward-only speed is insufficient.

Evaluate tensor-relative and absolute errors, gradient direction/norm, clipping frequency, FP8 saturation and matched held-out loss across languages/code/math, early and late dense/QAT checkpoints. Use one-change numerical experiments with frozen calibration budgets. A forward error tolerance alone does not license 2T-token drift. Hadamard/incoherent transforms may help FP8 robustness but incur transform cost and must preserve the stack's learned gain/value behavior.

**Falsifier:** quantization/transposes cost the saved GEMM time; quality upper confidence bound exceeds the noninferiority margin; observed speed relies on an unsupported backward. Retain BF16 escape paths for sensitive operations only when frozen and measured—not dynamically cherry-picked per evaluation.

## N2 — Saved-activation compression where rematerialization is too expensive

Keep FP32 master/optimizer states and the forward computation unchanged initially; compress selected saved tensors between forward and backward. This can increase microbatch or avoid MLP recomputation, at the cost of backward approximation and quantize/dequantize traffic. Reuse an already-defined FP8 operand when the reference backward already consumes it; that may remain E. Compressing a BF16 residual/activation needed for a different derivative is N.

Choose saves by measured bytes×lifetime, derivative sensitivity and avoided recomputation, not “all activations FP8.” Compare per-channel/block scale layouts, outlier residual storage and BF16 exceptions. Do not add an activation compression rule to the production contract without evidence. Zero/rare-token channels, Peri-LN denominators and value-lambda reductions deserve special scrutiny.

**Falsifier:** decompression bandwidth or altered scale/error destroys the full-update/quality advantage. **Gate:** matched trajectories and late checkpoints, not a ten-step synthetic run. Stochastic rounding adds RNG state to checkpoint/resume and reproducibility contracts.

## N3 — QAT representation and optimizer bandwidth

Dense-to-ternary QAT has low-entropy forward weights but FP32 masters, gradients and optimizer states. Investigate whether exact ternary code/scale generation can be cheaper, cached once per update, or fused into required weight passes under the **existing** rounding/clamping/scale rule. That can be E. Packing weights for storage/transfer is useful only if unpacking/casting and scale handling beat the current FP8 path in complete training.

H100 has no general native ternary tensor-core GEMM, and unstructured zeros do not satisfy its 2:4 sparse instruction contract. A ternary training kernel must also supply real-valued dgrad, wgrad and STE gradients. Do not use the 2-bit checkpoint size as the training memory estimate or sparsity as a fictitious 2× compute peak.

Optimizer-state compression, low-rank gradients or error-feedback communication change numerics/optimization and require a separate owner-selected campaign. They are not first-line speed work while large exact execution waste remains. **Falsifier:** conversion overhead is small, compression expands critical-path traffic, or STE/scale gradients differ.

## E13 — Distributed plan prepared on one GPU, tested later

Prepare DDP bucket-ready timestamps, parameter/gradient sizes, local accumulation shape, host input costs and candidate checkpoint sizes. Build an analytical communication overlay using several real hardware-topology scenarios. Injecting delays can stress scheduling, but is only a simulation. Test rank-sharded input and global normalization on CPU/Gloo where useful; that proves indexing, not NCCL performance.

Prefer replicated execution if it fits and leaves sufficient activations; use `no_sync` on nonfinal accumulation microsteps. Compare ZeRO-1 optimizer state sharding only if its extra gather/broadcast traffic is warranted; full FSDP adds parameter all-gathers and may destroy the small-model efficiency. Vocabulary parallelism can split the large head but introduces softmax reductions and embedding/hidden communication. TP/context/pipeline parallelism are not free substitutes for good local execution.

**Later hardware gate:** compare fixed-global-batch 1/8/32 profiles on actual production topology, all ranks and both phases. Tune bucket sizes from measured readiness/overlap; do not count bandwidth twice or assume all-reduce of gradients can finish before V₁ fan-in. A microsecond NCCL collective benchmark cannot prove full-step scaling.

## E14 — Dependency-aware overlap inside one update

After gradients are complete and global clipping scale known, independent parameter updates and exact next-step FP8 weight encodings may be pipelined with other optimizer groups on separate streams. Similarly, wgrad may run beside upstream dgrad if resource contention allows it. Build explicit events and lifetimes. Overlap that saturates the same tensor cores/HBM simply slows both operations.

A particularly useful question is whether large embedding/MLP wgrads or Muon work create a serial tail that can occupy otherwise unused resources. Trace actual readiness rather than overlapping by arbitrary stream assignment. Do not let forward read a partially updated model, or calculate clipping from incomplete gradients. No asynchronous/stale optimizer variant is permitted.

**Falsifier:** total makespan stays constant or grows despite apparent overlap; memory live range increases beyond the production envelope. **Tests:** race checking, repeatability across stream timings, deterministic event graph, injected synchronization changes and complete optimizer/state parity.

## E15 — Tune the saved-tensor/tiling schedule jointly

**Exploratory idea:** describe a whole block's legal schedules as a constrained graph with memory sizes, recomputation costs and dependencies. Search a small set of schedules that jointly choose saved packed activations, dual FP8 layouts, deferred wgrad group size and microbatch. A greedy local “save this tensor” decision can make a globally poor choice by forcing a checkpoint elsewhere.

Start with measured costs and an offline dynamic program or bounded enumeration; validate only the best few on GPU. The objective is update makespan with peak-memory ≤limit, not a made-up kernel sum. Include optimizer cadence and data buffers. Use actual lifetime intervals, not independent peak sums; model overlap resource contention conservatively. Feedback from the first measured schedules should update cost estimates.

This could find gains beyond isolated kernel tweaks while preserving every equation. **Falsifier:** scheduler predictions fail on measured makespan, or complexity adds negligible gain versus a simple selective-checkpoint policy. Keep reference schedules and deterministic plan generation for reproducibility.

## E16 — Persistent producer/consumer fusion beyond pointwise kernels

**Exploratory, only with a dominant measured boundary.** Consider a Hopper WGMMA/TMA pipeline for gate/up → SwiGLU → exact FP8 pack → down, or GEMM output → Peri-LN → residual. The opportunity is avoiding large HBM round trips. The problem is that down-projection or RMS reductions span dimensions larger than a single producer tile; synchronization, multicast/cluster storage or a second stage may be necessary.

Derive tile ownership, full reduction domains, available shared memory/registers, inter-CTA communication and backward saves. A kernel cannot normalize a partial output-channel tile as if it were the full vector. Persistent CTAs may reduce launch overhead but starve occupancy across 25 layers; a two-kernel design can beat a giant fusion. Start with an isolated exact-shape forward/backward prototype and an optimistic byte/time ceiling before writing complex CUDA.

**Falsifier:** compulsory global reductions or spills erase the theoretical traffic saving. **Pass condition:** measurable complete-update gain with clear correctness and maintainable dispatch/fallback, not a heroic microbenchmark. Treat public fused-MoE/megakernel results as scheduling inspiration, not proof for this dense stack.

## Composition and stopping map

| Combination | Likely interaction to test |
| --- | --- |
| E1 + E5 + E7 | FP8 layout caches/deferred wgrad retain memory that rematerialization wanted to free |
| E2 + E3 + E4 | Complementary pointwise boundaries; beware giant fused reductions and losing BF16 round points |
| E6 + E7 | Larger wgrad may make split-K unnecessary; do not keep both blindly |
| E8 + N1 | Exact head loss plus FP8 head changes both compute and error; isolate first |
| E9 + E10 + E14 | Optimizer batching/graphs/overlap share state, streams and memory; replay/cache hazards |
| E1 + N2 | Compression may avoid checkpoint compute, or make it redundant |
| E13 + all local changes | Better local compute shortens the overlap window and can worsen scaling efficiency |

After every major win, repeat the attribution and physics-gap worksheet. Stop optimizing a category once its maximum remaining exposed-time saving is below the campaign's relevance threshold. Keep a high-risk queue with falsifiable diagnostics; do not replace evidence with a longer list of fashionable techniques. If the combined implementation plateaus, explain whether remaining costs are algorithmic lower bounds, hardware ceilings or engineering uncertainty, and direct the next experiment accordingly.
