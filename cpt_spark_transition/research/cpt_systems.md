# Spark CPT: Preserve Semantics, Then Scale Training

## Executive Summary

- **Preserve Spark, Not A Llama Approximation**: Spark's fused QKV, output gate, GeLU MLP and layer-specific masks are part of the model, not interchangeable implementation details. Keep its remote-code forward as the reference. [39]
- **Fix The Vocabulary Bottleneck First**: At 4096 tokens, the stated vocabulary produces **1 GiB of BF16 logits per sequence**, before loss workspace. Replace the linear-plus-loss path, not merely CE after logits already exist. Liger and CCE accept an external classifier tensor. [28][80][27]
- **Packing Is Conditional**: NVIDIA's documented **50-70% padding-compute waste** concerns variable-length RL/SFT, not these fixed-length shards. Do not change document attention or RoPE during the pilot to solve an unmeasured problem. [82]
- **Separate Loading From Acceleration**: HF Trainer supports a supplied custom PyTorch model; LitGPT and native Megatron require architecture-specific integration. A supported trainer does not make Spark compatible with that trainer's Llama kernels. [33][71][46]
- **Replicate Before Sharding**: Under the explicit mixed-precision accounting below, 1.71B parameters require about **27.36 GB of replicated model/optimizer state**, excluding activations. Start with DDP when scaling; use sharding to solve measured memory pressure, not because 100B tokens sounds large.
- **Keep BF16 For The Pilot**: Published CPT experiments include both FP8 instability and successful BF16-like convergence. FP8 is a recipe-dependent numerical change, not an established adaptation improvement. [35][31]
- **Resume The Experiment, Not Just The Weights**: Preserve optimizer, scheduler, RNG, loader cursor and token counters separately from the portable weights export. Distributed checkpoint resharding does not guarantee cross-version compatibility. [6][78]
- **Watch Forgetting Alongside Indic Gains**: A 2026 Nemotron adaptation report improves MILU while losing English MMLU/MMLU-Pro performance. Evaluate source-language retention from the first checkpoint, not only at the end. [51]

**Recommendation:** an explicit PyTorch training loop around the pinned Spark implementation, standalone fused linear CE, BF16 model computation, no packing and eager attention for the pilot. Scale that same model and checkpoint schema with DDP first; introduce attention kernels or FSDP2 only behind separate correctness and performance gates.

## 1. Freeze Spark's Contract Before Choosing Accelerators

The published configuration identifies `Spark2_5ForCausalLM`, `model_type="spark2_5"`, **2048 hidden size, 8 query heads, 2 KV heads, head dimension 256, 28 layers, vocabulary 131072**, and tied embeddings. Its `auto_map` points to custom configuration/modeling code. This is not a configuration that can safely be relabeled Llama. [28]

For this project, preserve the specified **21 sliding-window layers and 7 full-attention layers**, including their order and window 512. Preserve the GeLU-gated MLP, fused QKV split, and the sigmoid headwise gate applied to attention output before the output projection. Also retain the different full/sliding RoPE parameters: full attention uses partial rotary factor 0.25 and theta 5M; sliding attention uses factor 1 and theta 10000. [28][39]

The important version distinction is **current ecosystem version versus model-supported version**:

| Component | Late-2026 status | Project decision |
|---|---|---|
| PyTorch | **2.15.0**; release announcement updated October 28, 2026. [5][73] | Candidate systems baseline; lock its CUDA/Triton build and test the custom code. |
| Transformers | **5.22.2**, published November 6, 2026. [74] | Do not automatically upgrade the model to it. |
| Spark Transformers contract | Official instructions and checkpoint configuration specify **4.57.1**. [8][28] | Pin **4.57.1** for the reference and pilot. |
| TorchData | **0.11.0** release. [68] | Use StatefulDataLoader or an explicit equivalent cursor implementation. |
| Evaluation harness | **0.4.13** release. [63] | Freeze the release plus task/configuration revisions. |

These are individually documented versions, **not a claim that every latest package forms a tested environment together**. In particular, current trainer packages may expect Transformers 5, whereas this model was published against 4.57.1.

**Spark-specific implementation case.** Its outer forward selects hidden states, calls `F.linear` with `self.model.embedding.weight` when tied, then calls the model loss function. Therefore a loss adapter can preserve the decoder and weight layout while bypassing the full-logit allocation. Calling the original outer forward first and discarding its logits would miss the optimization entirely. [39]

Make that adapter part of the model's invoked training forward, without adding or renaming Parameters. Do not bypass the DDP/FSDP wrapper by calling a deeply nested `.module.model(...)` from the loop. Retain the original inference/evaluation forward and canonical HF state-dict keys.

**Decision:** freeze the model repository revision, remote-code files, tokenizer and configuration before optimizing. The first optimization should change the loss implementation, not Spark's architecture.

## 2. Remove Full-Vocabulary Logits Without Untying The Embedding

### Memory arithmetic

For one sequence at length 4096:

`4096 * 131072 = 536870912 elements`

`536870912 * 2 bytes = 1073741824 bytes = 1 GiB`

That is **1 GiB of BF16 logits per microbatch sequence**. At length 2048 it is 0.5 GiB. An additional FP32 copy is **2 GiB at 4096**, if retained concurrently; gradients and loss workspace are additional. The tied embedding itself is `131072 * 2048 * 2 = 512 MiB` in BF16. These are dimensional calculations, not measured Spark peak memory. The dimensions and actual tied `F.linear` path come from the checkpoint. [28][39]

| Implementation | Hidden states plus existing tied `[V,H]` weight? | Avoids the full `[T,V]` allocation? | PyTorch CE equivalence and limitations |
|---|---|---|---|
| **cut-cross-entropy** | **Yes.** `linear_cross_entropy(embeddings, classifier, labels)` takes the tensor directly; it need not be an `nn.Linear` module. | Yes, its intended replacement is linear plus CE. | Same CE objective, but floating-point order differs. Current main exposes gradient filtering; disable it with `filter_eps=None` for the reference comparison and consider FP32 gradient accumulation. The published PyPI release is **25.1.1**; do not assume every current-main option exists in that tag. [27][38][57] |
| **Liger-Kernel 0.8.4 fused linear CE** | **Yes by interface:** `forward(lin_weight, _input, target, bias=None)`. Pass the existing embedding Parameter. | Yes. It chunks the token dimension and bounds transient logits rather than retaining the full tensor. | Defaults represent ordinary CE when smoothing, z-loss, softcap and token scaling remain disabled. Use explicit FP32 gradient accumulation for the correctness baseline; default accumulation can follow parameter dtype. No bitwise-equivalence promise. [13][80][83] |
| **flash-ce / Flash CE** | The name is ambiguous; no versioned universal tied-linear-loss API should be assumed. | FlashAttention's documented `CrossEntropyLoss` takes **already materialized `[batch,vocab]` input**, so it does not solve the linear-logits allocation. The separately named Apsod/FlashCE is a CPU-tested prototype, not a production GPU recommendation. | FlashAttention CE defaults to ordinary CE-style reduction, but remains CE-only. Apsod's repository reports no releases and leaves GPU implementation as an exercise. [81][50] |
| **torchao 0.18.0** | Not a documented standalone linear-CE provider. | Do not attribute this optimization to torchao. | TorchAO provides quantization and optimizer-memory tools; those are different changes from exact linear CE. [58][41] |
| **torchtitan 0.4.0** | Its loss machinery can be integrated with a compatible model/head, but not automatically with Spark remote code. | `ChunkedLossWrapper` skips the normal head and processes hidden-state chunks. Ordinary Titan CE and vocabulary-parallel CE still receive logits. | Chunked ordinary CE preserves the objective, subject to reduction/rounding details. Its `_skip_lm_head` integration contract requires an adapter or port. [11][22] |
| **PyTorch 2.15 native `F.linear_cross_entropy`** | **Yes**, accepts an external linear weight. | Only when its chunked options are enabled. **`options=None` is the reference path**, not a guarantee of memory saving. | Documented equivalent to `F.linear` plus `F.cross_entropy`, with the documented `ignore_index` qualification. Chunked mode precomputes gradients and lacks higher-order/forward-mode AD; compilation falls back at the chunked operation. [29] |

**Takeaway:** use a standalone tensor-level loss, not an architecture monkeypatch. My first candidate is **Liger 0.8.4**, with native PyTorch chunked linear CE as a fallback; CCE is also credible after its filtering and version details are controlled.

The integration should obtain final normalized hidden states from the unchanged Spark decoder and supply **the same `self.model.embedding.weight` Parameter** to the loss. Do not detach it, copy it into an independent classifier, or create a second optimizer entry. The gradient must combine both its input-embedding and output-classifier uses; shared embeddings are explicitly a shared-gradient use case. [39][3]

Apply the causal shift **exactly once**. Preserve `ignore_index=-100`, valid-target reduction and the existing label alignment. CCE has a `shift=1` route; Liger receives explicitly aligned flattened tensors. For accumulation/DDP, normalize by the total valid-target count, not an unweighted mean of microbatch means when counts differ. [27][80][83]

Before using the replacement, compare loss, hidden-state gradient, tied-weight gradient and one optimizer update against the original `F.linear` plus FP32 CE path, including ignored labels and accumulation. Treat tolerance as a numerical acceptance criterion, not bitwise equality. **Passing these checks makes this a systems replacement; changing the objective's defaults does not.**

## 3. Attention Kernels Can Preserve The Gate; Packing Needs A New Contract

The gate is not an exotic softmax mask. Spark computes attention, then multiplies its per-head output by a sigmoid derived from `g_proj(hidden_states)`, before the output projection. Therefore a backend can implement the attention core and leave the gate unchanged outside the kernel. Putting this gate into a logit bias or FlexAttention `score_mod` would implement a different model. [39]

| Backend and version status | Sliding/full attention | Headwise output gate | Document isolation / Spark work required |
|---|---|---|---|
| **FlexAttention, PyTorch 2.15 API** | Composable causal/window `BlockMask`; current API includes GQA. Still documented as **prototype**. [30][23] | Preserve the existing multiply after returned attention. | Compose same-document, causal, padding and window conditions. Must change Spark's mask/forward plumbing; it is not activated by a config flag. [30][23] |
| **PyTorch SDPA 2.15** | Can express the window through `attn_mask`; there is no dedicated window argument in the documented signature. Backend eligibility depends on the inputs/mask. [85] | Preserve the external multiply. | A dense block-diagonal mask can express correctness, but may retain quadratic mask storage or cause backend fallback. It is not a sparse-packing guarantee. |
| **FlashAttention-2, 2.8.3.post2 release** | Native causal/local window support. [64][32] | Preserve the external multiply. | Use variable-length sequence boundaries for isolated documents; adapt Spark's Q/K/V layouts and mask handling. |
| **FlashAttention-3, Hopper implementation** | Current interface includes local `window_size`, GQA-compatible separate Q/K/V, and varlen functions. BF16/FP16 forward and backward are documented; the advertised FP8 path is forward-only. [70][32] | Preserve the external multiply; there is no Spark-gate argument in the attention call. [70] | A manually integrated option for H100, not automatic Spark support. Pin the exact Hopper source/build. |
| **xformers 0.0.35** | Current exports include local, causal and block-diagonal-local bias classes. [62][55] | Preserve the external multiply. | Select and test an operator supporting this shape, bias and backward. Do not infer training support merely from a bias class existing. Retain explicit KV expansion if a selected operator cannot train with Spark's GQA layout. |

**Takeaway:** all are plausible attention-core implementations with an external gate; none makes the published Spark forward automatically kernel-compatible. Setting `attn_implementation="flash_attention_2"` does not rewrite its hardcoded eager attention.

An upgrade must preserve QKV partitioning, query/KV head mapping, scale, mask polarity, exact window boundary, layer order, rotary dimensions/frequencies, softmax precision, dropout and gate/projection order. FlashAttention windows are inclusive; match Spark's actual 512-token mask rather than guessing whether the argument should be 511 or 512. SDPA Boolean masks use `True` for allowed positions, and a nonzero dropout argument applies even during evaluation. [39][28][70][85]

### Packing: neither blanket refusal nor a blind position-ID fix

The published forward builds masks from a 2D padding mask, computes RoPE from `cache_position`, and does not use supplied `position_ids` for those rotations. Its checkpointed layer call passes hidden states, RoPE and the selected mask only. A generic trainer's position-reset packing is therefore insufficient. [39]

For isolated-document packing, implement all of the following together:

1. Carry document boundaries as document IDs or cumulative sequence lengths.
2. Enforce **causal AND same-document AND padding validity**, plus the local-window condition in sliding layers.
3. Mask the loss target that would cross each document boundary.
4. Pass the correct masks/metadata through checkpoint recomputation, not just the ordinary forward.
5. If resetting positions, explicitly make RoPE honor per-token `position_ids`, while preserving Spark's per-layer rotary parameters.

There is an important mathematical nuance: with fixed-frequency RoPE, adding the same positional offset to every token within an isolated document cancels in its Q/K dot products. Thus, monotonic positions alone do **not** prove isolated packing impossible. Nevertheless, changing to explicit per-document positions plus FlexAttention is a clean optional extension, not a one-line correction. Position resets alone do not create document isolation, and FlexAttention is not the only possible implementation.

**Pilot decision: no new document packing.** Keep the existing shard interpretation. Later, add the explicit contract only if there is a measurable utilization problem and compare each packed document against a separately evaluated reference.

### What does padding waste at 4k actually mean?

NVIDIA's NeMo RL documentation quotes **50-70% computation wasted on padding** for variable-length RL/SFT, without defining that as a 4096-token CPT result. Papers studying packing at maximum length 4096 likewise concern particular SFT datasets, not a universal Spark padding fraction. [82][37]

| Hypothetical useful tokens in a padded 4096-token row | Padding fraction, calculated |
|---:|---:|
| 512 | 87.5% |
| 1024 | 75% |
| 2048 | 50% |
| 4096 | 0% |

The formula is `1 - useful_tokens/4096`; these are illustrations, **not measurements**. Already full, nonpadding fixed-length memmap rows have no padding to recover. Packing may still change document boundaries or sparse attention work, but that is a different argument. Published packing work also reports changes in training-example/optimization behavior and some downstream regressions, so packing is not automatically semantically free. [37][36]

## 4. Which Trainers Can Actually Train This Remote-Code Model?

Here, **generic-capable** means the library can train a correctly loaded model through a generic path. It does not mean Spark has a validated optimized recipe, nor that its latest release supports Spark's older Transformers dependency unchanged.

| Trainer / release snapshot | Full-parameter Spark CPT status | Custom-architecture failure mode or required action |
|---|---|---|
| **HF Trainer, model-pinned Transformers 4.57.1** | **Yes: strongest generic fit.** Supply an instantiated remote-code model or compatible module. [33] | Override loss cleanly if desired; preserve model-output/labels contract. Trainer cannot repair Spark masks or make its attention dispatchable. [33] |
| **TRL 1.17.1** | **Generic-capable**, accepts a model instance; no PEFT config means no adapter wrapping. [52][34] | Configure ordinary all-token NLL, no chat formatting, completion-only loss, packing or padding-free assumptions. Current TRL docs target newer Transformers; do not assume installation compatibility with 4.57.1. [34] |
| **Axolotl 0.21.1** | **Generic-capable.** Its documentation explicitly distinguishes generic HF support from added architecture accelerations. [53][10] | Disable architecture patches and sample packing. Its fused LoRA kernels explicitly exclude `trust_remote_code`; a generic loading claim does not validate those kernels for Spark. [10] |
| **LlamaFactory 0.9.5** | **Conditional generic HF pretraining path**, not Spark-specific certification. [54][12] | Set remote-code trust explicitly, full training and pretraining stage. **Disable packing explicitly: it defaults on for pretraining.** Disable attention/architecture substitutions. [12] |
| **Unsloth 2026.10.6** | **Possible generic remote-code fallback; no verified Spark accelerated path.** Its FAQ allows remote-code models, while loader logic still contains architecture-dependent handling. [59][7][72] | Do not claim categorically that it cannot load Spark. Equally, do not assume its compiler/full-finetuning kernels preserve Spark's fused QKV, gate and masks. Use full finetuning, no quantized/adaptor path, only after a dedicated audit. |
| **LitGPT 0.5.13** | **Not an unchanged HF remote-code loader for Spark.** [56][71] | Native config/model support and weight conversion are required. Falling through a Llama/GPT-NeoX conversion does not implement Spark's gate or hybrid attention. |
| **torchtitan 0.4.0** | **Native recipes do not list Spark.** Its HF integration offers a dense-model route, but the documented route does not establish Spark `auto_map` compatibility. [60][69] | Add/test a Spark adapter or native recipe; wire hidden-state loss, layer sharding and masks. It is not a drop-in `trust_remote_code` replacement. |
| **Megatron-LM / Core 0.20.0** | **Not out of the box for this checkpoint.** [66][46] | Implement provider/bridge, custom layer components and bidirectional mappings. Fused QKV mapping exists, but custom sliding/RoPE fields and the output gate still need explicit handling. [46] |
| **NeMo AutoModel 0.7.0** | **Conditional generic support**, including documented remote-code enablement. NeMo's native Megatron path is a separate integration. [61][44][42] | Config must resolve; architecture registration alone does not register a new model type. Third-party/model-specific features can still fail. [44][42] |
| **DeepSpeed 0.19.9** | **Yes as an engine around the already loaded PyTorch model.** [65][26] | Avoid architecture-specific injection/AutoTP assumptions. **Universal Checkpointing is not universal architecture loading** and does not translate model families. [25] |

The practical distinction is illustrated by Axolotl versus Megatron: Axolotl can retain a generic HF forward while foregoing special kernels; Megatron's bridge documentation requires config translation, tensor mappings and custom blocks. A fused QKV converter alone does not reproduce Spark's computation. [10][46]

**Pilot loop:** explicit PyTorch, with a small training-only forward/loss adapter, transparent accumulation and token accounting. HF Trainer 4.57.1 with an overridden loss is a reasonable convenience alternative.

**Throughput loop:** keep that explicit model/loop, add distributed execution, profiling and tested compilation/kernel substitutions. A Titan-style recipe or Megatron port becomes attractive only if it delivers enough measured benefit to repay architecture and checkpoint-conversion maintenance. At 1.7B, a native-engine rewrite is not a prerequisite for a 100B-token run.

## 5. Use Memory Headroom Deliberately, Not Automatically

### DDP, FSDP2 and ZeRO on H100 80GB

For an illustrative conventional mixed-precision layout, assume **2 bytes BF16 weight + 2 bytes gradient + 4 bytes FP32 master weight + 8 bytes Adam moments = 16 bytes/parameter**. For approximately 1.71B parameters, that is **27.36 GB**. This is an accounting assumption, not a claim about every optimizer implementation.

| Strategy | Model-state accounting per GPU | Illustration at 8 GPUs | When I would use it |
|---|---:|---:|---|
| **DDP** | `16P` | **27.36 GB** | First multi-GPU baseline if activations fit. No parameter all-gathers during forward. |
| **ZeRO-1** | `4P + 12P/N` | **9.405 GB** | Shard optimizer/master state while retaining replicated compute weights. |
| **ZeRO-2** | `2P + 14P/N` | **6.4125 GB** | Additional gradient-memory pressure. |
| **ZeRO-3** | Approximately `16P/N`, plus transient buffers | **3.42 GB**, before gathers/buffers | Only when parameter sharding solves a real problem. |
| **FSDP2** | Shards parameters, gradients and optimizer state; exact bytes depend on policy | Roughly inverse-world-size persistent state, plus live unsharded groups | PyTorch-native sharding when larger microbatches or other requirements justify its communication. |

ZeRO's documented stages successively shard optimizer state, gradients and parameters. FSDP2 similarly reduces persistent state but gathers groups for computation; its grouping determines overlap and peak memory. Thus these figures are not complete peak-memory predictions. [76][48][47]

**Spark memory case.** One dense `[1,8,4096,4096]` BF16 attention-score tensor is **256 MiB**, or 512 MiB if represented in FP32. A sliding mask does not eliminate the quadratic tensor in the published eager implementation. This makes activation/loss memory a more immediate target than sharding a small model's weights. Start at microbatch 1 and sequence 2048, then test 4096; do not promise a microbatch size before measuring peak allocated/reserved memory.

Use **full decoder-block activation checkpointing** first if necessary, with training KV cache disabled. Spark's existing three-argument recomputation path is suitable only while all required attention information is contained in those arguments. Selective checkpointing can retain expensive operations and recompute cheaper ones, but blindly saving all matrix multiplications can retain the very quadratic attention tensors you wanted to remove. [39][40]

Defer CPU offload. FSDP CPU offload moves parameters, gradients and optimizer state and performs CPU-side optimizer work; this is distinct from activation offload. Both add transfer/resource-management costs that should solve a demonstrated capacity problem, not be enabled by default on an 80GB pilot. [47]

For FSDP2, shard Spark decoder blocks bottom-up, then the root. Keep the tied embedding/classifier in a consistent owning group and execute its loss while the relevant parameters are available. The standard FSDP2 mixed-precision example keeps FP32 sharded/optimizer state with BF16 compute and FP32 reduction. [48]

### BF16 versus FP8: the CPT evidence conflicts for a reason

A Llama-3-70B CPT study over **100B multilingual/code tokens** reports frequent FP8 loss spikes and disproportionate code/math degradation, while QA is relatively resilient. It also uses different hardware environments for BF16 and FP8, so its speed comparison is not a clean Spark capacity forecast. [35]

Conversely, a later FP8 recipe continues **Qwen2.5-1.5B and 7B for 160B tokens**. Its controlled 1.5B comparison reports nearly indistinguishable BF16/FP8 training and validation trajectories and broadly comparable downstream results. This is directly relevant evidence that FP8 CPT can work, but not that reduced precision itself improves adaptation or works unchanged for Spark. [31]

**Decision:** retain BF16 model computation and explicit FP32 optimizer/master state. Do not assume `torch.optim.AdamW` on BF16 Parameters automatically supplies FP32 masters and moments. For the plain loop, use an audited master-weight optimizer arrangement that updates unique FP32 copies and copies results back into the existing BF16 Parameters; preserve tying by Parameter identity. Do not simply cast Spark's entire module to FP32 and assume autocast is identical, because its forward contains explicit weight-dtype casts. [39]

An FP8 branch needs its own matched-token BF16 comparison, precision exclusions, scaling/amax state, spike analysis and adaptation/forgetting evaluation. Titan 0.4's MXFP8 path additionally requires torchao nightly rather than 0.18; do not mistake that release feature for a validated H100 Spark recipe. [11]

### Clipping, spikes and a token clock

As a starting engineering policy, clip global gradient norm at **1.0**, after accumulation and distributed reduction, before the optimizer update; tune this from observed norms, not as a model-specific proven optimum. Fail and diagnose nonfinite gradients. If a skip policy is necessary, synchronize the decision across ranks, clear all gradients and do not advance optimizer/LR-update state on a skipped update. Log consumed tokens separately from successfully optimized tokens. Megatron's rerun machinery provides explicit validation/rerun/skip infrastructure; skipping finite high-loss examples indiscriminately changes the training distribution. [77]

Use **warmup-stable-decay (WSD)** with boundaries expressed in the run's token budget. Titan's documented implementation is step-based; Megatron Bridge offers iteration- or sample-based WSD, not an automatic universal valid-token clock. With fixed-length rows, convert carefully; otherwise drive the schedule from an explicit accumulated counter. Save that counter and schedule boundaries so a world-size or accumulation change does not reset the LR trajectory. [75][20]

## 6. Load Fixed-Length Memmaps With A Committed Resume Cursor

For already-built int32 shards, the best loader is usually simple rather than transformation-heavy. Use a read-only map-style dataset with a deterministic mapping from global sample ID to shard/row; open mappings lazily in each worker. Batch fixed-size views, produce the integer dtypes expected by embedding/CE at the batch boundary, and keep all CUDA work out of workers.

| Concern | Recommended implementation |
|---|---|
| **Worker count** | Start with **0 workers per rank**, then benchmark **2 and 4**. Already-tokenized cheap reads may be faster without multiprocessing; excessive workers consume memory/shared-memory and IPC bandwidth. [14] |
| **Worker lifetime / prefetch** | With workers, use persistent workers and a small bounded prefetch depth; 2 batches per worker is the documented default, not a universal optimum. [15] |
| **Resume state** | Stateful sampler plus `StatefulDataLoader.state_dict()/load_state_dict()`. Include per-worker state if it exists; save state separately for each distributed rank. TorchData aggregates workers, **not ranks**. [6][15] |
| **H2D overlap** | Pinned batches, `non_blocking=True`, and a bounded copy-stream/double-buffer scheme with event synchronization and correct buffer lifetime. DataLoader pinning plus transfer overlap are documented techniques. [14] |
| **Ordering** | Deterministic global sampler/permutation, explicit epoch and cursor; avoid out-of-order delivery when exact sample replay matters. |
| **Integrity** | SHA-256 of shard contents and ordered manifest; record dtype, endianness, row length, shapes and sample counts. Also hash tokenizer and model/code/config identities. |

The key failure case is saving a producer's prefetched position rather than the last **committed training position**. On restart, this can silently omit data. Save the loader state corresponding to the batches incorporated in the checkpointed update; maintain a committed global cursor as a cross-check against the yielded sample IDs.

TorchData's custom sampler state avoids replaying an entire prefix. Worker-specific state can also be captured, but its documented iterable-worker restore requires the same worker count. Changing world size is a separate global-sample reassignment problem, not something per-rank worker state alone resolves. [6]

My resume acceptance test is an interrupted short run versus an uninterrupted one: compare subsequent sample IDs, token counters, LR, loss and parameter updates. Run it with prefetch enabled, not only with a single synchronous worker. This tests the interaction of loader state, accumulation and checkpoint publication.

**Decision:** optimize storage/H2D overlap only after profiling loader wait. For fixed-length memmaps, more workers and a more elaborate streaming framework are not automatically improvements.

## 7. Evaluate A Base Model Without Confusing Scoring Protocols

Use **lm-evaluation-harness 0.4.13 with its HF backend** as the reference evaluator. It accepts a model ID, local checkpoint or instantiated model, and exposes remote-code loading. This avoids assuming that an inference engine supports Spark merely because Transformers does. Freeze task YAML, dataset revision, few-shot split/seed, prompt, answer delimiter and metric. [63][67][43]

A custom letter-logit loop is valuable for frequent inexpensive monitoring, but it measures a particular protocol:

- For genuinely single-token labels, compare the correct contextual token IDs for `A/B/C/D`, including leading whitespace. Read the last **nonpadding** prompt position.
- If a label spans multiple tokens, teacher-force and sum its conditional log probabilities; do not inspect only its first token.
- Projecting only the final hidden state avoids all-position logits. Projecting only candidate-label rows preserves their argmax, but a softmax restricted to those labels is not the full-vocabulary probability or a general NLL estimate.
- Do not use a chat template or assistant-only scoring for a base checkpoint unless that is the explicitly chosen benchmark protocol.

**Length normalization is not one universal operation.** The harness code examined here computes multiple-choice `acc` from summed continuation log likelihood and `acc_norm` using continuation **character length**; it also tracks byte length. Do not describe that implementation as token-length normalization, or compare it directly with a four-letter logit scorer. Equal single-token labels do not require answer-text length normalization. [86]

### Use the Indic suites, but distinguish the papers and task identities

NVIDIA/AI4Bharat's October 2026 *Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs* evaluates **MILU and translated MMLU-Pro, ARC-Challenge and GSM8K**, and includes English retention evaluation. That is not evidence that every NVIDIA Indic paper uses the same prompts or a letter-logit protocol. [17]

A separate **Gnani.AI** report adapting NVIDIA's Nemotron-3-Nano-30B-A3B provides a useful success/failure case. Its before/after results include the four requested benchmark families, but its task is specifically **IndicMMLU-Pro**, not an unspecified interchangeable "Indic MMLU." [51]

| Benchmark in that report | Base | Annealed checkpoint | Monitoring role in this project |
|---|---:|---:|---|
| MILU | 40.63 | 53.52 | Target-language knowledge/understanding sentinel |
| ARC-Challenge-Indic | 35.23 | 48.20 | Translated reasoning/science sentinel |
| Belebele | 37.49 | 42.27 | Multilingual comprehension sentinel |
| IndicMMLU-Pro | 14.44 | 20.02 | Broader target-language academic/reasoning sentinel |
| English MMLU | 78.50 | 75.11 | Retention sentinel |
| English MMLU-Pro | 64.83 | 55.43 | Harder retention sentinel |

All scores in this table are that report's results, **not Spark measurements**. [51] The central lesson is that strong Indic gains can coexist with material English forgetting; an aggregate target score alone can approve the wrong checkpoint. [51]

For Spark, freeze exact variants of MILU, ARC-Indic/ARC-Challenge-Indic, Belebele and Indic MMLU or MMLU-Pro separately. Validate the custom scorer against the harness on identical examples before treating their outputs as interchangeable.

**Suggested token-based cadence, not a published NVIDIA rule:** evaluate the untouched baseline first; during the pilot run held-out NLL every **25-50M** successful nonpadding training tokens and a small task panel every **100-250M**. For the long run, start with NLL every **100M**, a small panel every **500M**, and the full suite every **1B**, then adjust to measured evaluation cost and observed change rate.

Track target and source-language NLL separately; per-language task accuracy and worst-language movement; source English knowledge, reasoning, math/code sentinels; loss/gradient-norm tails; and optional fixed-prompt logit drift. Keep tokenizer and evaluation tokenization fixed. Report confidence intervals or paired uncertainty rather than interpreting tiny checkpoint differences as real gains; the NVIDIA study itself cautions that single-run differences are descriptive. [17]

**Decision:** use the harness for reproducible release evaluation and a validated letter scorer for frequent monitoring. Select checkpoints on adaptation **and retention**, not training loss alone.

## 8. Publish Portable Weights And Separately Resume Training

Maintain two artifacts with different purposes:

| Artifact | Required contents | Compatibility promise |
|---|---|---|
| **HF weights-only export** | Safetensors, configuration, exact remote-code files, tokenizer files, model/code revision and hashes | Reloads the same Spark architecture without the training engine |
| **Training-resume checkpoint** | Optimizer/master state, scheduler, token counters, RNG, per-rank loader state, step/accumulation state, precision recipe and environment manifest | Resumes a specifically supported training environment |
| **Commit manifest** | File hashes, sizes, step/token identity, rank completion and export/resume linkage | Identifies a complete, internally consistent checkpoint |

The tied-weight case matters. Safetensors documents `save_model/load_model` for shared tensors rather than blindly applying `save_file(model.state_dict())`; alias handling can omit duplicate keys intentionally. For HF export, use the model-aware `save_pretrained(..., safe_serialization=True)` path and test reload with this custom model. Check that the embedding used by the classifier is still the intended Parameter and that no independent head has appeared. [3][39]

**Publication protocol:** save at an optimizer-update boundary, with no pending accumulated gradients. Write to a unique temporary directory on the same filesystem; finish writes, flush/synchronize as appropriate, verify hashes and collect all-rank completion. Publish the completed directory by rename, then atomically update the latest-checkpoint pointer. For object storage, publish a commit manifest last rather than assuming directory-rename semantics. Preserve the previous committed checkpoint until the next one is complete.

Store hashes of model tensor files and indexes; configuration and remote-code source; tokenizer vocabulary/configuration; shard contents and ordered manifests; training/evaluation configuration; task definitions; dependency lock/container image; and optimizer parameter-name/group mappings. Save Python/NumPy/PyTorch CPU and CUDA RNG states where used, data-generator states, successful-update and consumed-token counts, and any skip history. A BF16 weights export does not preserve the higher-precision optimizer trajectory by itself.

For distributed training, prefer **PyTorch Distributed Checkpoint with canonical model/optimizer FQNs**. It supports resharding across trainer counts/parallelism, but all ranks must participate consistently and use matching keys; mismatches may hang or error. Do not call a collective checkpoint API only on rank 0. [78]

Two concrete pitfalls deserve explicit treatment:

1. **Version changes:** DCP does not promise cross-PyTorch-version compatibility. PyTorch 2.15 specifically records stride metadata that prevents its checkpoints loading in 2.14 or earlier. Keep portable HF weights as the escape hatch, but recognize that restarting from weights is not exact optimizer resume. [78][5]
2. **Async mutation:** asynchronous saving must first obtain a training-safe staged snapshot. Do not mutate live tensors before staging is complete, and do not publish until the save completes. [78]

**Decision:** preserve canonical HF parameter names from the pilot onward. Use engine-specific sharded state for recovery and an independently reloadable HF export for portability; test both paths before the long run.

## Synthesis: Two Stacks, One Checkpoint Contract

The three main strategies differ in **mechanism, integration scope, numerical risk and maintenance cost**:

| Strategy | Mechanism / scope | Trade-off | Recommendation |
|---|---|---|---|
| **Generic HF/PyTorch + DDP** | Runs the actual remote-code model; replicates state | Lowest architecture-porting risk, higher per-GPU persistent memory | Pilot and first multi-GPU baseline |
| **Same model + FSDP2 or ZeRO** | Changes state ownership and communication | More memory headroom, more collective/lifecycle/checkpoint complexity | Adopt only when measured batch/activation needs justify it |
| **Native Titan/Megatron port** | Reimplements model integration and mappings | Broad systems control, largest semantic and conversion burden | Later option, not the default route for a 1.7B model |

This comparison follows the generic model contracts, sharding mechanics and native-port requirements documented above. [33][48][76][46] The non-obvious conclusion is that **100B tokens changes the importance of reliability and utilization, not the model's instantaneous parameter-memory footprint**. It does not by itself justify tensor/pipeline parallelism or a native architecture rewrite.

### Concrete recommended stacks

| Component | Single 80GB pilot | Approximately 100B-token run |
|---|---|---|
| **Runtime / model** | Python 3.11 as an environment choice; **PyTorch 2.15.0**, **Transformers 4.57.1**, exact Spark repository revision; explicit PyTorch loop | Same remote-code implementation and canonical names; retain the validated environment initially |
| **Precision / optimizer** | Existing BF16 model computation, explicit FP32 master/AdamW state, unique tied Parameter handling | Same baseline; precision changes only in a separately evaluated branch |
| **Loss** | Original loss for reference tests; then standalone **Liger-Kernel 0.8.4** fused linear CE with explicit accumulation/reduction settings | Same loss adapter; consider native PyTorch chunked linear CE or CCE only after matched tests |
| **Attention** | Published eager attention; original hybrid masks and RoPE | Initially identical; optional hand-integrated FA3 on H100 or FlexAttention after forward/backward/update comparisons |
| **Sequence / batch** | Start at 2048, microbatch 1; test 4096 and accumulation from memory measurements | Fix successful tokens/update deliberately; choose microbatch/accumulation from measured utilization |
| **Parallelism** | Single GPU; full-block checkpointing if needed | **DDP first**; FSDP2 or ZeRO-1/2 if headroom improves useful throughput; no automatic TP/PP |
| **Data** | Fixed-length memmaps; TorchData **0.11.0** stateful loader; start 0 workers, tune 2/4; pinned bounded prefetch | Same global sample-ID/cursor contract, per-rank state, measured H2D overlap |
| **Evaluation** | Harness **0.4.13**, fixed tasks and a cross-checked letter scorer | Same baseline/task identities, token-based cadence, source retention gates |
| **Checkpointing** | HF safetensors export plus separate complete resume state | Canonical-name DCP state plus the same HF export and commit manifest |

The library versions are release snapshots, not an assertion of pretested binary compatibility; build and lock the actual environment. In particular, do not install an old FlashAttention wheel built for a different PyTorch ABI into the proposed runtime. The inspected FA2 release assets include builds targeting older Torch versions. [64]

**Do not turn on during the pilot:** automatic packing/padding-free execution, position resets, RoPE scaling, attention monkeypatches, Llama/SwiGLU substitutions, FP8, quantized optimizer states, CPU/NVMe offload, native-engine conversion, or whole-model compilation. Start compilation later with narrowly scoped, tested regions; it is not required to establish correctness.

### Semantic changes versus systems changes

- **Systems changes, after validation:** pinned loading, data prefetch, accumulation with equivalent token normalization, activation checkpointing, DDP/sharding, atomic checkpoint publication, and linear-CE replacement that preserves the loss and tied gradients.
- **Numerically different implementations of the same intended operator:** SDPA/FlashAttention/FlexAttention, reduction ordering and some fused losses. Accept only after comparison with eager Spark. FlashAttention and FlexAttention explicitly discuss tolerance-based rather than bitwise equivalence. [32][30]
- **Training/semantic changes requiring a separate experiment:** document isolation where cross-document attention previously existed, new position-reset behavior, changed RoPE/window/MLP/gate, different loss weighting, changed successful tokens/update, and FP8 or optimizer quantization.

For planning, use `wall_seconds = 100000000000 / measured_successful_nonpadding_tokens_per_second`, then include evaluation, checkpoint and failure-recovery time. **There is no defensible Spark throughput or completion-time number in the sources reviewed.** The correct path is to measure the preserved model, remove its demonstrated bottlenecks one at a time, and retain one portable checkpoint contract throughout.

## References

1. *XHToken/Spark-X2.6-1.7B-Base at main*. https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base/tree/main
2. *XHToken/Spark-X2.6-1.7B-Base · Hugging Face*. https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base
3. *Torch shared tensors*. https://huggingface.co/docs/safetensors/torch_shared_tensors
4. *Releases · huggingface/transformers · GitHub*. https://github.com/huggingface/transformers/releases
5. *Releases · pytorch/pytorch · GitHub*. https://github.com/pytorch/pytorch/releases
6. *Stateful DataLoader Tutorial — TorchData main documentation*. https://docs.pytorch.org/data/main/stateful_dataloader_tutorial.html
7. *Troubleshooting & FAQs*. https://unsloth.ai/docs/basics/troubleshooting-and-faqs
8. *GitHub - XHToken/Spark-X2.6: Spark-x2.6 open model series. Pushing the Limits of Agentic Capabilities in On-Device Models · GitHub*. https://github.com/XHToken/Spark-X2.6
9. *Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py at main · linkedin/Liger-Kernel · GitHub*. https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py
10. *Support Matrix – Axolotl*. https://docs.axolotl.ai/docs/support-matrix.html
11. *Releases · pytorch/torchtitan · GitHub*. https://github.com/pytorch/torchtitan/releases
12. *Arguments ¶*. https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html
13. *Releases · linkedin/Liger-Kernel · GitHub*. https://github.com/linkedin/Liger-Kernel/releases
14. *Data Loading Optimization in PyTorch #*. http://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html
15. *Stateful DataLoader — TorchData 0.11.0 documentation*. https://meta-pytorch.org/data/beta/torchdata.stateful_dataloader.html
16. *Distributed Checkpoint - torch.distributed.checkpoint #*. https://docs.pytorch.org/docs/main/distributed.checkpoint.html
17. *Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs*. https://arxiv.org/pdf/2610.26006
18. *Torch.Nn.Functional.Scaled Dot Product Attention*. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
19. *xFormers optimized operators | xFormers 0.0.35 documentation*. https://facebookresearch.github.io/xformers/components/ops.html
20. *Optimizer and Scheduler Configuration — Megatron Bridge*. https://docs.nvidia.com/nemo/megatron-bridge/latest/training/optimizer-scheduler.html
21. *bridge.training.losses — Megatron Bridge*. https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.training.losses.html
22. *torchtitan/torchtitan/components/loss.py at main · pytorch/torchtitan · GitHub*. https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/loss.py
23. *torch.nn.attention.flex_attention #*. https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html
24. *configuration_spark.py · XHToken/Spark-X2.6-1.7B-Base at main*. https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base/blob/main/configuration_spark.py
25. *Universal Checkpointing with DeepSpeed: A Practical Guide - DeepSpeed*. https://www.deepspeed.ai/tutorials/universal-checkpointing
26. *Getting Started - DeepSpeed*. https://www.deepspeed.ai/getting-started
27. *ml-cross-entropy/README.md at main · apple/ml-cross-entropy · GitHub*. https://github.com/apple/ml-cross-entropy/blob/main/README.md
28. *config.json · XHToken/Spark-X2.6-1.7B-Base at main*. https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base/blob/main/config.json
29. *torch.nn.functional.linear_cross_entropy — PyTorch 2.15 documentation*. https://pytorch.org/docs/2.15/generated/torch.nn.functional.linear_cross_entropy.html
30. *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*. https://pytorch.org/blog/flexattention
31. *A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models*. https://arxiv.org/html/2509.22536
32. *GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention · GitHub*. https://github.com/dao-ailab/flash-attention
33. *Trainer*. https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
34. *SFT Trainer*. https://huggingface.co/docs/trl/en/sft_trainer
35. *Balancing Speed and Stability: The Trade-offs of FP8 vs. BF16 Training in LLMs*. https://arxiv.org/html/2411.08719v1
36. *Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in Supervised Fine-tuning*. https://aclanthology.org/2025.findings-acl.256.pdf
37. *Enhancing Training Efficiency Using Packing with Flash Attention*. https://arxiv.org/html/2407.09105v1
38. *ml-cross-entropy/cut_cross_entropy/cce.py at main · apple/ml-cross-entropy · GitHub*. https://github.com/apple/ml-cross-entropy/blob/main/cut_cross_entropy/cce.py
39. *modeling_spark.py · XHToken/Spark-X2.6-1.7B-Base at main*. https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base/blob/main/modeling_spark.py
40. *Current and New Activation Checkpointing Techniques in PyTorch*. https://pytorch.org/blog/activation-checkpointing-techniques/
41. *GitHub - pytorch/ao: PyTorch native quantization for training and inference · GitHub*. https://github.com/pytorch/ao
42. *Troubleshooting*. https://docs.nvidia.com/nemo/automodel/latest/model-coverage/troubleshooting
43. *lm-evaluation-harness/docs/task_guide.md at main · EleutherAI/lm-evaluation-harness · GitHub*. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
44. *Model Coverage Overview*. https://docs.nvidia.com/nemo/automodel/latest/model-coverage/overview
45. *Scoring & Metrics ¶*. https://lm-evaluation-harness.readthedocs.io/writing_tasks/scoring_and_metrics
46. *Contribute a New Model to Megatron Bridge — Megatron Bridge*. https://docs.nvidia.com/nemo/megatron-bridge/latest/adding-new-models.html
47. *torch.distributed.fsdp.fully_shard #*. https://docs.pytorch.org/docs/main/distributed/fsdp.fully_shard.html
48. *Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.15.0+cu132 documentation*. https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
49. *Fused Linear and Cross-Entropy Loss `torch.nn.functional.linear_cross_entropy` · Issue #124480 · pytorch/pytorch · GitHub*. https://github.com/pytorch/pytorch/issues/124480
50. *GitHub - Apsod/FlashCE: Flash-Cross-Entropy, using the same overarching ideas as FlashAttention (mapreduce/semigroup-folds)*. https://github.com/Apsod/FlashCE
51. *In-Place Indic Tokenizer Expansion and Continual Pretraining of Nemotron-3-Nano-30B-A3B*. https://arxiv.org/pdf/2610.15373
52. *Releases · huggingface/trl · GitHub*. https://github.com/huggingface/trl/releases
53. *Releases · axolotl-ai-cloud/axolotl · GitHub*. https://github.com/axolotl-ai-cloud/axolotl/releases
54. *Releases · hiyouga/LlamaFactory · GitHub*. https://github.com/hiyouga/LlamaFactory/releases
55. *xformers/xformers/ops/fmha/attn_bias.py at main · facebookresearch/xformers · GitHub*. https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/attn_bias.py
56. *Releases · Lightning-AI/litgpt · GitHub*. https://github.com/Lightning-AI/litgpt/releases
57. *cut-cross-entropy · PyPI*. https://pypi.org/project/cut-cross-entropy/
58. *Releases · pytorch/ao · GitHub*. https://github.com/pytorch/ao/releases
59. *unsloth · PyPI*. https://pypi.org/project/unsloth/
60. *torchtitan/README.md at main · pytorch/torchtitan · GitHub*. https://github.com/pytorch/torchtitan/blob/main/README.md
61. *Releases · NVIDIA-NeMo/Automodel · GitHub*. https://github.com/NVIDIA-NeMo/AutoModel/releases
62. *Releases · facebookresearch/xformers · GitHub*. https://github.com/facebookresearch/xformers/releases
63. *Releases · EleutherAI/lm-evaluation-harness · GitHub*. https://github.com/EleutherAI/lm-evaluation-harness/releases
64. *Releases · Dao-AILab/flash-attention · GitHub*. https://github.com/Dao-AILab/flash-attention/releases
65. *Releases · deepspeedai/DeepSpeed · GitHub*. https://github.com/deepspeedai/DeepSpeed/releases
66. *Releases · NVIDIA/Megatron-LM · GitHub*. https://github.com/NVIDIA/Megatron-LM/releases
67. *lm-evaluation-harness/lm_eval/models/huggingface.py at main · EleutherAI/lm-evaluation-harness · GitHub*. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
68. *Releases · meta-pytorch/data · GitHub*. https://github.com/meta-pytorch/data/releases
69. *torchtitan*. https://huggingface.co/docs/transformers/community_integrations/torchtitan
70. *flash-attention/hopper/flash_attn_interface.py at main · Dao-AILab/flash-attention · GitHub*. https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py
71. *litgpt/litgpt/scripts/convert_hf_checkpoint.py at main · Lightning-AI/litgpt · GitHub*. https://github.com/Lightning-AI/litgpt/blob/main/litgpt/scripts/convert_hf_checkpoint.py
72. *unsloth/unsloth/models/loader.py at main · unslothai/unsloth · GitHub*. https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py
73. *PyTorch 2.15 Release Blog*. https://pytorch.org/blog/pytorch-2-15-release-blog
74. *pypi.org*. https://pypi.org/rss/project/transformers/releases.xml
75. *torchtitan/torchtitan/components/lr_scheduler.py at main · pytorch/torchtitan · GitHub*. https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/lr_scheduler.py
76. *Zero Redundancy Optimizer - DeepSpeed*. https://www.deepspeed.ai/tutorials/zero/
77. *core.rerun_state_machine — Megatron Core*. https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.rerun_state_machine.html
78. *Distributed Checkpoint - torch.distributed.checkpoint #*. https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html
79. *E S P C - : A L L M I P*. https://arxiv.org/pdf/2107.02027
80. *Liger-Kernel/src/liger_kernel/transformers/fused_linear_cross_entropy.py at v0.8.4 · linkedin/Liger-Kernel · GitHub*. https://github.com/linkedin/Liger-Kernel/blob/v0.8.4/src/liger_kernel/transformers/fused_linear_cross_entropy.py
81. *flash-attention/flash_attn/losses/cross_entropy.py at main · Dao-AILab/flash-attention · GitHub*. https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/losses/cross_entropy.py
82. *Sequence Packing and Dynamic Batching — NeMo-RL*. https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html
83. *Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py at v0.8.4 · linkedin/Liger-Kernel · GitHub*. https://github.com/linkedin/Liger-Kernel/blob/v0.8.4/src/liger_kernel/ops/fused_linear_cross_entropy.py
84. *Indic-LLM-Eval-Suite*. https://ai4bharat.github.io/indic-llm-eval-suite-blog
85. *torch.nn.functional.scaled_dot_product_attention — PyTorch 2.15 documentation*. https://pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html
86. *lm-evaluation-harness/lm_eval/api/task.py at main · EleutherAI/lm-evaluation-harness · GitHub*. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py
