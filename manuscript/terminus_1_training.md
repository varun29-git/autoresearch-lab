# Terminus 1: the training procedure

This is an account of the code that trains the selected Terminus 1 model and of the code that built the tokens it reads. It follows a source row from the raw file, through the frozen encoder, the selection ledger, the quota cut, the packer, the mixture, and the manifest, and then through the weights, the loss, the optimizer step, the held-out score, and the checkpoint. It describes the working tree that is intended for the one-trillion-token run. That tree is not yet commit `75fca4a`. The last commit does not contain the validation cadence, the mixture-weighted validation loss, the development-choice scorer, or the removal of the license and deduplication gate.

The paper does not claim a measured MI325X throughput, a finished token budget, a dense-to-QAT quality transfer, or a production loss. Those have not been run. Numbers that come from the assembled manifest or from `production/amd/CONFIG_SELECTION.md` are planning figures and are labeled as such.

The archived H100 campaign controllers (`production/h100_24h`, `production/kimi_h100_5b`) are not this training loop. The data path that is this run is `production/native_corpus.py`, `production/packing_worker.py`, `production/document_packing.py`, `production/packed_mixture.py`, `production/translation_pairs.py`, and the manifest binder the trainer loads.

## 1. What is being trained

The selected dense model has 1,446,478,126 trainable parameters. The geometry is fixed in `production/recipe.py` as `SELECTED_MODEL` and narrowed for the AMD node by `production/amd/recipe.py::model_for`.

| Quantity | Value |
| --- | --- |
| Layers | 24 |
| Width | 2304 |
| FFN width | 5760 |
| Query heads / KV heads / head dimension | 18 / 6 / 128 |
| Context | 4096 |
| Local window, including the current token | 512 |
| Global layers, one-based | 6, 12, 18, 24 |
| Vocabulary | 65,536, tied embedding and readout |
| Residual stream dtype | BF16 |
| Weight mode for the first phase | dense |
| Branch norm | Peri-LN |
| Value residual | learned |
| Q/K norm | learned RMS, one gain per head channel |
| Activation | SwiGLU |
| Attention on the MI325X launch | `rocm_aiter` for every layer |
| Linear on the MI325X launch | `rocm_fp8_te` |
| Loss | streaming token cross-entropy, Triton, chunk 4096, no stored logits |
| Dropout | 0 |
| Gradient checkpointing | off |
| `torch.compile` and CUDA graphs | off |

A 25-layer checkpoint cannot be loaded into this model. `production/ARCHITECTURE.md` records the only reviewed change from that baseline: layer count, the global-layer period, and the local window. Width, the residual initialization scale, and the optimizer family were kept.

The AMD launcher refuses any model or optimizer that is not exactly this candidate. Dense training uses spectral Muon at 0.017 and tied-embedding AdamW at 0.001. The QAT rates (Muon 0.0085, embedding 0.003, norms 0.00015, Q/K norms 0.00005) are a later phase. They are not the dense run.

The balanced launch is one node, eight-way data parallel, microbatch 16, accumulation 4, global batch 2,097,152 tokens, DDP bucket 25 MiB, fused FP8 producers off. If microbatch 16 does not fit, the same global batch is restarted at microbatch 8 and accumulation 8.

## 2. Initialization

Construction happens in `prepare_training` after every rank has executed `random.seed(1337)`, `np.random.seed(1337)`, and `torch.manual_seed(1337)`. `BitNetLlama` is then built and moved to the device. Because the seed is the same on every rank, the initial tensors match before `DistributedDataParallel` broadcasts rank 0. After a fresh start, each rank then calls `torch.manual_seed(1337 + rank)`. Dropout is zero, so that second seed does not change the forward. It does separate any later CUDA RNG use. A resume restores the per-rank RNG from the checkpoint instead.

There is no bias on any projection. `nn.Linear` is used only if an embedding bottleneck is configured. The selected model sets neither `embedding_dim` nor `use_embedding_projection`, so `embed_rot_in` and `embed_rot_out` are `nn.Identity`.

### 2.1 Matrix weights

Every `BitLinear` allocates `weight` of shape `[out_features, in_features]` with the default parameter dtype, which is FP32. The selected initialization is `weight_init="normal"`:

```text
W ~ Normal(0, 1 / sqrt(in_features))
W <- W * init_scale
```

`init_scale` is 1 for QKV and for the packed gate-up projection. It is `residual_init_scale = 0.14078571632817444` for the two residual writes: attention `o_proj` and FFN `down_proj`. That scale is applied in place under `torch.no_grad` immediately after the normal draw. Orthogonal initialization exists in the same constructor and is not selected.

The selected projections, and the fan-in that sets the normal standard deviation, are:

| Parameter | Shape | Fan-in | Scale |
| --- | --- | --- | --- |
| `qkv_proj` | `[2304 + 2 * 768, 2304]` = `[3840, 2304]` | 2304 | 1 |
| `o_proj` | `[2304, 2304]` | 2304 | 0.14078571632817444 |
| `gate_up_proj` | `[11520, 2304]` | 2304 | 1 |
| `down_proj` | `[2304, 5760]` | 5760 | 0.14078571632817444 |

`qkv_proj` is one packed matrix because `fuse_projections` is on. Its rows are query (2304), key (768), value (768). `gate_up_proj` is gate (5760) then up (5760). The Muon step later splits those packed matrices back into logical slices. The initialization does not. One normal draw covers the whole packed matrix.

Each `BitLinear` also stores a buffer `quantization_strength = 1`. The dense stage is not allowed to anneal it. `set_quantization_strength_` rejects any value other than 1 on a native FP8 backend.

### 2.2 Embedding and norms

The token embedding is `nn.Embedding(65536, 2304, padding_idx=3)`, drawn from `Normal(0, 1/sqrt(2304))`. The padding row, id 3, is then set to zero and stays zero because `padding_idx` is set. The same matrix is the readout. There is no second copy.

RMSNorm gains start at one. With Peri-LN and learned Q/K norms, the gain tensors are:

- `attn_norm.weight` and `ffn_norm.weight`: length 2304, channel gain
- `attn_out_norm.weight` and `ffn_out_norm.weight`: length 2304, channel gain
- `q_norm.weight` and `k_norm.weight`: length 128, one gain per head channel, shared across heads
- `norm.weight`: length 2304, the final norm

`use_subln` is false, so the scalar attention and FFN sub-layer norms are `Identity`. Their parameters do not exist.

From layer index 1 onward, each attention module owns `value_lambda`, a length-2 parameter initialized to `[0.5, 0.5]`. Layer 0 has no lambda. It produces the value tensor that later layers mix.

### 2.3 What is not initialized

Optimizer state does not exist until the first `step`. Muon matrices receive a zero `momentum_buffer` of the same shape and dtype as the weight. Adam groups receive zero `exp_avg` and `exp_avg_sq` and a step counter. The first update therefore sees a zero momentum and a bias-corrected Adam step of 1.

FP8 weight caches are empty. They are filled on the first forward and cleared after every optimizer step, because the latent FP32 weight has changed.

## 3. One forward pass

`complete_update` calls the model with `return_logits=False` and, for packed data, a `document_layout`. Autocast is BF16. The loss kernel disables autocast internally and does its own casts.

### 3.1 Embedding and residual dtype

```text
h = tok_embeddings(input_ids)          # FP32 parameter, looked up
h = h.to(bfloat16)                     # residual_dtype == "bf16"
```

The residual stream stays BF16 for the rest of the stack. RMSNorm computes in FP32 and casts back to the input dtype, which is BF16. RoPE sine and cosine are computed in FP32 with autocast disabled, then stored in the residual dtype.

### 3.2 Rotary positions

`RotaryEmbedding` uses θ = 10,000 and dimension 128. For a position p and a pair index i:

```text
angle(p, i) = p * 10000^(-2i / 128)
```

The rotation is the half-split form, not the interleaved form. If a head vector is `[first, second]` along the last axis:

```text
[first * cos - second * sin, second * cos + first * sin]
```

On a packed batch the positions are not `0 .. length-1` of the physical row. `document_layout.positions` indexes the precomputed tables, so each document restarts at position 0. A position past 4095 is rejected. Padding positions are zero and are not attended.

### 3.3 The block

Peri-LN is a norm on the branch output before it is added back. With fused FP8 producers off, which is the balanced launch, `BitNetBlock.forward` is:

```text
h = h + AttnOutNorm(Attention(AttnNorm(h)))
h = h + FfnOutNorm(FeedForward(FfnNorm(h)))
```

`AttnNorm` and `FfnNorm` are the pre-norms. `AttnOutNorm` and `FfnOutNorm` are the Peri norms. Each is

```text
y = x * rsqrt(mean(x^2) + 1e-5) * gain
```

with the mean over the last dimension, the arithmetic in FP32, and the result cast back to BF16.

### 3.4 Attention

Because `fused_qkv` is on, the QKV projection is one GEMM. `packed_qkv_rope` then splits the rows, applies learned Q/K RMSNorm, and applies RoPE. The GEMM itself is `production.amd.backend.linear` with backend `rocm_fp8_te`.

The weight that enters the GEMM is not the FP32 parameter. `_weight_operands` quantizes the current dense weight to FP8 E4M3 with blockwise 128 scales, through the ROCm quantizer, and caches the codes until the parameter version changes. Dense mode does not round the latent weight to {-1, 0, +1}. Ternary rounding exists on the same module and is the QAT phase.

Activations are quantized as part of that FP8 recipe. `quantization_strength` is 1, so the straight-through residual `x + 1 * (dequant(x) - x)` is the quantized activation. The latent weight remains the FP32 parameter that Muon updates.

Query shape is `[batch, 18, length, 128]`. Key and value are `[batch, 6, length, 128]`. Grouped-query attention repeats each KV head across three query heads inside the attention kernel.

Layers whose one-based index is divisible by 6 are global: the causal window is the whole segment, capped by the 4096 context. The other twenty layers are local. A query at position p may attend to keys q where

```text
0 <= p - q < 512
```

inside the same document. The window includes the current token, so the width is 512, not 511. `local_attention_backend` is empty on the balanced launch, so local and global layers both call `rocm_aiter`. The window argument is the only difference.

If a document layout is present, attention is self-attention over the unpadded segments. `DocumentLayout` requires contiguous segments numbered from zero, positions `0, 1, 2, ...` inside each segment, and a tail of segment id -1 with position 0 for padding. The cumulative lengths are int32. The reference implementation, used only when the backend is `sdpa_reference`, loops over spans and runs scaled dot-product attention inside each span. Production on this launch does not take that path. It calls the ROCm variable-length kernel with the same layout. A boolean padding mask is rejected unless the backend is the reference. Packed training never builds one.

### 3.5 Learned value residual

Layer 0's value tensor, after projection and before the output projection, is `V1`. It is saved, made contiguous, and the module attribute is cleared so the activation is not retained on the module between updates. For every later layer, before the cache window and before attention:

```text
V = λ0 * V1 + λ1 * V_layer
```

`λ` is that layer's `value_lambda`. The mix is the fused `value_mix` kernel when the value tensor is on CUDA and fused pointwise is on. `V1` is the layer-0 value at the same new positions, not the mixed value of the previous layer. Only layer 0 is the source. Later layers do not feed their mixed values forward. That is why a checkpointed layer 0 would be wrong: a checkpointed stash would be a no-gradient tensor, and layer 0's value projection would stop learning. Gradient checkpointing is off in this launch, and the code still refuses to checkpoint layer 0 if it were turned on.

### 3.6 The feed-forward

`fused_mlp` is on and fused FP8 producers are off, so the FFN is one packed gate-up GEMM, then `packed_swiglu_norm`, then the down projection:

```text
[gate, up] = gate_up_proj(FfnNorm(h))
mid = packed_swiglu_norm(gate, up)     # SiLU(gate) * up, no extra sub-layer norm
h = h + FfnOutNorm(down_proj(mid))
```

`ffn_sub_norm` is `Identity` because `use_subln` is false, so `packed_swiglu_norm` is called with a null gain. The down projection is the second residual-scaled matrix. ReLU² exists in the same module and is not selected.

### 3.7 The head

After the last block, the final RMSNorm runs and the tied embedding is the classifier. Training asks for no logits. The loss is `streaming_cross_entropy_sum` with backend `triton` and chunk 4096, because `loss_backend` on the selected model is `triton`.

The hidden states and the FP32 embedding are cast to the autocast dtype, BF16. The sequence is flattened to rows. The kernel walks rows in chunks of 4096. For each chunk it forms logits with a matrix multiply against the tied weight, then a fused cross-entropy that writes the loss and the input gradient in place. The gradient of the hidden state is `softmax(logits) - one_hot(target)`, with ignored labels zeroed, divided by the number of valid targets in the whole sequence, not in the chunk. The gradient of the embedding is accumulated in FP32 with `addmm`. The chunk does not keep the `[chunk, 65536]` logit tensor for a later backward. Backward of this autograd function only multiplies the saved hidden gradient and weight gradient by the upstream scalar.

Ignored targets are -100. They contribute neither to the sum nor to the normalizer, except that the normalizer is at least 1 so an empty row does not divide by zero. A row whose every target is -100 still has to be rejected by the trainer, which requires the global count of non-ignored targets to equal the planned count.

The returned `loss` is the mean. The trainer does not use it. It uses `loss_sum`, the unnormalized sum, and rescales that itself.

## 4. The optimizer

`build_optimizer` partitions every trainable parameter into exactly one group. A parameter left out, or placed twice, is an error.

Muon groups, one parameter each, are every `BitLinear` weight. The selected rule is `spectral_unclamped`. For that rule the learning-rate scale returned by `muon_update_scale_for_param` is 1, and the weight decay stored on the group is 0.02. The aspect ratio is applied later, inside the step, per logical slice:

```text
shape_scale = sqrt(logical_rows / fan_in)
W <- W * (1 - lr * 0.02) - lr * shape_scale * NewtonSchulz5(direction)
```

There is no Moonshot width-growing factor and no clamp. Packed QKV is three logical matrices of row counts `(2304, 768, 768)`. Packed gate-up is two of `(5760, 5760)`. Every other Muon matrix is one slice, the whole parameter. Each slice has its own momentum view and its own `shape_scale`. The momentum buffer itself is one tensor the size of the packed parameter, updated as `lerp`.

The direction is Nesterov. With β = 0.95:

```text
momentum <- lerp(momentum, grad, 1 - β)
direction <- lerp(grad, momentum, β)
update <- zeropower_via_newtonschulz5(direction, steps=5)
```

The Newton–Schulz iteration is the pinned `muon` package at commit `f98f1cacc0263b04290753e32be8d498c1efc806`. A different installed commit raises before the optimizer is constructed. Same-shaped slices are stacked for one Newton–Schulz call. The code treats that stack as equal to per-slice calls up to floating-point reduction order.

The Muon learning rate on every matrix is 0.017. Role-specific overrides exist in `OptimizerConfig` and are all null in the selected optimizer, so QKV, output, gate-up, and down share 0.017. The spectral coefficient then makes a wide slice take a larger step than a tall slice.

AdamW, with `use_muon` false, covers everything else. Betas are `(0.9, 0.95)`, epsilon `1e-8`. The update is the ordinary bias-corrected AdamW step, implemented with foreach kernels:

| Group | Parameters | Learning rate | Weight decay |
| --- | --- | --- | --- |
| Tied embedding | `tok_embeddings.weight` | 0.001 | 0 |
| Input norms | pre-norms, channel vectors that are not otherwise named | 0.0003 | 0 |
| Peri norms | `attn_out_norm`, `ffn_out_norm` | 0.0003 | 0 |
| Final norm | `norm.weight` | 0.0003 | 0 |
| Value mix | each `value_lambda` | 0.0003 | 0 |
| Q/K norms | `q_norm`, `k_norm` | 0.0001 | 0 |

The historical `SELECTED_OPTIMIZER` embedding rate is 0.003. The AMD dense candidate replaces it with 0.001. That is `0.003 * 768 / 2304`, the width transfer already applied in `production/amd/recipe.py`. No further scaling is applied at launch. QAT uses a different table and does not apply during the dense phase.

Weight decay is decoupled: the weight is multiplied by `1 - lr * decay` and the update is subtracted. Embedding decay is zero, so the tied matrix is not decayed. Norms and lambdas are not decayed.

After `optimizer.step`, `clear_fp8_weight_cache` drops every cached FP8 weight and the tied-head cache. The next forward requantizes.

## 5. One update

`PreparedTraining.complete_update` is one optimizer step. `cadence` is everything after it and is not part of the measured step.

### 5.1 The token arithmetic

For a packed manifest the batcher plans the update before any forward. Each rank lists the sequence indices it owns:

```text
global_start = micro_step * microbatch * world_size
index = global_start + i * world_size + rank
```

`plan_update` sums, from the descriptors, `valid_targets`, `content_tokens`, and `sequences * 4096` for those indices. All-reduce sums the ranks. The three sums are the planned labels, the planned content, and the planned input slots. The optimizer clock advances by the planned labels. If that count is zero, or if it would walk past the sealed scored horizon, the update is refused before the forward.

The loss on each rank is

```text
loss = local_loss_sum * (world_size / planned_labels)
loss.backward()
```

Data parallel averages gradients, so the factor `world_size` puts the mean back on the global label count. The last microstep synchronizes. Earlier microsteps run under `no_sync`, so the bucket all-reduce happens once per update.

After the microsteps, the ranks all-reduce the loss sum, a stop flag, and the count of targets that were not -100. The update is applied only if that count equals the planned label count. A padded tail is legal only when the descriptors said those positions were ignored. A mismatch writes `incidents.jsonl` and does not call `optimizer.step`. The same refusal fires if the last update of the manifest does not land exactly on the sealed scored total.

### 5.2 The schedule

The schedule is warmup-stable-decay. The clock argument is the scored labels after this update, minus `phase_start_tokens`. Dense training has a zero phase offset. Warmup is `160 * 6,291,456 = 1,006,632,960` scored labels, which is 480 updates at the 2,097,152 global batch. During warmup the multiplier is `tokens / warmup`, linear from just above zero to 1.

Decay length is computed by the assembler as

```text
decay = ((dense_input_examples + 9) // 10) * 4096
```

which is 81,743,212,544. Decay starts when the scored clock reaches `dense_scored_labels - decay`. The assembled dense scored total is 800,000,317,356, so decay starts at 718,257,104,812. Through the stable region the multiplier is 1. Inside the decay window, with cosine style and final fraction 0.1:

```text
progress = min(1, (tokens - decay_start) / decay)
multiplier = 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))
```

Every group learning rate is `base_lr * multiplier`. Base rates are the values in section 4. Clipping is global `clip_grad_norm_` at 1.0, foreach, and non-finite gradients raise. The coefficient that was applied is recorded as `min(1, 1 / (grad_norm + 1e-6))`.

Quantization warmup is zero. Strength stays 1. A dense configuration that asks for a quantization curriculum is rejected.

### 5.3 Counters

After a successful step the state is:

- `updates` increased by 1
- `next_micro_step` increased by the microsteps just consumed, normally 4
- `tokens_seen` increased by the planned labels
- `input_slots_seen` increased by the planned slots
- `content_tokens_seen` increased by the planned content

These three clocks are not the same number. The batch geometry is input slots. The loss and the learning-rate schedule use scored labels. The mixture accounting uses content tokens. A checkpoint that disagrees with the product of updates, accumulation, and the manifest horizon cannot resume.

### 5.4 Distributed training

Eight processes, one GPU each. The process group is NCCL, which on this stack is RCCL, with a five-minute timeout. The model is wrapped in `DistributedDataParallel` with `broadcast_buffers=False`, `gradient_as_bucket_view=True`, and `bucket_cap_mb=25`. Buffers are not broadcast because the FP8 caches must not be treated as collective state. Ranks gather a fingerprint of the resume contract and refuse to start if the data, source, configuration, or runtime identity differs.

SIGINT and SIGTERM set a flag. The current update finishes. `cadence` then treats the run as stopped, drains the checkpoint, and returns.

## 6. Data engineering

A training token exists only after four programs have agreed. `native_corpus.convert_job` turns one source object into a `uint16` bin and a JSONL index, and it refuses the job unless the encoder identity and the original train/validation flag survive. `packing_worker.Worker` reads those bins in a seeded order, inserts each document's text hash into a SQLite ledger, and stops at an integer quota. `document_packing` cuts the selected ids into 4096-token fragments and packs the fragments. `packed_mixture` then throws away the order in which the files were written and defines the order the eight GPUs will read. The trainer never sees the source text.

### 6.1 Unicode, before any piece is chosen

`terminus/unicode_contract.py` pins the character database at Unicode 16.0.0 and the `regex` package at 2026.7.19. If the standard-library database is older, the process imports `unicodedata2` and refuses any other version. NFC, character categories, and the Indic/Latin classes are therefore the same on the CPU that encoded the corpus and on any later machine that checks the tokenizer. A different Unicode version is not a compatible re-encode.

### 6.2 The frozen encoder

The artifact is `tokenizer-65k-v2.json`, SHA-256 `dc636d96224d059d03619e61f0ecddc3ac1e5c8c6fb46ee051f66cabc65fb35c`. `Frozen65kTokenizer` loads it and checks the spec against the code. A file with the same vocabulary size and a different router, normalizer, or piece layout does not load. The November 6 sealed pack was produced by an older encoder and is not bound to this run.

The id layout is fixed:

| Ids | Contents |
| --- | --- |
| 0 | `<unk>` |
| 1 | `<s>`, a begin token, not a document boundary |
| 2 | `</s>`, a sentence or generation stop, not a document boundary |
| 3 | `<pad>` |
| 4 | `<eod>`, the only document separator the packer inserts |
| 5–31 | reserved names that ordinary text must not emit |
| 32–287 | English byte alphabet, id = 32 + byte |
| 288–543 | byte-fallback pieces `<0x00>` through `<0xFF>` |
| 544–40346 | Indic pieces, plus twelve `<reserved_space_*>` placeholders that are never emitted |
| 40347–65535 | English BPE tail, one piece per merge |

`encode` returns integer ids only. It rejects a non-string. The route is per stretch, not per document:

- A run of tab, newline, carriage return, or two or more spaces is protected whitespace. It is cut out of the surrounding text and encoded on the English path.
- A stretch with Indic letters and no Latin letters is one Indic span, including its digits, punctuation, and single spaces.
- A stretch with Latin letters and no Indic letters, or with neither, is English.
- A stretch with both scripts is split on single spaces. Inside a word, Latin letters and Indic letters form separate runs. Digits and punctuation attach to the run on their left, or to the run on their right if they begin the word. A word made only of symbols is English.
- A single space is pulled into the Indic span only when the runs on both sides are Indic. Otherwise that space is its own English token.
- Format characters take the script of the nearest non-format neighbor, preferring the left neighbor.
- An Indic span that starts with a real space emits that space as its own byte token and does not treat it as the dummy prefix.
- Indic normalization is NFC, with a dummy prefix and the space mark recorded in the artifact. The loader refuses any other triple.

The conversion job calls `encode_batch_bytes`. Each document's bytes are the UTF-8 of the source string. The written record stores `n_tokens = len(data) // 2` and the little-endian `uint16` bytes themselves. Ids 0–31 are not produced by this encoder for ordinary text. The packer later rejects a source array that contains padding, `<eod>`, or any id from 3 through 31. Pad and `<eod>` are inserted only by the packer, after selection.

Staging of a packed manifest reloads this file and requires the manifest's tokenizer hash, vocabulary size, and special ids to match, and it requires the end-of-document id to be 4.

### 6.3 From a source object to a token bin

`convert_job` is one source object, one job id, one tokenizer. It refuses to start if a declared runtime file has changed hash. If the plan says `resume_verified`, a job whose fingerprint is already published as completed is returned as-is and not encoded again.

The source is either Parquet or strict text lines.

Parquet uses the columns named in the job, defaulting to `text`. A missing column rejects the job. The renderer is either one field or `join_nonempty_strings` with an explicit separator and an explicit field order. Empty text rejects the row. Batches are 512 rows, and PyArrow threads are off so the row numbers stay reproducible.

Text lines require a proven 0-based or 1-based line convention. UTF-8 is strict. The first physical line may be decoded as `utf-8-sig`. A reviewed recovery recipe may skip zero, one, or two unselected prefix lines, and it may append a separately hashed tail of at most 1 MiB from a different key. Skipping a line that the index selected is an error.

Identity is decided before encoding. Each index row has a document id. The row key is the source key plus the coordinate, or the id itself when `id_encodes_row` is set. Rows that look like `source:number` are joined by row number. Other rows are joined by UID. Mixing the two schemes in one job is rejected. A UID that lands at two coordinates is rejected. A frozen anchor hash, when the part carries one, must match the SHA-256 of the raw UTF-8.

The original index already says whether the part is validation. That Boolean is copied onto every output row. The converter does not invent a split. `original_holdout_assignments_preserved` is set only after every selected document has been written. A conversion receipt whose validation flag disagrees with the source job is refused by the packer.

Optional comparison against a reference tokenizer is exact on the bytes, or exact after a declared atom-id expansion. A candidate that grows the token count, or that differs beyond that expansion, rejects the job. The published bin hash must also match `expected_output_bin_sha256` when the plan pinned one.

Each output row is one JSON object:

```text
doc_id, row_key, source_text_sha256, validation,
split_family_id, token_start, n_tokens
```

`token_start` is the running sum of `n_tokens` inside that part. The bin is the concatenation of those documents in index order, little-endian `uint16`. The receipt records document count, token count, both hashes, and the original index key. Publication is a multipart S3 upload whose checksum is the composite part hash. The local whole-file SHA-256 is the identity the packer downloads against. The progress object is rewritten at least every 15 seconds, and the verified result is written again under `completed/<job_fingerprint>.json`. Local work files are deleted only after that publication, and only if the plan asked for it.

`source_text_sha256` is SHA-256 of the UTF-8 source string. It is the key the selection ledger uses. Two rows with the same text and different ids are the same document for allocation purposes.

### 6.4 The ledger that decides what may be trained on

`packing_worker.Worker` opens `selected.sqlite` in WAL mode. The table `selected` has primary key `text_sha`, plus `row_key`, `leaf`, and `allocation`. A second insert of the same text hash is an integrity error and is counted as `exact_duplicate_documents`. The duplicate row is not packed. This is exact-text ownership across the whole campaign, not a near-duplicate sketch. The packer does not run MinHash. A manifest field that says `id_exact_minhash_v1` would not be evidence that this loop deduplicated anything, which is why that field is not stamped onto this run.

`drained_outputs` records an index hash once every eligible row in that index has been considered. A later phase that sees the same index hash must see the same bin hash or it raises. Restarting a nonempty ledger without an explicit predecessor plan is refused. A predecessor ledger is downloaded from a pinned S3 object and `PRAGMA quick_check` must return `ok`.

Held-out exclusions are a frozen JSONL file with a pinned hash. A row is excluded when `validation` is anything other than false, or when its document id, row key, or split family hits the exclusion sets. Families are also matched as fingerprints of `['published_id', doc]`, `['explicit', family]`, and `['row', row]`. An excluded document is counted and skipped. It is not inserted into `selected`, so a later phase can still see it as held out rather than as already consumed.

Phases share the ledger. The last phase's suffix is the global frontier, because every earlier selected text hash is already in the table. The QAT finishing supplement and the earlier finishing phase published the same suffix-file prefix; the supplement's ledger file is the larger one, so that ledger is the frontier the validation set was taken past.

The worker above is the reference implementation: one SQLite file, text hash stored as text, row key stored as text. `FastWorker` is the same selection order and the same segment contract, with a different ledger. `CompactLedger` stores the text hash as 32 raw bytes, stores SHA-256 of the row key rather than the row key, and interns leaf and owner strings in a `names` table. A compact predecessor that fits the memory limit is backed up into an in-memory database. A larger one stays in `working-ledger.sqlite`. A non-compact predecessor is copied in batches of 100,000 rows. `ReadOnlyLedger` is allowed only for a fixed materialization that already has a predecessor and is not `reservation_only`. The fast worker's durable outputs are still closed, verified shard uploads. Its scratch directory is not a second corpus.

### 6.5 Quotas, visit order, and the cut

`source_targets` is an integer number of content tokens per leaf. Leaves are processed in sorted order so the owner of a duplicate text is reproducible. Inside a leaf, each stratum's weight is the sum of non-validation token counts in its jobs, or 1 for a fresh native source that has no part list. `integer_split` assigns

```text
quota[k] = total * weight[k] // sum(weights)
```

and hands the remainder, one token at a time, to the keys with the largest remainder, breaking ties by key name. The quotas sum to the target.

Jobs inside a stratum are sorted by `fingerprint([plan.seed, job_id])`. Outputs inside a job are sorted by `fingerprint([plan.seed, index uri])`. Rows inside an index are sorted by `fingerprint([plan.seed, row_key])`. The fingerprint is SHA-256 of canonical JSON: sorted keys, no spaces, no NaN. The physical order of the Parquet file is not the visit order.

For each row, in that order, the worker inserts the text hash. On success it takes `min(n_tokens, target - selected)`. Those tokens are sliced out of the memmap at `token_start` and turned into fragments. If the take is shorter than the document, the remainder is appended to `quota-suffix-reservations.jsonl` with `next_document_offset` and `remaining_content_tokens`. The cut document is in the ledger, so it will not be selected again. Its untaken tail is reserved, not returned to the pool. If the take covers the document, `document_end` is true.

When the quota is filled, the loop stops. Later rows in that index are not visited and the index is not marked drained. Those unvisited rows, with fingerprints strictly after the cut, are the residual. The validation set was built from that residual, not from the quota-suffix tails. The suffix files are the cut document's remainder, on the order of a couple hundred rows. They are not the unused 600 billion tokens.

A stratum that cannot fill its quota records a shortfall. A same-leaf stratum fallback may absorb it up to `stratum_fallback_max_tokens`, preferring a receiver whose domain matches the donor. A source fallback may move the deficit to another leaf in the same language. Forward fallbacks add the deficit to a leaf not yet processed. Backward fallbacks, a receiver earlier in sorted order, run a fresh `process_stratum` against the live ledger. They do not relabel tokens already packed. A cycle, a cross-language fallback, or a backward receiver that itself has a fallback is rejected before any download.

`reservation_only` does not write shards. It writes `protected-records.jsonl` and marks the ledger row `protected:<allocation>`. A later fixed materialization downloads that list and packs exactly those spans. The fixed path checks that the ledger still says `protected:` plus the allocation id, that the span lies inside the original document, and that the row is not in the held-out exclusion file. A fixed source range, when present, must cover the same number of tokens as the leaf quota. That path is how a later phase can be forced to replay a frozen list. The assembled 40B bridge is a different operation: its shards were already published, and the assembler only gives them the same mixture seed as the last ordinary dense tranche. It does not reselect their documents.

The worker's deadline is the plan's `deadline_unix` minus 180 seconds. Past it, the process raises and keeps the shards already published. Each finished shard is uploaded, the local payload is deleted, and a progress object is written to S3. Heartbeats during a stratum are at most every 30 seconds.

### 6.6 How a document becomes a 4096-token row

`fragments` walks the selected ids in steps of 4096. A piece shorter than 4096 that is the true end of the document gets a final id 4, `<eod>`, and no lookahead label. A piece that ends because the next source token did not fit carries that next id as `last_label` and does not write it as an input. A piece that ends because the quota ended, `document_end` false, has no lookahead. The label is masked so the model does not train on a token that belongs to a later phase. Content tokens do not count the `<eod>` and do not count a synthetic separator. The source array is rejected if it is empty, if any id is outside 0–65535, or if it already contains pad, `<eod>`, or a reserved id.

`best_fit` packs those fragments. It sorts them longest-first, with the original index as a tie break, then places each into the smallest open gap that can hold it. The open gaps are a bitset over capacities 1 through 4096. A Python integer's trailing-one bit finds the smallest fitting capacity. Gaps that nothing can fill stay empty. After placement, `random.Random(seed).shuffle` reorders the finished packs. The seed is `fingerprint([plan.seed, stratum, buffer_number])`. The shuffle changes sequence order inside the buffer. It does not change which fragments share a sequence. The training reader does not use this order. Section 6.8 replaces it.

`encode_pack` writes one `uint16` row of length 4096, filled with pad id 3. Segments are contiguous from the start of the row. For each segment the descriptor stores start, end, content tokens, document offset, whether this is the document end, whether `<eod>` is an input, the lookahead label, and the document identity (`doc_id`, `row_key`, `split_family_id`, `source_text_sha256`, and the source bin and index hashes). `valid_targets` counts every position that has a next-token label, including the lookahead. `padding_slots` is the unused tail.

`ShardWriter` appends the row and a one-line JSON descriptor. Sequences 0, 1024, 2048, and so on are decoded immediately and must round-trip. On close it returns the payload hash, the index hash, and, when offsets were requested, a `uint64` table whose entry `i` is the byte offset of descriptor `i`. The trainer's reader requires that table: `sequences + 1` entries, starting at 0, ending at the index size, strictly increasing. A shard without it cannot be finalized into a mixture candidate.

`decode_pack` is the oracle both the writer audit and the GPU batch use. Inside a segment, target `t` is input `t+1`. The last position gets `last_label` when one exists, and stays -100 when the quota cut masked it. Positions restart at 0 inside the segment. Segment ids are 0, 1, 2, … and the pad tail is segment -1. Any disagreement in contiguity, `<eod>` placement, content count, padding, or the number of non-ignored targets raises. `causal_mask` in the same file is a CPU reference for tests. The GPU path never builds that square mask.

Published shard files, per part, are:

| File | What it is |
| --- | --- |
| `*.uint16` | `sequences * 4096` little-endian ids |
| `*.segments.jsonl` | one descriptor per sequence, UTF-8 JSON, no spaces |
| `*.offsets.uint64` | `sequences + 1` little-endian offsets into the JSONL |

The mixture candidate copies those three objects and adds leaf, stratum, plan hash, and the source-content totals. `mixture_candidate` refuses a report whose quotas are not all zero-shortfall, whose physical creation is not finished, or whose component totals disagree with the packed content. The seed of a normal phase is `ci1-packed-mixture-v1/` plus the plan hash. `repeats_sequences` is false. `native_training_accepted` is left false in the candidate itself. The trainer's acceptance is the separate staging receipt.

### 6.7 Translation pairs

A pair is not two documents that happen to sit in the same sequence. `compound_pair` accepts two passages of different languages that share one source lineage: repository, config, file, original id, and the hash of the seed text. Both must score at least 95 under the lineage check. The original family is passed through the same exclusion function as a normal document. Each side's tokens must equal `tokenizer.encode` of that side's text, must match the recorded `n_tokens`, and must contain no id below 32.

The packed form is the left tokens, then `encode('\n\n')`, then the right tokens. That byte string must equal `encode(left_text + '\n\n' + right_text)`. If the join changes a boundary, the pair is rejected. If the join is longer than 4096, the pair is rejected and the two sides stay ordinary documents. A short pair gets a trailing `<eod>`. The separator length is `synthetic_tokens`. It is not charged to either language's content count. The descriptor kind is `unlabeled_translation_pair`, with two components, each carrying its own leaf, stratum, and content tokens. `decode_pack` calls `validate_components` and checks the spans, the separator bytes, and the two content totals. The shard writer's source counters add each component to its own leaf, so a bilingual sequence still fills two language quotas exactly.

`PairBudget` is a single-process accountant. It is not the SQLite ledger. A pair replaces the ordinary occurrences of its two records. Using it from several workers without a single owner would double-allocate.

### 6.8 The order the GPUs actually read

Allocation order and file order are not training order. `PackedMixture` builds a tree. Languages, leaves, and strata are sorted by name. Each list is split at its midpoint, recursively, so the tree is balanced rather than a chain. A stratum leaf holds `(leaf, stratum)` and a sequence count. Every internal node stores `rotation = seed64(seed) mod count`, where `seed64` is the first eight bytes of SHA-256, little-endian.

Routing one index adds that rotation, modulo the node count. Let `before = index * left.count // count`. If `before` increases at the next index, the slot belongs to the left child and the child's local index is `before`. Otherwise the slot belongs to the right child and its local index is `index - before`. Each child therefore receives exactly its own sequence count, spread through the parent, and the map is a bijection. There is no rejection sampling and no dropped tail.

The global index is then blocked by 4096 sequences. Inside a block, `permute` runs six Feistel rounds on a power-of-two half-width and walks until the value falls inside the block. The block key is `seed64(phase_seed + '/block/' + block_index)`. A partial last block is permuted over its real length. Within a stratum, shard files are ordered by SHA-256 of `seed + '/' + payload_sha256`, then by URI. The ordinal inside a shard is permuted again with `seed64(seed + '/' + payload_sha256)`. Two runs with the same shards and the same seed read the same rows. A changed seed is a different experiment, and it changes the resume contract because the phase seed is inside the manifest hash.

`PackedSequenceReader` memory-maps the three files. It checks size, and either the staging receipt's inode tuple or a fresh SHA-256 if no receipt was passed. The offset table is checked for length and bounds. Up to 128 shards stay open. `describe` reads only the JSON slice, which is what `plan_update` uses. `sample` copies the `uint16` row, runs `decode_pack`, and returns inputs, targets, positions, and segment ids. Content from a translation pair is attributed to the component leaves. Content from an ordinary segment is attributed to the shard's own leaf.

On the GPU, `PackedTokenBatcher` gives rank `r` the indexes `step * microbatch * world + i * world + r`. The rows are stacked as `int64` and the positions and segment ids become a `DocumentLayout` on CPU, then one copy moves them to the device. A rank past the end of the horizon, which is only a short tail, gets pad inputs and an all-ignored target row. The global check that valid targets equal the planned count still has to pass, so that tail cannot change the loss. Prefetch is a side thread. The cursor on the batch must equal the trainer's microstep. A skip or a repeat is a hard error.

### 6.9 The trillion that was assembled

The assembler walks the published phase receipts in a fixed order. It checks each receipt hash, requires the atom tokenizer, and sums content. The sum must be exactly 1,000,000,000,000. Dense content must be 800,000,000,000 and QAT content 200,000,000,000. There are 10,163 training shards. The `uint16` payloads are 1.86 TiB. Input slots are larger than content because of padding and inserted markers: 817,432,117,248 dense slots and 205,671,276,544 QAT slots. Scored labels, which drop the ignored pad positions, are 800,000,317,356 and 199,999,999,453.

Two merges are done in the manifest and not by repacking. The last ordinary dense tranche and the 40B bridge share one mixture seed, `last-dense-tranche-bridge-v1`, under the phase id `dense-stable-15-with-core-bridge`. Shard bytes and source plan hashes stay the published ones. The Telugu-only HQ supplement is folded into QAT finishing-02 under `qat-finishing-hq-v1`, so the supplement tokens stay in the finishing window and every QAT phase still contains all eight languages. The read order is warmup-50B, dense-stable-01 through the bridged dense-15, then qat-stable-01, qat-stable-02, qat-finishing-01, and the merged finishing phase.

The language weights stored for the validation loss are the integer content targets of that mixture. They sum to 1e12: English 500,000,000,000; Hindi 95,500,000,000; Bengali and Gujarati 68,166,666,666 each; Tamil 68,166,666,667; Kannada, Marathi, and Telugu 66,666,666,667 each. An equal-language mean would give each language an eighth. That is not the recorded mixture loss.

The English code cell has no shards in this assembly. Nothing in the packer invents it.

`stream_spec` is the trainer's view of the same object. Packing must be `document_segments_v1`. The stream schema is 1. Train and validation phase lists are non-empty. Each phase schema is `ci1_packed_mixture_candidate_v1` with `repeats_sequences` false. A continuous `source_mix` or `phase_source_mixes` on a packed manifest is rejected, because the tree above already is the mixture. Sequence length is 4096. Every artifact URI has a path and a 64-character hash. The sum of training sequences times 4096 equals the manifest input-slot horizon. `tokens_before` is zero on the dense manifest and equal to the dense scored total on the QAT manifest.

### 6.10 The validation residual

The validation phase is `extension-residual-validation`. It was not cut out of the training shards, and the training shards were not repacked. It was packed from documents the mass tokenizer had already written and the quota loop had never visited.

For each language the frontier is the maximum fingerprint that the finishing supplement actually cut, inside one source bin. Documents in that bin whose fingerprint is strictly greater were not inserted into `selected`. The cut document itself, and its `source_text_sha256`, stays out. The bins used are the atom-tokenizer outputs: English from the UltraData-Math L3 textbook train bin under warmup-50B; Hindi, Bengali, Gujarati, Marathi, and Telugu from candidate-output bins; Tamil from the heavy-campaign native-script bin; Kannada from its first tail plus a second bin, because the first unused tail was only 1,084,614 tokens. The recorded selection read about 4.8 million unused content tokens per language and then kept the first 1,025 packed sequences. That is 8,200 sequences and 33,587,200 token slots. Packed content is 31,091,802 because of padding. Telugu is the sparsest, 3,163,673 content tokens and 1,033,651 pad slots. The other 241 million held-out tokens were not packed and are not scored.

The phase plan hash is `89bfc50af81fc01687e69ba2d94a8a3d760cd0d1c6b4bad8cd939b7b7ede343f`. Its mixture seed is `ci1-packed-mixture-v1/extension-residual-v1/` plus that hash. One shard per language, leaf `{language}/extension-residual`, 1025 sequences, payload 8,396,800 bytes. The role is development. The final-role split stays sealed and is not on this phase. The same phase object is attached to the dense manifest and the QAT manifest. Both point at the same payloads, so a payload hash cannot appear in train and validation.

Unvisited exact-duplicate copies of a text that was selected under a different row key are a residual leakage that the 24 GB ledger was not scanned for. The paper does not claim that scan.

### 6.11 What this pipeline does not establish

The packer writes whatever quota it was given. It does not check that a source's license was recorded, and the production checker no longer requires that field. It does not run a near-duplicate pass. It does not compare the packed text with the benchmark prompts. The benchmark overlap audit is open. CC-Code is missing. A `document_dedup` label on an older packer manifest is not a receipt for this corpus.

Staging, when the shards are on the training disk, is the last data gate. It decodes every descriptor, rejects reserved ids, checks that source-content totals match the phase, and refuses a payload hash shared by two shards. Until that receipt exists, the manifest is a map of S3 objects, not a training snapshot.

## 7. Validation and the choice tasks

Every 960 updates, which is 2,013,265,920 scored labels at the balanced batch, `cadence` evaluates and then checkpoints. The same interval is the checkpoint. Logging is every 12 updates. The stability probe, when enabled, is every 96 updates. Smoke runs force the validation interval to 10 and are not a resume point for the long schedule.

`evaluate_packed` builds a reader per language from the validation phase only. It takes up to 1,024 sequences per language, strided across the 1,025 packed sequences, and splits them by rank. Each sequence is a normal forward with `return_logits=False`. The ranks all-reduce the loss sum and the token count. The reported loss is the sum divided by the count. Perplexity is `exp(loss)` when the loss is below 700.

If every validation phase carries `language_loss_weight_tokens` and those keys match the evaluated languages, the result also contains `mixture`: the weighted mean of the language losses. The weights on this run are the packed mixture's integer token counts, which sum to 1,000,000,000,000. `cadence` lifts that mixture object out of the language map and writes it beside `languages` on the validation event. An equal-language mean is not what is recorded.

When `benchmark_tokens` is set, and on the assembled config it points at the frozen choice-token directory, every rank calls `score_frozen_choices`. The loader keeps rows whose kind is `choice`, whose cadence is every validation stop, and whose role is `development`. Final rows and reference-continuation tasks are not returned. The prompt ids and choice ids are used as stored. The text is not encoded again. Each choice is a masked continuation: the last prompt position predicts the first choice token, earlier prompt positions are -100, and a prompt that would overflow 4096 is truncated from the left. The correct choice is the one with the lowest loss. Ties take the lower index. The margin is the best wrong choice minus the correct choice. An all-reduce sums correct counts, item counts, and margins. Rank 0 writes the accuracy and the mean margin on the same validation event.

The development set that loader returns from the prepared directory is 27,247 items: MMLU, HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, PIQA, BoolQ, Belebele in all eight languages, Global-MMLU for English, Hindi, Bengali, and Telugu, and INCLUDE for Hindi, Bengali, Tamil, and Telugu. Tasks that require the model to write an answer are tokenized and are not scored on this stop.

## 8. Checkpoints

A checkpoint is `step_XXXXXXXX.pt` plus a JSON receipt and `latest.json`. The payload holds schema version 1, the resume contract, the counter state, the model `state_dict`, the optimizer `state_dict`, and one RNG state per rank. Model and optimizer tensors are checked for finite values before the file is published. The write goes to a temporary file, is fsynced, hashed, and then renamed. `latest.json` is not allowed to move backward. An existing step file is not overwritten.

The resume contract binds the model config, the training config with the operational fields removed, the optimizer config, the manifest report without the node-local file table, the world size, the source hashes, and the runtime. Operational fields that may change without breaking resume are the intervals, the benchmark path, prefetch, the time limit, and the async-checkpoint switch. Changing the model, the data, the world size, or the Muon scaling is a different run.

Async checkpointing is on in the AMD config. Rank 0 copies the payload to CPU synchronously, then a background thread serializes and hashes it. The background thread never reads a live GPU tensor. One snapshot may be pending. At a stop or at the end of the run the writer is flushed before success is reported. A background failure is broadcast and becomes a run failure.

The operations layer, when the human passes `--operations-config`, uploads each committed file to a dedicated S3 prefix. The upload checks the whole-file SHA-256. S3 multipart checksums are a hash of the parts and are not used as that digest. A local file is deleted only after the remote payload, receipt, and completion marker verify, and only if it is outside the last three checkpoints and is not the latest. Remote objects are not deleted by this pruner. Restoring a checkpoint checks the marker, the run contract, the hash, and the counters before it writes a new `latest.json`.

## 9. The end of dense training, and QAT

Dense training stops when `tokens_seen` equals the manifest's scored total, or when a signal or a stop-after-updates bound asks `cadence` to commit and return. The final event is `complete` or `stopped`.

QAT is not a flag flip inside the dense process. `terminus/transition.py` loads a dense checkpoint, checks that its contract matches the parent run, and writes a child checkpoint whose model differs only by `weight_mode` and `quantization_strength`. Any other architectural change is rejected. The child config must set `phase_start_tokens` to the parent's `tokens_seen` unless the policy is an explicit restart. The AMD launcher refuses a ternary model without that offset, and it refuses a dense model with one.

Carrying the optimizer allows learning-rate changes and nothing else. The QAT table changes Muon, AdamW, norm, and Q/K rates. Those are `_lr` keys, so the state can be carried. A change to betas, momentum, or weight decay would require an explicit optimizer reset.

On the child, `quantized_weight` is no longer the latent FP32 matrix. It is the straight-through combination of the latent weight and the ternary target `scale * round(W / scale)` clamped to {-1, 0, +1}. Strength is 1 from the first QAT step. The FP8 GEMM then sees that ternary-directed weight. The dense checkpoint has to be kept until that phase has been accepted. A successful kernel preflight is not that acceptance.

The QAT manifest is a later slice of the same packed stream, with `tokens_before` equal to the dense scored total. The trainer refuses to open it unless the resume checkpoint and `phase_start_tokens` both name that offset. The validation phase on the QAT manifest is the same eight-language residual. The QAT launch config itself is generated from the dense checkpoint, on the training host, because the resume contract includes the runtime identity.

## 10. What has to be true before the first update

The human launches the machine. The code does not. On that machine the order the launcher enforces is:

1. The manifest paths resolve to read-only files inside the data root. Staging hashes every payload, checks every descriptor, checks the tokenizer, and writes a receipt. Later startups compare inode, size, and timestamps to that receipt instead of hashing the terabytes again.
2. A native preflight receipt matches this model and this device. The AMD preflight covers attention, FP8, the packed loss, and the spectral Muon step on a reduced model. It does not train the 1.45B model.
3. A runtime report is captured after the ROCm imports. The resume contract hashes those native files.
4. Eight-rank collectives run before the first full update.
5. Unbounded training requires an operations file: the allocation record and a backup destination. The allocation record stores the real billing start, 5,760 GPU-hours, and 8 GPUs, and it cannot be recreated to move the deadline. The supervisor's stop and verify commands have to be the provider's commands. A failed verification means the machine may still be billable.

The first full updates are a memory test of microbatch 16. They are not a quality result. The 320 GPU-hour bring-up allowance in the config note is a planning reserve, not a measurement of how long bring-up will take.

## 11. What this run does not do

The English code cell is absent from the mixture. The final-role benchmarks and the tasks that require a written answer are not on the every-stop scorer. Overlap between the training shards and those benchmarks has not been audited. Licenses and a document-deduplication method are not fields the packed checker demands anymore. Their absence is not evidence about the corpus. The November 6 sealed pack was built with a different tokenizer and is not the validation set.

No line of this procedure has been timed on the eight MI325X GPUs. The 13 GB checkpoint size, the 76-minute checkpoint gap, and the tokens-per-second figures in the config note are illustrations from an assumed rate. The first saved checkpoint and the first measured update replace them.
