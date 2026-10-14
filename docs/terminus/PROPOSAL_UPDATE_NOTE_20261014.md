# Terminus proposal update: 1.5B dense-to-ternary model

**Document date:** 14 October 2026

**Updates:** *Terminus-2.6B: Native Ternary Small Language Model for Offline Indic AI*, dated 13 July 2026

**Purpose:** Replacement technical and program language for an external presentation and revision of the July proposal

## Executive update

Terminus is now specified as a **1,500,456,496-parameter decoder-only language model**. The selected model has 25 Transformer layers, hidden width 2,304, SwiGLU intermediate width 5,760, 18 query heads, six key/value heads, a tied 65,536-token embedding and output table, and a 4,096-token training context. Four local-attention layers with a 1,024-token window precede each full-context global-attention layer. Peri-LN and a learned cross-layer value residual are part of the selected architecture.

The training plan targets **one trillion valid training tokens across dense pretraining and ternary quantization-aware continuation**. Dense pretraining uses FP8 E4M3 linear computation with FP32 master weights, BF16 residuals, FlashAttention-3, and DeepGEMM. A later quantization-aware training (QAT) stage adapts eligible projection weights to scaled ternary values. The selected dense starting configuration and its optimizer are implemented. The trillion-token run, final QAT transfer, and deployment artifact are planned outcomes, not completed results.

The **seven initial Indic languages** are **Hindi, Bengali, Tamil, Telugu, Kannada, Marathi, and Gujarati**. English remains in the broader training and evaluation program. Language support will be reported from per-language evaluation of the trained release candidate; the language list states the intended support scope rather than a claim of already measured production quality.

This update replaces the July proposal's 2.6B size, ternary-from-initialization training method, 2.5T-token corpus budget, ten-language initial set, 8,192-token initial context, ZeRO-2 default, and integrated speculative drafter. Offline use remains the intended deployment direction, with runtime memory, latency, and quality to be established by exported-model and device measurements.

## 1. Controlled change record

| Subject | July 2026 proposal | Updated Terminus plan | Required change in the proposal |
| --- | --- | --- | --- |
| Model scale | Approximately 2.60B parameters | Exactly 1,500,456,496 trainable parameters in the selected model | Replace the model name and all 2.6B scale claims; use “Terminus 1.5B” as the proposal label. |
| Training sequence | Ternary weights from the start | Dense FP8 pretraining followed by ternary-weight QAT | Replace “native ternary training from initialization” throughout. |
| Training exposure | 2.5T pretraining tokens | One trillion valid training tokens across dense and QAT stages | Replace the 2.5T target, 1.0T Indic allocation, and derived source-count tables. |
| Indic coverage | Ten initial languages | Hindi, Bengali, Tamil, Telugu, Kannada, Marathi, Gujarati | Replace the ten-language list and its language-specific claims. |
| Depth and width | 35 layers; width 2,560 | 25 layers; width 2,304 | Revise architecture tables, parameter arithmetic, and training-memory discussion. |
| FFN | SwiGLU width 6,912 | SwiGLU width 5,760 | Revise architecture and compute tables. |
| Attention heads | 20 query heads; distinct local/global KV ratios | 18 query heads and six KV heads in all layers | Replace the asymmetric GQA description with a uniform 3:1 ratio. |
| Context | 8,192 tokens initially | 4,096 tokens in the selected starting model | Replace initial-context and KV-memory figures. Treat extension as a separately evaluated change. |
| Local/global attention | Four local for each global; local window 1,024 | Same cadence and window, yielding 20 local and five global layers | Retain the mechanism, update the layer count. |
| Normalization | Pre-RMSNorm; local Q/K norm; global QK-Clip | Pre-RMSNorm plus Peri-LN branch-output norms; learned Q/K RMSNorm in all layers | Replace the stability mechanism description. |
| Value path | Ordinary per-layer V projections | Learned mixture of each later layer's V and layer-zero V | Add the cross-layer value residual to architecture and cache descriptions. |
| Linear arithmetic | Ternary weights with full-precision activations in the stated regime | Dense-weight FP8 E4M3 operands and FP8 activations, with FP32 masters and BF16 residuals | Revise training precision and memory accounting. |
| Ternary grouping | One absmean scale per 128 weights in the proposal | Existing QAT implementation uses one scale per 128 × 128 tile | Describe the implemented quantizer accurately; do not present finer grouping as selected. |
| Optimizer | Muon for ternary matrices; AdamW for remaining groups | Spectral Muon on logical hidden matrices; separate AdamW groups for tied table and learned gains | Add the exact update rule and starting rates. |
| Data parallelism | ZeRO-2 default | Replicated 32-rank DDP is the starting strategy | Replace the default parallelism and its memory/communication rationale. |
| Speculative decoding | Jointly trained four-layer MTP drafter | No MTP drafter in the selected model | Remove it from the core architecture, parameter count, and expected latency claims. |
| Thinking mode | Optional named runtime mode | No selected thinking-mode specification | Remove it as an asserted feature of this model release. |
| Edge footprint | Sub-700 MB RAM target and Android kernels | Offline deployment remains a goal; packed file and runtime RAM are unmeasured | State a measurement program rather than a delivered footprint or kernel. |

The 25-layer/2,304-width model has about **1.10B fewer parameters** than the paper's approximate 2.6B target. Its depth is lower by ten layers, width by 256, FFN width by 1,152, and initial context by half. Its 128-dimensional attention heads, tied vocabulary table, SwiGLU, 4:1 local/global cadence, and 1,024-token local window remain central design elements.

## 2. Architecture specification for the revised proposal

### Model geometry and attention

The selected model has 25 decoder blocks. Each block contains grouped-query causal attention and a SwiGLU feed-forward network. Blocks 5, 10, 15, 20, and 25 use full causal attention over the 4,096-token context; the other 20 use a 1,024-token causal sliding window. Every layer has 18 query heads and six key/value heads, with head dimension 128. The query-to-KV ratio is therefore exactly 3:1 in both local and global layers.

The paper's “20 query heads with 8:1 global grouping” does not define an integer number of global KV heads. That combination should be removed from the revised tables rather than retained as an approximate implementation claim. The current 18/6 layout is explicit and consistent across layers.

The model uses a frozen **65,536-piece** tokenizer and ties the input embedding to the output classifier. The vocabulary size is a model invariant. The selected model contains **1,349,222,400 projection-weight parameters**, **150,994,944 tied embedding/readout parameters**, and **239,152 other trainable parameters**, summing to 1,500,456,496. The tied table stays outside the ternary projection set in the current design.

### Peri-LN and learned value residual

Each block retains learned RMSNorm before attention and before the FFN. It additionally normalizes the attention branch output and FFN branch output with separately learned RMSNorm gains immediately before each residual addition. This output-side normalization is the Peri-LN component. The selected model disables the older SubLN path. Query and key heads also receive learned-gain RMS normalization before rotary position embedding in every attention layer.

The value residual mixes the layer-zero value vector for a token with that token's value vector in each later layer. Two independently learned coefficients start at 0.5 and 0.5. The mixture occurs before the layer's attention mask and value aggregation. Cached inference must preserve the corresponding layer-zero values for newly decoded positions. This mechanism adds cross-layer reuse without changing the 4:1 local/global attention schedule.

The July proposal's local-only Q/K norm and post-optimizer global QK-Clip should not be described as the present selected stability stack. A global gradient clip of 1.0 in training is a separate optimizer safeguard; it is not QK-Clip.

### Precision and QAT interface

During the selected **dense stage**, the projection matrices are dense FP32 master parameters. Their GEMM operands are block-scaled FP8 E4M3 representations; activations used by the linear kernels are also FP8, while the residual stream is BF16. The selected path uses FlashAttention-3, DeepGEMM, fused QKV and MLP projections, fused pointwise operations, cached FP8 weight operands, and streaming cross-entropy. The tied embedding/readout and learned gains retain their separately specified parameter treatment.

During a later **ternary QAT stage**, eligible projection matrices retain trainable master weights while the forward path uses scaled ternary `{-1, 0, +1}` codes and FP8 arithmetic. The currently implemented scale is the mean absolute weight in each 128 × 128 weight tile, with a scale floor and a straight-through weight-gradient path. This is 16,384 weights per scale. The July paper's 128-weight grouping, clipped scale gradients, and post-update master-weight renormalization do not describe this implementation. QAT stage rates, duration, and acceptance criteria are established separately from the dense-stage configuration; the dense rates below are not a QAT prescription.

## 3. Optimization and training configuration

The selected dense optimizer applies **spectral Muon** to each logical hidden matrix. Q, K, V, gated FFN, FFN up, attention output, and FFN down projections retain their individual matrix geometry even when their physical projections are packed. For a matrix of shape `(d_out, d_in)`, the Newton–Schulz update direction is multiplied by `sqrt(d_out / d_in)`. The base learning rate is not multiplied by the older width-growing Muon factor. Tied embeddings, normalization gains, Q/K gains, and value-mixture coefficients use AdamW groups.

| Dense-stage setting | Selected starting value |
| --- | ---: |
| Spectral Muon base learning rate | 0.017 |
| Tied embedding/readout AdamW learning rate | 0.003 |
| Ordinary, Peri-LN, final-norm, and value-mixture AdamW learning rate | 0.0003 |
| Learned Q/K norm AdamW learning rate | 0.0001 |
| Hidden-matrix weight decay | 0.02 |
| Tied embedding/readout weight decay | 0 |
| Global gradient clip | 1.0 |
| Schedule family | Warmup–stable–decay, with cosine decay |
| Final learning-rate fraction | 0.1 of peak |

These are selected **starting settings** for the dense stage. The global batch, per-rank microbatch, accumulation, warmup tokens, decay tokens, and source manifest are bound in the production run configuration. The configuration builder checks them against the actual manifest and world size. Checkpoint and validation cadence are operational defaults subject to the run plan, rather than additional model-architecture claims.

The dense optimizer settings follow the completed 200M research campaign. A 700M-width bridge supported the tied-table learning rate of 0.003 against two higher rates on matched English cross-entropy at 500M tokens. That evidence supports a starting rate; it is not a measured one-trillion-token trajectory or a guarantee of width-transfer quality.

## 4. One-trillion-token data program

The revised program target is **1,000,000,000,000 valid training-token presentations** across the dense FP8 and later ternary-QAT stages. This number excludes padding and held-out evaluation tokens. Corpus uniqueness, repeat exposure, and per-source token counts are reported separately; a token presentation is not automatically a unique source token. This wording is the replacement for the paper's 2.5T-token pretraining claim and its derived 1.0T Indic/1.5T non-Indic allocation.

The initial Indic-language support set is:

| Language | ISO 639-1 code | Primary script |
| --- | --- | --- |
| Hindi | `hi` | Devanagari |
| Bengali | `bn` | Bengali |
| Tamil | `ta` | Tamil |
| Telugu | `te` | Telugu |
| Kannada | `kn` | Kannada |
| Marathi | `mr` | Devanagari |
| Gujarati | `gu` | Gujarati |

Training and evaluation also cover English and relevant code, mathematics, and mixed-language material. For the seven Indic languages, the release evaluation reports each language separately, including native-script text and relevant code-mixed usage. Neither the one-trillion-token target nor the support list implies equal token allocation, equal benchmark performance, or a completed language-quality result.

The July proposal's 40% Indic prior, ten-language coverage statement, and named source-token table are retired as fixed production allocations. The production data manifest is the source of truth for the actual mixture, tokenizer identity, source provenance, valid token counts, filtering, duplicate handling, and train/evaluation separation. Data governance, licensing review, contamination controls, and per-language quality checks remain release requirements.

## 5. Compute and production execution

The initial distributed strategy is **32-way replicated data parallelism**, with tensor, pipeline, and context parallel degrees of one. Each rank holds the entire selected model and optimizer state. The trainer accumulates local microsteps and reduces FP32 gradients through DDP before clipping and optimizer updates. ZeRO-2 is no longer the declared baseline; any sharded-state design requires a separate implementation and numerical comparison, especially because spectral Muon operates on complete logical matrices.

The existing production planning envelope is **6,000 aggregate H100 GPU-hours** across training phases. A one-trillion-token run within that envelope requires an average of **46,296 valid training tokens per second per GPU across all allocated GPU-hours**. At 32 GPUs, 6,000 aggregate GPU-hours correspond to 187.5 hours of allocation time. These are budget equations, not measured cluster throughput.

The optimized dense stack measured about **41,000 tokens/s on a single H100 NVL** in short complete-update receipts, with 55.5 GB peak allocated memory at the measured microbatch. Holding that rate for 6,000 GPU-hours with zero overhead would process about **886 billion dense tokens**. The comparison identifies a real capacity gap: the one-trillion target requires higher all-in throughput, more aggregate GPU-hours, or a revised allocation. Full 32-rank performance, the actual dense/QAT time split, communication, validation, checkpoints, and recovery have not yet been measured on the intended cluster. The proposal should therefore state one trillion tokens as the **training target**, not as an achieved throughput or completed dataset exposure.

The production acceptance record will measure full updates on the actual H100 NVL/PCIe topology, slowest-rank latency, memory, data loading, distributed communication, optimizer parity, checkpoint/restart cost, and separate dense and QAT rates. It will then reconcile actual valid tokens and GPU-hours against the one-trillion target.

## 6. Offline deployment, evaluation, and release language

Offline Indic assistance remains the intended use. The selected training model, however, does not include the July proposal's jointly trained four-layer MTP drafter or a defined “thinking mode.” Neither speculative-decoding speedup nor a special deliberation mode should appear as an implemented feature or as part of the model's parameter count.

The July proposal's **sub-700 MB RAM** statement must be treated as a device-level target until a packed artifact is exported and measured. With current parameter counts, two-bit storage of ternary-eligible projection weights plus a BF16 tied embedding and other BF16 parameters has an arithmetic lower bound of **639,773,792 bytes before scale metadata, file structure, alignment, activations, and KV cache**. On-disk model bytes and peak loaded RAM are different measurements. Android CPU kernels, device compatibility, latency, battery use, and sustained offline task quality require implementation and device tests before being reported as results.

Evaluation of a release candidate will separate dense and QAT checkpoints and report the change caused by the transition. It will include held-out next-token loss by language and domain, native-language and code-mixed tasks for all seven Indic languages, general English capability, factuality, safety, and the offline workflows named in the proposal. Benchmark versions, prompt formats, tokenizer identity, and contamination checks accompany the scores. Instruction tuning, preference alignment, safety work, and edge packaging remain subsequent program phases; they are not implied by the dense configuration.

## 7. Replacement copy for the July proposal

The following passages are written for direct insertion after updating section numbering and surrounding references. They deliberately distinguish the selected design from milestones that require training or device evidence.

### Replacement title and short description

> **Terminus 1.5B: A Dense-to-Ternary Small Language Model for Offline Indic AI**
>
> Terminus is a 1.5B-parameter text model designed for staged dense and ternary training, with initial Indic-language support focused on Hindi, Bengali, Tamil, Telugu, Kannada, Marathi, and Gujarati. The program targets one trillion valid training tokens and evaluates the resulting model for useful offline deployment.

### Replacement executive summary

> Terminus is a 1,500,456,496-parameter decoder-only language model for an offline Indic AI program. It uses 25 Transformer layers, a 65,536-token tied vocabulary, 4,096-token context, grouped-query attention, and a four-local-to-one-global attention cadence. Peri-LN and a learned cross-layer value residual are part of the selected architecture. The training plan targets one trillion valid tokens across dense FP8 pretraining and a later ternary-weight quantization-aware continuation. The seven initial Indic languages are Hindi, Bengali, Tamil, Telugu, Kannada, Marathi, and Gujarati. The model's multilingual quality, final ternary quality, and mobile memory profile will be reported from the trained and exported release candidate.

### Replacement architecture and training paragraph

> The selected model has 25 decoder layers at hidden width 2,304 and SwiGLU width 5,760. It uses 18 query heads and six key/value heads of dimension 128 throughout the network. Twenty layers have a 1,024-token local causal attention window and five layers have full causal attention over the 4,096-token starting context. Dense pretraining retains FP32 master projection weights while FP8 E4M3 operands execute the linear computation; the residual stream uses BF16. FlashAttention-3 and DeepGEMM provide the selected GPU path. A separate ternary QAT stage adapts the trained dense model. The tied embedding/readout remains outside the projection-weight ternary set. Hidden matrices use spectral Muon, with AdamW for the tied table and learned gains.

### Replacement data and language paragraph

> The production training plan targets one trillion valid token presentations across dense pretraining and ternary QAT. The final data manifest records the allocation by source and language, the frozen tokenizer, filtering and licensing status, duplicate handling, and held-out evaluation boundaries. The initial Indic-language support and reporting set is Hindi, Bengali, Tamil, Telugu, Kannada, Marathi, and Gujarati. Evaluation reports these languages individually and includes relevant code-mixed use, alongside English and domain-specific assessments.

### Replacement infrastructure and deployment paragraph

> The initial scaling plan uses 32 replicated H100 data-parallel ranks. Full-cluster throughput and the dense/QAT cost split are measured before a one-trillion-token schedule is declared feasible within the aggregate GPU-hour envelope. Offline deployment is evaluated from an exported ternary artifact on target devices. The release report distinguishes packed model-file size from peak runtime RAM and gives measured latency, memory, task quality, and device compatibility. The selected model does not include the speculative drafter or thinking-mode feature described in the July 2026 proposal.

## 8. Source and claim basis

- **July baseline:** *Terminus-2.6B: Native Ternary Small Language Model for Offline Indic AI*, 13 July 2026, especially Sections 1, 3, 4, and 11. The PDF is the source for the superseded 2.6B architecture, native-ternary method, 2.5T allocation, ten-language set, ZeRO-2 plan, MTP drafter, and sub-700 MB target.
- **Selected model and dense settings:** [production/recipe.py](../../production/recipe.py) and [production/README.md](../../production/README.md). These are the authoritative selected baseline in this repository.
- **Current distributed starting plan and cluster acceptance:** [production/PARALLELISM.md](../../production/PARALLELISM.md).
- **Spectral Muon rule:** [docs/MUON_SPECTRAL_IMPLEMENTATION.md](../MUON_SPECTRAL_IMPLEMENTATION.md).
- **Implemented quantization behavior:** [terminus/quantization.py](../../terminus/quantization.py) and [terminus/model.py](../../terminus/model.py).
- **Single-GPU speed and limits of transfer:** [docs/speed/TRANSFER_32GPU.md](../speed/TRANSFER_32GPU.md).
- **Projection and tied-table storage arithmetic:** [manuscript/terminus_draft.md](../../manuscript/terminus_draft.md), Section 7.
- **One-trillion-token target and seven-language support set:** revised program direction supplied for this update on 14 October 2026. These replace the older token and language targets in the July proposal. They are program targets, not claims of a completed production run.
