# Transition plan: from-scratch Terminus pretraining to continued pretraining on Spark

Date: 2026-11-12. Status: for review. No existing tree was modified to produce this plan.

This plan is the safe path if from-scratch pretraining of the selected Terminus 1.45B model becomes unaffordable. It assumes the stated fallback: about 100B tokens of continued pretraining on `XHToken/Spark-X2.6-1.7B` (Apache 2.0), with a 50:50 split of Nemotron-pretraining-Indic-v1 and replay data, then SFT/DPO, then a ternary / 1.58-bit model. Only the continued-pretraining stage is planned here. Vocabulary surgery and ternary conversion are constraints on the design. They are not tasks in this folder.

The from-scratch track stays in place. `production/`, `terminus/`, the AMD readiness docs, and `Alternate direction/` stay read-only. A later CPT trainer, if this plan is accepted, is new code under this folder and reads external data. It does not become the production launcher.

## 1. Decision this plan is asking for

Accept or edit these choices before any implementation:

1. CPT parent is **`XHToken/Spark-X2.6-1.7B-Base`**, pinned by commit, loaded with the vendor `modeling_spark.py`. The instruct repository is a post-training sibling with the same config and tokenizer and much worse raw-text perplexity. It is the wrong parent for a pretraining objective.
2. The pilot and any following 100B run that claims to continue this pilot use the **unmodified** Spark tokenizer and the **unmodified** embedding matrix.
3. Vocabulary surgery is a **gate in front of the 100B run**, not a step inside it. The 8B pilot may run on the stock tokenizer either way. Its weights are disposable if surgery is later approved, because token ids would change.
4. The 50:50 mix is counted in **scored Spark tokens** (labels that enter the loss). Gemma-4 token counts published with the Indic corpus are inventory numbers, not this budget.
5. The pilot Indic half is a frozen native-script sample of **Nemotron-Pretraining-Indic-CC-v1** in the ten languages that already have a task harness. Romanized rows, the ten lower-resource languages, and Specialized-v1 wait. The umbrella name “Nemotron-pretraining-Indic-v1” is two datasets, not one repository.
6. Replay is independent English, code, and math text. It is not the `source_text` of the Indic rows that are being trained, and it contains no Chinese slice in the pilot.
7. Training sequences are **one document, or one contiguous window of a long document**, at 4,096 tokens if a memory smoke test allows it. Cross-document packing from the Terminus packer is not used.
8. Optimizer is **AdamW**. Spectral Muon, FP8, and the Terminus dense-to-ternary handoff stay on the Terminus track.
9. A single training-class GPU runs the pilot. Multi-GPU is a later gate. The 24 GB A10G boxes used for the Spark probes are inference machines.
10. No GPU launch is implied by accepting this document.

## 2. What the two systems actually are

### Terminus production track

The selected model is 24 layers, width 2,304, FFN 5,760, 18 query / 6 KV heads, tied vocabulary 65,536, context 4,096, local window 512 on 20 layers, global attention at layers 6, 12, 18 and 24, Peri-LN, learned value residual, 1,446,478,126 trainable parameters. Training is spectral Muon plus a small AdamW set, on an FP8/DeepGEMM stack. `production/train.sh` rebuilds `BitNetLlamaConfig` from `production/recipe.py` and refuses any other geometry.

Data is an external manifest of `uint16` shards. `production/document_packing.py` rejects any id at or above 65,536, uses end-of-document id 4, and rejects ids 3 through 31. Segment masks, reset RoPE, and a valid-target denominator are part of that contract. Checkpoints are contract-hashed for that model, optimizer, manifest, and runtime.

SFT and DPO are named in the program-budget documents and are not implemented. Ternary training is an in-graph straight-through conversion inside this same architecture (`terminus/transition.py`). It is not a converter for an arbitrary Hugging Face checkpoint.

### Spark-X2.6-1.7B

Public config for the Base repository, `transformers_version` 4.57.1:

| Field | Value |
| --- | --- |
| Class | `Spark2_5ForCausalLM` (`model_type` `spark2_5`), remote code |
| Layers | 28: three sliding-window layers, then one full layer, repeated |
| Sliding window | 512 |
| Hidden / FFN | 2,048 / 6,656 |
| Attention | 8 query heads, 2 KV heads, head dim 256, fused `q_k_v_proj`, headwise sigmoid `g_proj` |
| MLP | GeLU-gated (`gate_proj`, `up_proj`, `down_proj`), not SwiGLU |
| RoPE | Sliding layers: theta 10,000, full rotary. Full layers: theta 5,000,000, rotary on 25% of the head dim |
| Context in config | 1,048,576 |
| Vocabulary | 131,072, tied embeddings |
| Special ids | bos 0, eos 1, pad 2, then think / chat / FIM tokens through id 13 |
| License tag | Apache 2.0 |
| Weights on disk | two safetensors shards, 1,997,108,320 + 1,418,232,176 bytes (`Alternate direction/results/probe1_repo_facts.json`) |

A published parameter total of 1,707,657,216 matches that shard size up to safetensors headers. Recompute it from `model.safetensors.index.json` on the pinned revision before any run. This plan does not treat that total as a local measurement.

The Base and instruct repositories have identical `config.json`, `tokenizer.json`, and `generation_config.json` (probe 1). They do not have identical behavior. On the Wikipedia probe, Base English perplexity is 9.6 and Hindi is 9.3; the instruct checkpoint is 49 and 160 on the same text (`probe5_perplexity.json`). The Base model card still says the repository “contains model weights for the post-trained model” and lists only English and Chinese. That sentence disagrees with the perplexity gap. Treat it as a card defect until the two weight trees are hashed. CPT uses Base.

Vendor fine-tuning advice points at `XHToken/LlamaFactory`. That fork is unreviewed. CPT is a causal-language-model continuation, not chat SFT. The pilot trainer, when it is written, should be a small Hugging Face loop pinned to transformers 4.57.1. Adopting the fork is a separate review.

### Why the stacks do not meet in the middle

Spark ids do not fit in `uint16`. Spark’s attention is eager, fused, gated, and hybrid, with two RoPE regimes. The vendor forward builds RoPE from `cache_position`, which is a monotonic range, and does not read `position_ids` for the rotation. A Terminus segment mask and a per-document position reset would not do what they do on Terminus even if the tensors were smuggled in. Full-attention logits are materialized over 131,072 rows. The production runtime image (PyTorch 2.8, Triton, FA3, DeepGEMM, spectral Muon) is the Terminus image. Spark needs a separate, smaller environment.

So “reuse” below means reuse of contracts, checks, and harnesses. It does not mean pointing `production/train.sh` at a Spark config.

## 3. Reusable components

| Component | What carries over | How it is used |
| --- | --- | --- |
| Language-quota discipline | `production/packed_mixture.py` interleaves languages and dataset children by exact finite counts | Copy the rule: deterministic 50:50 by scored tokens, equal Indic languages, no physical file order. The reader itself expects Terminus descriptors. |
| Document window rule | `production/document_packing.py` splits long documents into contiguous windows and emits end-of-document only on a real ending | Same window rule, Spark bos/eos/pad, int32 ids, one window per sequence. The `uint16` writer and id-4 end marker stay unused. |
| Valid-target accounting | Terminus refuses a step whose loss denominator disagrees with the real target count | Single-GPU pilot uses mean loss over non-ignored labels. Before any multi-GPU run, re-implement that global check. Different ranks will have different pad fractions. |
| Checkpoint contract | `terminus/checkpoint.py`: write, fsync, hash, monotonic `latest`, refuse a changed contract | Same operational standard, new schema: model revision, tokenizer sha256, data cursor (`source_uuid`, row), tokens seen, seed, LR schedule. Terminus payload fields are not the schema. |
| Run health | `production/status` and `production/checkpoints` watch disk, stale logs, and receipt hashes | Same operator habits. Those commands understand Terminus run directories, so they are not pointed at a Spark output dir. |
| Held-out exclusion | Packing docs require held-out families to be absent from training inputs, not merely loss-masked | Same rule for a withheld Indic slice, a replay slice, and the task sets. |
| Indic task harness | `Alternate direction/scripts/eval_indic_mcq.py` and `eval_indic_gsm8k.py` | Protocol is reusable: letter log-probabilities for multiple choice, completion prompting for GSM8K. The scripts hardcode Hugging Face repo ids and a batch of 32. A future copy in this folder should take a local checkpoint path and a smaller batch. Do not edit the originals. |
| Step-0 measurements already on disk | Probe 2 fertility, probe 4 embedding categories, probe 5 perplexity, probe 1 identity, MILU / Belebele / BhashaBench Base JSON | These are the starting picture. Several other JSON files are incomplete (section 6). |
| Quality-gate checklist | `production/amd/pretraining_readiness/QUALITY_GATES.md` | Tokenizer identity, decontamination, finite grads, resume, and “do not launch from a document” all still apply. The numeric gates (FP8, Muon, 65,536-id manifest) do not. |
| Program-budget habit | Reserve post-training and a final conversion before spending the long run | Hours and token rates from the MI325X / H100 Terminus plans do not transfer. |

Nothing in the kernel, optimizer-group, QAT, AMD profile, or `uint16` manifest path is a Spark component. `production/build_runtime.sh` should not be the CPT image.

## 4. What breaks if the current infrastructure is reused as-is

These are the breaks that would silently waste a run or refuse it at startup.

1. **`production/train.sh` identity check.** The wrapper sets `weight_mode` on `SELECTED_MODEL` and raises if the loaded config differs. Spark never passes.
2. **`uint16` and the 65,536 ceiling.** Document packing, the AMD profiles, the tokenizer gate, translation-pair checks, and manifest building all assume ids below 65,536. Spark’s vocabulary is 131,072. Existing shards are the Terminus tokenizer’s ids even when the original text was Indic.
3. **End-of-document id and the reserved band.** The packer writes id 4 and rejects 3–31. On Spark, id 4 is `</think>`, ids 0–13 are specials, and eos is 1. Reusing the packer would either throw or teach the think marker as a document boundary.
4. **RoPE.** Vendor `Spark2_5Model.forward` computes cos/sin from `cache_position`. Passing Terminus per-document `position_ids` does not reset positions. Packed sequences would train full-attention layers on the wrong positions.
5. **Attention mask type.** A normal 2D padding mask is turned into causal and sliding masks inside the model. A Terminus segment tensor is a different object. Until a padded-versus-unpadded loss check passes, even ordinary padding is unverified.
6. **Eager attention and the 131k logit row.** There is no FlashAttention path in `modeling_spark.py`. Attention weights are materialized, and `logits_to_keep=0` builds the full `[batch, sequence, 131072]` logit tensor. A 4,096-token step that is routine for Terminus can OOM here. The 22 GB probe GPU already OOM’d during evaluation.
7. **Gradient checkpointing signature.** Checkpointing is declared, and the recompute call passes the mask, but it does not pass cache arguments. That is probably what training wants (`use_cache` must be off). It is an assumption until one backward pass is compared with checkpointing on and off.
8. **Tied embedding.** Forward uses `model.embedding.weight` when `tie_word_embeddings` is true, and also constructs `lm_head`. The pilot must assert the two tensors share storage. A failed tie doubles the embedding update and breaks a later export.
9. **Optimizer and schedule.** Muon 0.017 and embedding AdamW 0.001 are Terminus numbers. Spark’s original pretraining LR is not on the model card. Importing either number is an unreviewed recipe change.
10. **Checkpoint resume.** `terminus.checkpoint.load_checkpoint` checks a Terminus contract hash. A Spark state dict will not satisfy it. Resume has to be implemented and then kill-tested.
11. **Evaluation binding.** `terminus/task_eval.py` checks a Terminus tokenizer sha256. Indic numbers live in the Alternate-direction scripts, and several of those result files contain CUDA OOM errors rather than accuracies.
12. **Ternary handoff.** `terminus/transition.py` accepts a dense Terminus checkpoint and changes `weight_mode`. Spark tensor names are `q_k_v_proj`, `g_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`, plus RMSNorms and a tied embedding. A Llama-shaped converter will not find `q_proj`. Partial RoPE and the hybrid mask live in code, not in the weight file.

## 5. Step-by-step transition

Each step stops if its check fails. Later steps are not started to “save the queue.”

### Step A. Freeze the parent, still read-only

Download `XHToken/Spark-X2.6-1.7B-Base` to an external directory. Record the Hugging Face commit, the sha256 of `config.json`, `modeling_spark.py`, `configuration_spark.py`, `tokenizer.json`, `tokenizer_config.json`, and both weight shards. Hash the instruct weight shards far enough to prove they differ. Confirm Apache-2.0 text in `LICENSE`.

Load with `trust_remote_code` and transformers **4.57.1**. Assert:

- `config.vocab_size == 131072` and `tie_word_embeddings is True`
- `lm_head.weight` and `model.embedding.weight` share storage
- special strings still map to ids 0–13 as in the released vocab
- one forward on a short English string and a short Hindi string matches a second process given the same ids

Do not convert the model to Llama, do not call `resize_token_embeddings`, and do not save a “cleaned” tokenizer.

### Step B. Close the data contract on paper, then on a sample

The Indic corpus the user named is the NVIDIA umbrella of about 2.8T Gemma-4 tokens. The public pieces are:

- [`nvidia/Nemotron-Pretraining-Indic-CC-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Indic-CC-v1): 1,933,876,906 rows, about 2.16T Gemma-4 output tokens, about 8.65 TiB parquet, 34 language-script configs. Native-script priority languages are the bulk (1,732.7B of those tokens).
- [`nvidia/Nemotron-Pretraining-Indic-Specialized-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Indic-Specialized-v1): the non-overlapping companion, about 295.8B Gemma-4 output tokens.

CC-v1 rows are English web and PDF text translated into Indic, with `text`, `source_text`, `source_uuid`, `language_code`, and a carried `license`. Output token counts used Gemma-4, not Spark. Access is a gated NVIDIA Data Agreement for Model Training, plus ODC-By for the FinePDFs-derived rows. A person accepts that agreement before a download. This plan does not.

Pilot Indic sample, after acceptance:

- Configs only: `hi-Deva-IN`, `bn-Beng-IN`, `ta-Taml-IN`, `te-Telu-IN`, `mr-Deva-IN`, `gu-Gujr-IN`, `kn-Knda-IN`, `ml-Mlym-IN`, `pa-Guru-IN`, `or-Orya-IN`.
- Equal scored Spark tokens across those ten.
- Each `source_uuid` appears in at most one language, so the small budget is not ten translations of one English article.
- Drop romanized (`*-Latn-IN`) rows.
- Drop the ten lower-resource configs until a fertility check exists for Ol Chiki, Meitei Mayek, and the languages with no task harness.
- Stream a sample. Do not mirror 8.65 TiB for an 8B-token pilot.

Replay, still unnamed on disk:

- Inside the 50% replay budget: 60% English prose, 25% code, 15% math, unless the available licensed corpora force a different internal split. Record the actual split.
- Exclude `source_text` of every selected Indic row, so the mix is not “article plus its translation.”
- Exclude Chinese from the training mix. Keep a Chinese holdout only as a drift probe.
- Raw text, then the pinned Spark tokenizer. Terminus `uint16` shards are ineligible.

Hold out, before shuffling:

- Per Indic language, a fixed id list that never enters training.
- An English, code, and math holdout from the replay sources.
- Every task item used in section 6, by normalized substring or 13-gram overlap against `text`. Over-dropping is acceptable at this budget.

Filter, without editing the tokenizer:

- NFC the text.
- Reject a document whose Spark ids intersect `{0,1,2,3,4,5,6,7,8,9,10,11,12,13}` after tokenization. Those are bos, eos, pad, think, unk, and chat/FIM markers. The trainer will add bos and eos itself.
- Reject empty documents and documents that decode back to a different string.

Token budget for the pilot pool: 8B scored training tokens (4B Indic, 4B replay) plus holdouts. Store them outside Git as int32, with a manifest of shard hashes, tokenizer sha256, and the `source_uuid` list. Preparing the 100B pool waits until the pilot gate.

### Step C. CPU checks on the sample

- Fertility on the actual sample, per language, versus probe 2’s Wikipedia rates. A large disagreement means the sample is not the language we think it is.
- Id histogram: fraction of bytes that fell back to byte pieces, fraction of specials (must be zero after the filter), max id `< 131072`.
- Pad fraction under the one-document 4,096 rule. This number prices the run. It is allowed to be ugly. It is not allowed to be hidden.
- One unpadded sequence and the same sequence padded to 4,096 produce the same mean loss on the real labels, with the model in eval mode, on CPU or on one GPU. If they differ, padding is wrong and training stops.
- Deterministic sampler: a second process rebuilds the same first 1,000 sequences from the manifest.

### Step D. One-GPU numerical smoke, human-launched

On a GPU with room for AdamW master weights (fp32 parameters, gradients, and two moments are on the order of 16 bytes per parameter, about 27 GB before activations) plus eager attention:

- Microbatch 1, sequence lengths 512, 2,048, and 4,096, gradient checkpointing on, two optimizer steps each.
- Record peak memory, tokens/s, and whether the loss is finite.
- Lock the longest length that finishes with a margin. The pilot target is 4,096. If only 2,048 fits, the token budget stays in scored tokens and the length change is written down.
- Assert unused special rows and a sample of CJK rows are bitwise unchanged after one step when embedding weight decay is 0 and those ids are absent from the batch.
- Assert the pad row’s gradient is zero.
- Save a checkpoint, kill the process, resume, and match the next step’s loss to a process that was not killed.

Environment: a new virtualenv, transformers 4.57.1, a torch build that can run this GPU. Not `production/build_runtime.sh`.

### Step E. Learning-rate screen, then the 8B pilot

Details are in section 7. Three AdamW peaks share one data prefix and one schedule defined on an 8B horizon. Only the winner continues. The other two checkpoints are archived and not merged.

### Step F. Gate before any 100B discussion

A person compares the pilot report with the pre-registered margins in section 7. Inconclusive movement means the pilot is extended or stopped. It does not mean the 100B job starts.

Before a 100B job is even scheduled, write down one of these, explicitly:

- **Surgery deferred.** The 100B run continues the stock tokenizer. The pilot checkpoint may be the parent only if length, mix, optimizer, and tokenizer hash match.
- **Surgery required first.** The pilot checkpoint is not the parent. Surgery is a separate reviewed project (new ids, new embeddings, new pack, a new short CPT). This plan does not design that project.
- **CPT stopped.** The from-scratch track remains the only training plan.

Also before 100B: measured tokens/s from step D or E, on the GPU family that would actually run it. GPU-hours for 100B scored tokens are `1e11 / (tokens_per_second) / 3600` per GPU, divided by the number of GPUs only after a multi-GPU smoke with the valid-target check. The historical Terminus figure of about 41,000 tokens/s is a different stack and is not an input.

### Step G. Later stages, not designed here

Post-training starts from a frozen Hugging Face export of the accepted CPT base: config, modeling files, stock tokenizer, bf16 weights. SFT/DPO data, chat-template training, and preference pairs are a new project. The pilot’s only duty to that project is to leave ids 0–13 and the chat template file unchanged, and to show that unused special rows did not drift.

Ternary / 1.58-bit conversion starts from that same dense export, or from the post-trained export if post-training happens first. It uses a Spark-aware converter. It does not call `terminus.transition`. Tensors that must stay high precision until a converter is reviewed: tied embedding, all `Spark2_5RMSNorm` weights, and `g_proj` (a `Linear(2048, 8)` per layer in the current modeling file). Candidate low-bit matrices are `q_k_v_proj`, `out_proj`, `gate_proj`, `up_proj`, and `down_proj`. The hybrid mask and partial RoPE have to ship with the code, because they are not in the checkpoint. No in-training ternary is used during CPT, so the dense weights stay a valid parent.

## 6. Risks and mitigations

| ID | Risk | Why it is real here | Mitigation |
| --- | --- | --- | --- |
| R1 | Catastrophic forgetting of English, code, and math | 1–2B models forget quickly. Spark’s card lists English and Chinese, and Indic script tokens are a few thousand rows. A 50% Indic shift is large. | Keep the user’s 50% replay. Put code and math inside replay. Screen LR. Stop if the English holdout or MILU English moves past the margin in section 7. An earlier research note in `Alternate direction/CPT_BENCHMARKS_DEEP_RESEARCH.md` discussed 75% Indic; this plan stays at 50:50 because that is the more stable Indic precedent and it is the mix that was requested. |
| R2 | Learning-rate overshoot or a dead run | Original Spark LR is unpublished. Terminus Muon settings are for another optimizer. | Three peaks only: 1e-5, 3e-5, 1e-4. Same data, batch, and 8B horizon. Continue one arm. Refuse a second grid until this one is reported. |
| R3 | Embedding drift on Chinese and specials | About 33,955 CJK tokens and the chat/think ids will be rare or absent. Weight decay would pull them toward zero and damage a later SFT template and a later CJK prune. | Weight decay 0 on embeddings and RMSNorms. Weight decay 0.1 on matrices. Step-1 bitwise check. Chinese holdout perplexity is a monitor, not a training target. |
| R4 | Packing leaks context or breaks RoPE | Vendor RoPE ignores `position_ids`. Full layers attend the whole 4,096. `uint16` packing cannot store the ids. | One document or one contiguous window per sequence. Bos at the start, eos only on a true end, pad id 2, label −100 on pad. No Terminus segment file. |
| R5 | Padding dominates cost and silently enters the loss | Indic fertility is high (probe 2: English 1.42 tokens/word, Hindi 2.16, Tamil 4.01, Telugu 4.72, Malayalam 5.25, Odia 6.64 on the Wikipedia sample). Short documents leave most of a 4,096 window empty. | Count only scored tokens toward 8B and 100B. Log pad fraction. The padded-versus-unpadded loss check is a launch blocker. |
| R6 | Cross-document attention looks like a free speedup | The Terminus packer is the obvious tool to copy, and it would raise throughput. | Leave it unused until a reviewed model change honors per-document positions and a block mask. Throughput is not a reason to change attention during the pilot. |
| R7 | Synthetic translationese | CC-v1 is machine-translated English web and PDF, judged by another LLM. Odia’s Gemma token count is far higher than Hindi’s on a similar input-token budget, which is consistent with fertility, and also with uneven generation. | Pilot on ten eval languages, native script, one `source_uuid` per language, quality field recorded. Judge success with task metrics and a human spot-read, not with training loss alone. |
| R8 | Double-counting English | Training on `text` and on that row’s `source_text` is a parallel corpus, not replay of an independent distribution. | Replay corpora are separate. Selected `source_uuid`s are banned from replay. |
| R9 | Tokenizer surgery invalidates a long CPT | Adding Indic pieces and dropping CJK renumbers rows. A 100B run on today’s ids does not survive a resize, and published experience is that a cold swap needs a large further token budget to recover. | Surgery decision is a written gate before 100B. Pilot weights are allowed to be thrown away. This folder does not prune or add tokens. Probe 3’s dry run is evidence only. |
| R10 | Ternary conversion assumes Llama names and SwiGLU | Spark is fused QKV, a head gate, GeLU, partial RoPE, and a sliding pattern. | CPT saves bf16 Hugging Face weights plus the exact modeling files. No `weight_mode`, no FP8, no Terminus QAT state. |
| R11 | Base card is misread as “already instruct” | The Base README text says post-trained. | Parent choice follows the perplexity gap and a weight hash, not that sentence. |
| R12 | Instruct eval protocol is used on a base CPT model | GSM8K and the shootout scripts use the chat template and thinking budgets for the instruct model. | During CPT, multiple choice is letter log-probability and GSM8K is completion prompting, matching the base path already sketched in `eval_indic_gsm8k.py`. Instruct leaderboard numbers are not the baseline. |
| R13 | Incomplete baselines get averaged | On the 22 GB probe GPU, `mmlu_indic__spark_base.json` has a score for English only. ARC Base is missing Malayalam, Odia, and Telugu. MMLU-Pro Base is mostly OOM errors. | A missing cell is missing. MILU Base, Belebele Base, and BhashaBench Base have no error cells and are the current task picture. Re-run the harness on the training-class GPU before calling anything a delta. |
| R14 | Wikipedia perplexity looks like success | Probe 5 used Wikipedia openings and the write-up already says that text is likely in pretraining. | Acceptance uses withheld Indic and replay text, plus tasks. Wikipedia perplexity is a drift side-check. |
| R15 | Eager attention makes the cost model wrong | No measured Spark training tokens/s exist in this repo. | Smoke test records tokens/s and memory. The 100B price uses that measurement. |
| R16 | Multi-GPU loss scale | Pad fraction will differ by rank. Averaging mean losses without a global target count changes the objective when the world size changes. | Pilot is one GPU. Multi-GPU waits on an explicit valid-token reduction. |
| R17 | Remote code and transformers drift | `modeling_spark.py` calls `create_causal_mask`, `create_sliding_window_causal_mask`, and `PreTrainedModel.loss_function` as of the 4.57.1 line. | Pin 4.57.1 and the modeling-file hash. A newer transformers is a new smoke test, not a silent upgrade. |
| R18 | License | CC-v1 is gated. FinePDFs-derived rows carry ODC-By. Replay corpora are not chosen yet. | No download and no training until a person accepts the terms and names the replay sources. |
| R19 | Data contamination of MILU, ARC, GSM8K, HumanEval, Belebele, BhashaBench | Synthetic web translations can contain exam text. | Decontaminate the sample against those sets before the shuffle. Record the drop count. |
| R20 | From-scratch track is damaged by “just a small hook” | Shared launchers and manifests are how a fallback becomes an accidental production change. | CPT code, if approved later, lives only under `cpt_spark_transition/`. Production files stay as they are. |
| R21 | Low-resource scripts look trainable because the corpus is large | Santali (Ol Chiki) and Manipuri (Meitei, via a Bengali mapping) are outside probe 2. Byte fallback would spend the budget on fragments. | Out of the pilot. A fertility spot-check is a next action, using the stock tokenizer as a reader. |
| R22 | Initial loss dip is mistaken for failure | CPT literature expects an early dip on the new mixture. | Judge the LR screen on holdout perplexity and gradient health at 250M, not on the first few thousand steps. Task gates start at 2B. |

## 7. Recommended pilot

Purpose: learn whether this base, this tokenizer, this 50:50 mix, and a conservative AdamW schedule move Indic holdout loss and MILU without giving back English, and what a scored token costs on the real GPU. The pilot is not a quality claim and not a 100B authorization.

### Configuration

| Knob | Value |
| --- | --- |
| Parent | Pinned `XHToken/Spark-X2.6-1.7B-Base` |
| Tokenizer | Stock, sha256 pinned in the manifest |
| Loss | Next-token cross-entropy, mean over labels that are not −100 |
| Context | 4,096 if step D passes, otherwise 2,048 |
| Example | `[bos] + window + [eos if document ends]`, pad to context with id 2 |
| Global batch | 262,144 scored-token slots (64 sequences at 4,096), accumulated from microbatch 1. If memory forces a smaller batch, lock it before the LR screen and do not retune mid-screen. |
| Indic | 50% of scored tokens, CC-v1 native script, ten languages, equal shares, `source_uuid` unique |
| Replay | 50% of scored tokens; proposed internal mix 60/25/15 English/code/math |
| Seed | 20261112 |
| Optimizer | AdamW, betas 0.9 and 0.95, grad clip 1.0, matrix weight decay 0.1, embedding and norm weight decay 0 |
| Peaks | 1e-5, 3e-5, 1e-4 |
| Horizon | Schedule is written as 8B scored tokens. Warmup 80M. Constant until 6.4B. Cosine decay over the last 1.6B down to 0.1× peak. |
| Screen | Each peak runs 250M scored tokens (still in the constant-LR region) and stops. |
| Continuation | One winner runs on to 8B from its 250M checkpoint. Same schedule, no rewarm. |
| Precision | bf16 compute, fp32 Adam state, gradient checkpointing on, `torch.compile` off, FP8 off |
| Devices | One GPU |
| Checkpoints | Every 250M scored tokens, plus the resume drill. Weights-only Hugging Face export beside the full optimizer state. External disk. |
| Environment | transformers==4.57.1 and the hashed modeling file |

Unique training text required is 8B scored tokens, because the two losing arms repeat the winner’s first 250M. Holdouts are extra.

### What to measure

At step 0, on the pilot GPU, paired with every later eval:

- Holdout perplexity per Indic language, English, code, math, and the Chinese probe.
- MILU, the ten Indic languages plus English, full protocol (`n=1000`, 5-shot letter logits, seed 42). Today’s Base file is complete: English 53.7, and the ten-language mean is 35.08 (per-language values in `Alternate direction/results/evals/milu__spark_base.json`). Re-measure anyway so the delta shares a device and a code revision.
- Belebele and BhashaBench Base files are complete enough to re-use as a second view. ARC, MMLU-Indic, and MMLU-Pro Base files are not complete and are re-run rather than patched.
- A small GSM8K completion set (English plus the Indic split), exact match of the final number, base prompting.
- Code holdout perplexity. A short HumanEval-style pass rate only at 8B, completion prompting, not the instruct thinking template.
- Pad fraction, tokens/s, peak memory, fraction of steps whose grad norm was clipped, nonfinite count.

At 250M, for all three arms: holdout perplexity, grad-norm health, special-row drift, Chinese probe. Tasks only if perplexity already separates the arms; otherwise save the GPU for the continuation.

At 2B, 4B, and 8B on the winner: the full list above.

### How the winner is chosen

At 250M, discard an arm whose replay or English holdout perplexity is more than 5% worse than step 0, or whose loss or grad norm was nonfinite, or whose clip fraction stayed above half the steps over the last 20 updates. Among the survivors, pick the lowest mean Indic holdout perplexity. If none survive, stop. If the three Indic perplexities differ by less than the run-to-run noise (a second 250M is not automatically launched), prefer the middle peak, 3e-5, and write that the screen did not separate.

### Pre-registered reading of the 8B result

These margins are for a human gate, not an automatic launcher.

Proceed to a **proposal** for 100B only if all of the following are true:

- Mean Indic MILU is at least 3 points above the paired step-0 mean, and at least 8 of 10 Indic languages moved up.
- English MILU is at most 2 points below the paired step-0 score.
- English, code, and math holdout perplexities are at most 10% worse than step 0, relatively.
- Indic holdout perplexity improved in at least 8 of 10 languages.
- Resume drill passed, special-id mapping is unchanged, and the modeling-file hash is unchanged.
- Tokens/s and pad fraction are recorded.

A smaller MILU move is an inconclusive pilot. Options are then: extend that same arm, or stop. Starting 100B on an inconclusive pilot is outside this plan.

Abort earlier if, at 2B or 4B, English MILU is already more than 5 points down and Indic MILU is not up, or if holdout perplexity diverges.

### What the pilot deliberately does not answer

Whether 100B is enough for production Indic quality. Whether a 75% Indic mix would score higher. Whether vocabulary surgery would have been cheaper. Whether the model can be served at 1.58 bit. Whether 1,048,576-token training is useful. Long context stays a property of the released architecture. CPT does not try to occupy that context.

## 8. Immediate next actions

All of these are read-only or are notes. None of them edit a tokenizer, pack a training shard, start a VM, or add a trainer.

1. Review this file and mark section 1 accepted, edited, or rejected.
2. Have a person accept or refuse the NVIDIA data agreement and the FinePDFs ODC-By carry-through, and name the replay corpora with their licenses. Code and math sources are still missing pieces.
3. Pin the Base commit and the sha256 list in an external note. Confirm instruct weights differ.
4. Read one CC-v1 parquet schema (`text`, `source_text`, `source_uuid`, `language_code`, `license`, `overall_score`) on a machine that is allowed to see the gated files. Confirm `source_uuid` is stable across languages before relying on the uniqueness cap.
5. Spot-check fertility, with the stock tokenizer, for Santali and Manipuri samples, so the exclusion of those configs is measured rather than only suspected. Do not write a new vocab.
6. List the exact eval item fields that the decontamination pass will hash, from the datasets already named in `eval_indic_mcq.py` and `eval_indic_gsm8k.py`.
7. Treat incomplete eval JSON as a rerun list: ARC Base (`ml`, `or`, `te`), MMLU-Indic Base (everything except English), MMLU-Pro Base (most languages). Do not impute those cells.
8. Choose the pilot GPU family from machines that can hold roughly 27 GB of Adam state plus eager activations. Keep the A10G probe boxes out of the training plan.
9. After acceptance, and only then, implement under this folder: an external-manifest writer, a single-GPU train loop, and a copy of the eval harness that accepts a local path. That implementation is not authorized by this draft.

## 9. Dependencies still open

| Dependency | Why it blocks | Owner |
| --- | --- | --- |
| Gated-data acceptance | CC-v1 cannot be sampled until a person agrees | Human |
| Replay corpus names, paths, licenses | 50% of the mix is unspecified on disk | Human |
| Base commit pin and weight hash versus instruct | Parent identity | Next read-only pass |
| `source_uuid` uniqueness across the ten configs | The dedup rule depends on it | Schema sample |
| transformers 4.57.1 on the intended GPU | Remote code imports version-specific mask helpers | Smoke test, after approval |
| Peak memory and tokens/s | 100B cost and the legal sequence length | Step D |
| Decontamination list | Task gains are untrustworthy without it | Sample prep, after approval |
| Surgery go / no-go | Determines whether the 8B checkpoint can be a parent | Human, after the pilot |
| SFT/DPO corpora and recipe | Not in this repository | Separate project |
| Spark-aware ternary recipe | Terminus QAT does not apply | Separate project, after dense CPT and post-training |
| Multi-GPU valid-token reduction | Required only if 100B is multi-GPU | After the single-GPU pilot |
| Original Spark pretraining tokens and LR | Would tighten the LR grid | Unpublished on the card that was read; the grid does not wait for it |
| India AI software image | The production CUDA/DeepGEMM image is the wrong image | Human, when a machine exists |

## 10. Compatibility with the two later projects

**Vocabulary work.** This plan never rewrites `tokenizer.json`, merges, or the embedding. Sequences are stored with today’s ids. A surgery project can start from the original Base checkpoint without undoing a resize, because no resize happened. The cost of that safety is that the pilot, and a stock-tokenizer 100B run, spend tokens on a vocabulary that is weak for Indic. Probe 4’s named script buckets contain 4,743 Indic-script pieces (Bengali, Devanagari, Gujarati, Gurmukhi, Kannada, Malayalam, Odia, Tamil, Telugu), which is 3.6% of 131,072, against 33,955 pieces in the named CJK buckets. The 20,897 `other_mixed` pieces are not in that Indic count, so 3.6% is a bucket sum, not a proof that nothing else can spell an Indic word. Probe 2’s tokens-per-word rates are the practical consequence. If that weakness is unacceptable, the right time to pay for surgery is before the 100B run, not after it.

**Ternary / 1.58-bit.** CPT stays in bf16 with a tied embedding and the vendor module names. Exports include `modeling_spark.py` and `configuration_spark.py` at the pinned hash so a converter sees `q_k_v_proj` and `g_proj` and the two RoPE regimes. No Terminus `weight_mode` is written into the checkpoint. The dense file remains the parent until a Spark-specific conversion has its own numerical tests. Those tests are out of scope here.

## Sources

Repository evidence:

- `production/README.md`, `production/recipe.py`, `production/ARCHITECTURE.md`, `production/train.sh`, `production/document_packing.py`, `production/PACKING.md`, `production/OPERATIONS.md`
- `terminus/checkpoint.py`, `terminus/transition.py`, `terminus/task_eval.py`
- `Alternate direction/INFRA.md`, `Alternate direction/CPT_BENCHMARKS_DEEP_RESEARCH.md`
- `Alternate direction/results/probe1_repo_facts.json`, `probe2_fertility.json`, `probe4_embeddings.json`, `probe5_perplexity.json`
- `Alternate direction/results/evals/milu__spark_base.json` and the partial ARC, MMLU-Indic, and MMLU-Pro Base files
- `Alternate direction/scripts/eval_indic_mcq.py`, `eval_indic_gsm8k.py`

External, retrieved 2026-11-12:

- [XHToken/Spark-X2.6-1.7B-Base config.json](https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base/raw/main/config.json)
- [XHToken/Spark-X2.6-1.7B-Base modeling_spark.py](https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base/raw/main/modeling_spark.py)
- [XHToken/Spark-X2.6-1.7B-Base README](https://huggingface.co/XHToken/Spark-X2.6-1.7B-Base/raw/main/README.md)
- [XHToken/Spark-X2.6-1.7B model card](https://huggingface.co/XHToken/Spark-X2.6-1.7B)
- [Spark-X2.6-1.7B architecture summary](http://slm.expert/models/spark-x2-6-1-7b/) (parameter total and architecture cross-check)
- [nvidia/Nemotron-Pretraining-Indic-CC-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Indic-CC-v1)
- [nvidia/Nemotron-Pretraining-Indic-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Indic-Specialized-v1)
- [NVIDIA, Scaling Indic Pretraining](https://huggingface.co/blog/nvidia/scaling-indic-pretraining)
