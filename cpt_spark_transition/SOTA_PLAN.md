# Complete CPT engineering plan

Date: 2026-11-12. Status: the CPT plan to build. Code is still unwritten.

Vocabulary surgery, if accepted, happens first and is specified in [VOCAB_PLAN.md](VOCAB_PLAN.md). That cut keeps every Spark row for the seven target scripts. The learning-rate screen then starts from that checkpoint. This file is the training plan for whichever tokenizer passes its gates.

This replaces the schedule in [ENGINEERING.md](ENGINEERING.md). Data quality, licenses, and mixture design stay outside this folder. The shards you provide are the data. This plan is how those shards are trained, checked, resumed, and judged.

Two research reports sit beside this file. Numbers below are tagged **P** when a primary source published them for some other run, **D** when they are arithmetic, and **X** when applying them to Spark is our engineering choice. Spark’s own pretraining learning rate is unpublished. Nothing here is a hidden Spark hyperparameter.

- Recipe evidence: [research/cpt_recipe.md](research/cpt_recipe.md), interaction `trun_b2ed6650f0164e45864e0ca360e72cb0`
- Systems evidence: [research/cpt_systems.md](research/cpt_systems.md), interaction `trun_b2ed6650f0164e45977bfd69facdd25c`

## 1. What “best engineering” means for this model

The released Spark base is the function we adapt. The best system is the one that changes that function only through the optimizer, on a schedule we can interpret, and that can be resumed and exported without renaming tensors.

That rules out a stack of techniques that are state of the art for other models and wrong for this checkpoint:

| Technique | Why it stays off |
| --- | --- |
| Muon, as in XHToken’s Spark full-SFT examples | Those YAMLs are supervised fine-tunes at 4e-5, often with `use_muon: true` and warmup 0. They are not a CPT recipe. Adam betas in the same file do not mean every matrix is AdamW. [LlamaFactory Spark examples](https://github.com/XHToken/LlamaFactory) |
| QK-norm, Gemma logit soft-cap, a new tokenizer, untied embeddings | Each changes the forward pass before the first CPT step. [OLMo 2](https://arxiv.org/html/2501.00656v3), [Gemma 2](https://arxiv.org/html/2408.00118v3) |
| z-loss, max-z, NEFTune, dropout, EMA-as-the-export | Either the coefficient is disputed, or the method is for instruction tuning or scratch pretraining. Off until a measured failure needs one named fix. |
| FP8 training | A 70B CPT study saw FP8 loss spikes and worse code/math. A later 1.5B/160B recipe matched BF16. FP8 is a separate matched-token experiment, not the pilot. [FP8 vs BF16](https://arxiv.org/html/2411.08719v1), [FP8 reasoning recipe](https://arxiv.org/html/2509.22536) |
| Megatron or TorchTitan rewrite | Neither loads this remote-code model unchanged. A 1.7B run does not need tensor or pipeline parallelism to reach 100B tokens. |
| Document packing in the pilot | Spark’s published forward does not isolate documents. Full layers would attend across packed documents. Packing is a later, tested change. |
| Continuing a fully cooled 3B checkpoint into a 100B cosine | At the same token count, that model has seen a different learning-rate integral than a 100B cosine. [Ibrahim et al. 2024](https://arxiv.org/pdf/2403.08763) |

What we do take from the literature:

- Full-parameter AdamW, betas 0.9 and 0.95, epsilon 1e-8, clip 1.0, matrix weight decay 0.1. Repeated on Swallow, Sailor, and Gupta’s rewarming study. **P**, applied to Spark as **X**. [Swallow](https://arxiv.org/pdf/2404.17790), [Sailor](https://arxiv.org/html/2404.03608v1), [Gupta et al.](https://arxiv.org/pdf/2308.04014)
- Weight decay 0 on the tied embedding. OLMo 2 removed embedding decay after decay drove embedding norms into a bad regime. The tied tensor is one Adam state, trainable at the global learning rate, not frozen. **P** mechanism, **X** on Spark. [OLMo 2](https://arxiv.org/html/2501.00656v3)
- A three-point learning-rate screen on one shared horizon, then a separately fitted long run from the original base. Peaks 1e-5, 3e-5, and 1e-4 are the published anchors from Gnani’s selected CPT peak, Tower’s 3e-5, and Sailor-1.8B’s 1e-4. They are not 0.5× Spark’s original peak, because that peak is unknown. [Gnani Evon](https://www.gnani.ai/resources/research/evon-v3-3-building-sovereign-llms-for-indian-languages), [Ibrahim](https://arxiv.org/pdf/2403.08763), [Sailor](https://arxiv.org/html/2404.03608v1)
- Equal replay. Nemotron-Mini-Hindi used 200B Hindi and 200B English. Typhoon 2 used 50% English in CPT. Your 50/50 shards match that precedent. The trainer does not rebalance them. [Hindi Mini](https://aclanthology.org/2025.indonlp-1.6.pdf)
- Select on retention, not on Indic loss alone. Gnani’s sweep preferred 1e-4 on perplexity and then switched to 1e-5 once English retention was in the criterion. Hindi Mini’s base-to-base English MMLU moved 58.60 to 56.37 and ARC-Challenge 50.90 to 46.08, with equal replay. Those drops are a warning, not our acceptance threshold. [Gnani](https://www.gnani.ai/resources/research/evon-v3-3-building-sovereign-llms-for-indian-languages), [Hindi Mini](https://aclanthology.org/2025.indonlp-1.6.pdf)
- Fused linear cross-entropy so a 131,072-way logit row is never materialized. At sequence 4096 that row is 1 GiB of bf16 per sequence before the loss workspace. **D**. Liger-Kernel 0.8.4 accepts the embedding parameter directly. It is turned on only after it matches a reference loss and a reference tied-weight gradient.
- BF16 compute, explicit fp32 master weights and Adam moments. Casting the module and hoping autocast is enough is unsafe: Spark’s forward casts activations to the weight dtype.
- One portable Hugging Face export plus one resume file. Later ternary work and vocabulary work must be able to open the export without this trainer.

## 2. The run, end to end

```text
E0  CPU contract, labels, cosine clock, checkpoint tests
E1  Load pinned Base. Memory at 2048 and 4096. Reference loss vs fused loss
E2  Two optimizer steps. Kill, resume, match the next step
E3  Letter-logit scorer checked against lm-evaluation-harness on the same items
        |
        human launches, one GPU
        v
Screen  three arms, same 3B-token shard stream, same batch, same cosine shape
        peaks 1e-5, 3e-5, 1e-4, each decayed to 0.1x by token 3B
        |
        human picks a peak, or declares the screen inconclusive
        v
100B    NEW run from the original Base
        fresh Adam, winning peak, cosine fitted to 100B, warmup 1B tokens
        DDP only after a single-GPU resume and a two-rank loss-scale check
```

The screen checkpoints are evidence. They are not the parent of the 100B run. A cooled short run has already spent its learning rate. Restarting from the released base is the Ibrahim-style rewarm: the long schedule is fitted to the long budget. [Ibrahim et al.](https://arxiv.org/pdf/2403.08763)

Compute for the screen is 9B token-presentations. Unique tokens are 3B, because every arm reads the same stream. The 100B run is additional. Wall-clock is `tokens / measured_scored_tokens_per_second`, plus eval and checkpoint time. There is no measured Spark training rate in the research, and this plan does not invent one.

An early drop on a task score is not, by itself, a stop. The stability-gap work saw task scores fall and recover while domain perplexity improved, and an OpenLLaMA-3B decline lasted through about the first 5B tokens. [Guo et al.](https://aclanthology.org/2025.acl-long.1578.pdf) Non-finite loss or gradients stop the run immediately. A retention failure at the end of a fully decayed arm removes that arm. If every arm fails the retention gate, the 100B run does not start.

## 3. Recipe

### Shared optimizer

| Knob | Value | Tag |
| --- | --- | --- |
| Parameters | All of them, including the tied embedding and `g_proj` | Fixed |
| Optimizer | AdamW, fresh moments at the first CPT step | **P** Ibrahim’s open-weight setup; **X** here |
| Betas, epsilon | 0.9, 0.95, 1e-8 | **P** Swallow, Sailor, Gupta |
| Clip | Global norm 1.0, after accumulation, before the step | **P** |
| Matrix decay | 0.1 on `q_k_v_proj`, `out_proj`, `g_proj`, `gate_proj`, `up_proj`, `down_proj` | **P** on ordinary matrices; **X** on Spark’s gate |
| Embedding decay | 0, one Parameter, same learning rate as the matrices | **P** OLMo 2 mechanism; **X** |
| Norm decay | 0 | **X** |
| Precision | bf16 activations, fp32 master and Adam state | **X**, see FP8 evidence above |
| Aux loss | none | **X** |

Gnani’s bug is the reason the embedding group is tested, not assumed. Decoupled AdamW decay updated embedding rows that had zero gradients, and they had to snapshot and restore rows. Our case has no new tokens, but the same optimizer will move unused rows if decay is on. Decay 0 is the fix. A step-1 check still asserts that a special-token row absent from the batch is bitwise unchanged. [Gnani](https://www.gnani.ai/resources/research/evon-v3-3-building-sovereign-llms-for-indian-languages)

### Learning-rate screen

Each arm trains on the same 3B scored tokens. **X** horizon: long enough that a stability-gap dip is not the whole experiment, short of the 100B commitment. If a hard cap forces a smaller screen, use 1B per arm and treat it as a stability check, not a ranking of the 100B learning rate.

Cosine in scored tokens, warmup `W`, horizon `T`, peak `p`, terminal fraction `a = 0.1`:

```text
lr(t) = a*p + 0.5*(1-a)*p*(1 + cos(pi*(t-W)/(T-W)))    for t >= W
```

`a = 0.1` is the cosine end-ratio used in Ibrahim’s listed configurations. **P** shape, **X** on this horizon. [Ibrahim](https://arxiv.org/pdf/2403.08763)

| Arm | Peak | Terminal at 3B | Anchor |
| --- | --- | --- | --- |
| Retention-first | 1e-5 | 1e-6 | Gnani’s selected CPT peak; also Occiglot and SEA-LION v3 at 1e-5 |
| Middle, run this one first | 3e-5 | 3e-6 | Tower’s published pair |
| Adaptation-first | 1e-4 | 1e-5 | Sailor-1.8B constant 1e-4 after warmup; screened here, not the default |

Pilot warmup is 100 optimizer updates, linearly from 0 to the peak. Fanar Prime warmed up for 100 steps. At Ibrahim’s batch that is about 226M tokens, 7.5% of 3B. A 1% warmup of a 3B run would be only about 13 updates at that batch, which is a weak rewarm. **P** step count, **D** tokens, **X** here. [Fanar](https://alt.qcri.org/~ndurrani/pubs/fanar.pdf)

Only the peak and the matching terminal rate change between arms. Batch, data order, precision, and groups stay fixed. The winner is the lowest peak that still moves Indic holdout NLL and the Indic task panel without breaking the retention gate. Best Indic perplexity alone is the criterion Gnani had to reverse.

### Batch

Literature global batches for this kind of run sit near 1.5M to 4M tokens per update (DictaLM about 1.5M, Ibrahim’s listed setup 1,104 × 2,048 = 2,260,992, Sailor and Swallow 4M). **P** and **D**. Those batches assume many GPUs.

Rule: measure one microstep, then lock one global batch for the screen and for the 100B run. Candidates, in tokens per update:

`262144, 524288, 1048576, 2260992`

Pick the largest candidate whose projected wall-clock for one 3B arm is acceptable, after the E1 timing. Microbatch starts at 1. Sequence length starts at 2048 in the memory smoke and becomes 4096 if the 4096 step fits with margin. Sailor-1.8B and Swallow both trained at 4096. **P** context, **X** until the smoke passes. Changing the batch after the screen changes what the winning learning rate means, so the 100B run inherits the locked batch.

Scored tokens, not pad slots, advance the schedule and the budget. If your shards are already dense fixed-length rows, padding waste may be near zero and packing buys nothing. NVIDIA’s “50–70% padding” figure is for variable-length RL and SFT, not for these shards. [NeMo RL packing note, cited in the systems report]

### The 100B run

| Knob | Value |
| --- | --- |
| Initialization | Original Base weights, not the screen checkpoint |
| Optimizer | Fresh AdamW |
| Peak | The retention-feasible screen winner |
| Horizon | 100B scored tokens |
| Warmup | 1B tokens, linear, which is 1% of the horizon. **P** Ibrahim’s listed 1% configuration; **X** here |
| Decay | The same cosine, recomputed so the 0.1× floor lands at 100B |
| Batch, sequence, groups, loss | Identical to the screen |
| Parallelism | One GPU until resume is proven, then DDP. FSDP2 or ZeRO only if a profile shows replicated state is the limiter |

At about 1.71B parameters, a conventional mixed-precision layout is 16 bytes per parameter: bf16 weight, bf16 grad, fp32 master, two fp32 moments. That is about 27.4 GB before activations. **D**. DDP replicates it. On an 80 GB GPU the pressure is eager attention and the vocabulary loss, not the parameter shard. Sharding is a measured optimization, not the default for 100B tokens.

Activation checkpointing of decoder blocks is allowed in the pilot if the memory smoke needs it. Spark’s recompute path passes hidden states, RoPE, and the mask. That is valid only while the mask still carries everything attention needs. KV cache stays off in training.

### Loss

Reference path, used to certify the fast path: Spark decoder, unchanged, then `F.linear` on `model.embedding.weight` and fp32 cross-entropy.

Training path, after the certificate: Liger-Kernel 0.8.4 fused linear cross-entropy on that same Parameter, with label smoothing, z-loss, and soft-cap left off. Fallback if Liger’s torch build does not match the pin: PyTorch 2.15 `F.linear_cross_entropy` with chunking enabled. `options=None` is the reference path and does not save the 1 GiB logit tensor.

The causal shift happens once. Pad labels are −100. Accumulation and, later, DDP divide by the global count of non-ignored labels, not by the mean of per-microbatch means.

Certificate, on a fixed batch that includes pads: loss, hidden-state gradient, tied-embedding gradient, and one Adam step agree with the reference within a tolerance written down before the run. Bitwise equality is not the bar. A fused kernel that fails the certificate is not used.

### Data surface

Unchanged from the engineering contract, with one addition from the loader research. Manifest and int32 memmaps live outside the repo. Each row is `[bos, content..., optional eos, pad...]`, content free of ids 0–13, `seq_len` 2048 or 4096 as locked. The trainer checks rows and stops. It does not mix, filter, or retokenize.

Loader: a deterministic map from sample id to shard and row. Start with zero DataLoader workers. Benchmark 2 and 4 only if the GPU is waiting on the host. Pin memory, non-blocking copies, a bounded prefetch. The checkpoint stores the sample id of the last *committed* update, not the producer’s prefetched position. A resume test compares the next sample ids against an uninterrupted run, with prefetch on.

TorchData 0.11.0 `StatefulDataLoader` is the candidate. Worker state restores only at the same worker count. Changing world size is a new sample assignment, not a silent reuse of per-rank worker state.

### Evaluation

Two scorers, cross-checked on the same items before either is trusted.

1. lm-evaluation-harness 0.4.13, Hugging Face backend, remote code on, task YAML and dataset revision pinned. This is the release number.
2. A letter-logit loop in this folder for frequent checks. Last non-pad position. Single-token labels compared as the actual ids of `" A"` through `" D"`. Multi-token labels are summed conditional log-probs, not the first piece. No chat template on a base checkpoint. `acc_norm` in the harness is character-length normalization of a continuation, not this letter protocol. Do not mix the two numbers.

Cadence, **X**, scaled to cost after the first measurement:

| When | What |
| --- | --- |
| Before any update | Full baseline on this GPU |
| Every 50M scored tokens in the screen | Holdout NLL by stream and by language |
| Every 250M in the screen | Small task panel: MILU English plus MILU Indic, one reasoning set, one math or code sentinel |
| End of each arm | Full panel |
| During 100B | Holdout NLL every 100M, small panel every 500M, full suite every 1B |

Holdout NLL is separate for Indic and for replay. A single mixture loss can fall while English gets worse. Report a paired uncertainty. A one-point wiggle on 1,000 items is not a decision.

NVIDIA’s 2026 Indic evaluation paper uses MILU and translated MMLU-Pro, ARC-Challenge, and GSM8K, plus English retention. Gnani’s table is a different model and a caution: MILU 40.63 to 53.52 while English MMLU-Pro 64.83 to 55.43. [Scaling Synthetic Pre-training Data](https://arxiv.org/pdf/2610.26006), [Gnani](https://www.gnani.ai/resources/research/evon-v3-3-building-sovereign-llms-for-indian-languages). Our thresholds stay a human gate. The screen does not auto-promote an arm.

### Checkpoints

Two files at every committed update boundary, after gradients are applied, with no partial accumulation left over:

| Artifact | Contents |
| --- | --- |
| Hugging Face export | Safetensors via the model’s `save_pretrained` path so the tied embedding is stored once, config, the exact remote-code files, the stock tokenizer, hashes |
| Resume state | fp32 master, Adam, token clock, learning-rate peak and horizon, RNG, committed sample id, loss-implementation id, environment lock |

Write to a temp directory on the same filesystem, fsync, hash, then rename. `latest.json` moves only forward. Rank 0 does not call a collective checkpoint API alone. PyTorch distributed checkpoint, if DDP or FSDP is added, uses canonical parameter names. PyTorch 2.15 checkpoint metadata is not readable by 2.14. The Hugging Face export is the escape hatch. Restarting from that export is a new optimizer, not an exact resume.

## 4. Runtime pin

These versions are release snapshots from the systems report, not a combination that has been installed together. The lockfile is written when E1 first succeeds, and then it is frozen.

| Piece | Pin |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.15.0, with the CUDA build that matches the machine |
| Transformers | 4.57.1, because the Spark config names it. Transformers 5.22.2 is current and is not the pilot |
| Liger-Kernel | 0.8.4, off until the loss certificate passes |
| Eval harness | 0.4.13 |
| Model | Pinned Base revision, hashed `modeling_spark.py` and `tokenizer.json` |

Do not install a FlashAttention wheel built for a different PyTorch. Do not set `attn_implementation="flash_attention_2"` and expect Spark’s eager forward to change. It will not.

## 5. Code, still only in this folder

```text
cpt_spark_transition/spark_cpt/
  contract.py     manifest, row rules, hashes
  batch.py        mask, labels, scored-token count, one shift
  schedule.py     token cosine, identical across arms except peak
  optim.py        fp32 master AdamW, tied parameter once, decay groups
  loss.py         reference CE and Liger path, certificate
  checkpoint.py   export plus resume, atomic publish
  load.py         from_pretrained, tie assert, hash assert
  data.py         memmap cursor, committed sample id
  loop.py         accumulate, clip, step, jsonl
  eval_mcq.py     letter logits, local export
  screen.py       three-arm command builder, does not launch
```

No imports from `production` or `terminus`. No writes outside this folder. Output paths inside the git work tree are refused.

| Milestone | Done when |
| --- | --- |
| E0 | CPU tests for rows, labels, cosine values, scored-token counts, and a checkpoint round-trip on a tiny module |
| E1 | Real Base loads; tie holds; 2048 and 4096 peak memory and seconds per microstep are written down; sequence length and global batch are locked; reference loss matches fused loss or fused loss stays off |
| E2 | Two real steps, finite grads, unused special row unchanged, kill and resume match including the next sample id |
| E3 | Letter scorer and the harness agree on a fixed MILU slice of the untouched Base |
| E4 | A person can point `screen.py` at a manifest and get three external run directories. The script does not start them |
| E5 | A person can start the 100B configuration from the original Base and the chosen peak. The script refuses a screen checkpoint as the weight initialization |

E0 is the first slice to write. E1 through E3 are the correctness gate. E4 and E5 do not spend tokens by themselves.

## 6. Later systems work, each behind a numeric gate

These are how the 100B run gets faster after the pilot is correct. None of them is part of E0–E3.

1. **Fused linear loss**, once the certificate in section 3 passes. This is the first speed and memory win, because the logit tensor dominates.
2. **DDP** across GPUs that already fit the microstep. The valid-target sum is all-reduced. A two-rank test must match a single-process update that saw the same tokens.
3. **Attention kernel**, hand-integrated, gate left as the existing sigmoid multiply outside the kernel. FlashAttention-3 is the H100 candidate. Window argument must match Spark’s 512, inclusive versus exclusive checked against eager. Forward, backward, and one optimizer step compared to eager. FlexAttention stays a prototype and is not the first kernel.
4. **Document packing**, only if measured pad fraction or a profile says the GPU is idle inside padding. Required together: document bounds, causal and same-document and window masks, loss cut at the document boundary, and those masks passed through gradient checkpointing. A constant RoPE offset inside an isolated document cancels in the QK product, so position reset is optional once the mask is real. Position reset without the mask does not isolate documents. Each packed row is checked against the same documents run alone.
5. **FSDP2 or ZeRO-1** if replicated Adam state, not activations, is the measured ceiling. Tied embedding stays one parameter group. Canonical names stay stable so the export path does not change.
6. **FP8** only as a matched-token branch against the bf16 run, with its own spike log and the same retention panel.
7. **A base/CPT weight interpolation** as a named sibling, ` (1-c) * base + c * cpt `, never as the default parent. Branch-and-Merge is evidence that merging can trade retention against adaptation. It does not supply `c` for Spark. [Branch-and-Merge](https://aclanthology.org/2024.findings-emnlp.1000.pdf)

Vocabulary surgery and ternary conversion remain later projects. This plan keeps their inputs possible: stock tokenizer bytes, tied embedding, original module names, bf16 export, modeling file hash in the checkpoint. No `resize_token_embeddings`. No `weight_mode`.

## 7. Invariants

1. Tokenizer sha256 matches the manifest. The tokenizer is never saved over.
2. `resize_token_embeddings` is never called.
3. The embedding Parameter and the classifier weight are the same storage, and Adam lists that storage once.
4. Embedding and norm weight decay are 0.
5. A row with an id outside `0 .. 131071`, or a special id in the content span, aborts.
6. Padding does not change the loss on real labels.
7. The schedule and the budget count scored tokens.
8. Resume refuses a different manifest, tokenizer, sequence length, peak, horizon, or loss implementation.
9. The 100B command refuses screen weights as its initialization.
10. The output directory is outside the git work tree.
11. Modeling-file hash is stored on every export.

## 8. What is still unknown, on purpose

- Spark’s original pretraining peak. The grid is absolute, not a fraction of that peak.
- Whether 1e-5, 3e-5, or 1e-4 wins. The screen exists because the papers disagree, including Gnani’s reversal from 1e-4 to 1e-5.
- Tokens per second, the legal sequence length, and the legal global batch. E1 measures them.
- Whether 3B tokens is past the stability gap for this model. A task dip inside the screen is recorded, not auto-killed.
- English acceptance limits. Hindi Mini’s roughly 2-point MMLU and 5-point ARC drops are observed on a different 4B-class model. A person sets Spark’s gate before the 100B launch.
- Long-context behavior. Training at 2048 or 4096 does not measure the 1,048,576-token configuration. The config value stays. We do not train at 1M.

## 9. First build

When you say to implement, the only code written is E0: contract, label mask, token cosine, checkpoint round-trip, and CPU tests. No weights, no shards, no GPU, and no files outside `cpt_spark_transition/`.
