# Production implementations: what they say about depth and throughput

Read on 2026-10-09 from official model configs, official architecture code, and maintained training-framework sources pinned below. These are observations about **different models and training systems**, not matched Terminus trials or a throughput ceiling. The question for this campaign is whether the selected 25-layer stack-004 should become 30 layers before optimizing its dense 1.5B training path.

## Actual small-model geometries

| Released model and primary config | Layers | Width / FFN | Query / KV heads | Vocabulary | Relevant difference from Terminus |
| --- | ---: | ---: | ---: | ---: | --- |
| [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/8faed761d45a263340a0528343f099c05c9a4323/config.json) | 28 | 1536 / 8960 | 12 / 2 | 151,936 | Much larger vocabulary, narrower residual stream, wider FFN ratio |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json) | 28 | 2048 / 6144 | 16 / 8 | 151,936 | Different GQA and norm/attention recipe; named size is not a matched parameter count |
| [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/effd688a12921b4cc83e3312b6feb579f70f9c71/config.json) | 24 | 2048 / 8192 | 32 / 32 | 49,152 | Full multi-head K/V and a wider FFN; [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/effd688a12921b4cc83e3312b6feb579f70f9c71/README.md) says 11T pretraining tokens using Nanotron |
| [StableLM 2 1.6B](https://huggingface.co/stabilityai/stablelm-2-1_6b/blob/f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef/config.json) | 24 | 2048 / 5632 | 32 / 32 | 100,352 | Untied embeddings and full multi-head K/V |
| [Gemma 3 1B, official PyTorch definition](https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py) | 26 | 1152 / 6912 | 4 / 1 | 256,000 | Local/global attention, unusual head geometry and large tied vocabulary; official file defines architecture, not Google's training runtime |
| [Gemma 2 2B, same official definition](https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py) | 26 | 2304 / 9216 | 8 / 4 | 256,000 | Different parameter scale, attention pattern, head dimension and normalization |

The maintained designs span 24–28 layers near this size. This range is evidence that 25 is a normal design point, **not** evidence that a median or majority depth is optimal for Terminus. Vocabulary, FFN allocation, attention, normalization, optimizer, data and training horizon differ. Qwen's public config does not disclose an end-to-end training speed recipe; Google's PyTorch file is an architecture definition. Do not infer H100 tokens/s or matched quality from these configs.

## Training systems worth inspecting

| Primary implementation | Verified mechanism | What Terminus can test |
| --- | --- | --- |
| [Nanotron](https://github.com/huggingface/nanotron/blob/a1d47c7ef0f2442a25871e9d5aa4105313d1c37a/README.md), used by SmolLM2 | 3D DP/TP/PP, ZeRO-1, FP32 gradient accumulation, tied-parameter handling and CUDA-event timing; its README marks FP8 training as unfinished | Recheck gradient accumulation and tied-head accounting; consider ZeRO-1 only if replicated optimizer state blocks the fastest local batch. Its BF16-focused throughput cannot be transplanted to our FP8/Muon/Peri-LN stack. |
| [NVIDIA Megatron Core](https://github.com/NVIDIA/Megatron-LM/blob/ab1a28486b92adb3702f1289ff3a332cdb74294f/README.md) | FP8-capable building blocks; DP gradient-reduce/parameter-gather overlap, TP communication overlap, PP; its published H100 results include data, optimizer, communication and logging | On 32 GPUs, compare plain data parallel with gradient buckets ready during backward; add TP/PP/FSDP only if memory or measured network exposure warrants their communication and scheduling cost. The published large-model MFU is not a 1.5B stack speed baseline. |
| [PyTorch TorchTitan](https://github.com/pytorch/torchtitan/blob/41d0fe107e6fe1f036fad95288317c68833a9eb6/README.md) | Maintained large-scale training reference with FSDP2, TP/PP/CP, selective activation checkpointing, `torch.compile` and asynchronous checkpointing | Use its decomposition as implementation options; benchmark one at a time on our full update. A wholesale framework swap would not preserve custom value-residual, Peri-LN and Muon behavior automatically. |

Our current path already has FA3, DeepGEMM FP8, packed projections/MLP and a streaming loss. External stacks point toward **exposed** cast/scale traffic, saved activations, optimizer work, host gaps and 32-rank gradient readiness as testable remaining costs. They do not show that merely adding layers makes training more efficient.

## Near-matched 25-versus-30 physics check

The actual speed target is 25 layers, width 2304, FFN 5760, 18/6 heads, 65,536 vocabulary, 20 local plus 5 global layers: **1,500,456,496 parameters**. Its normalized config differs from the winning 200M stack only by tripling width, FFN width and both head counts; depth and all stack switches remain unchanged.

The earlier Terminus [architecture proposal](../terminus/architecture_review_20261005/ARCHITECTURE_AND_TUNING_PROPOSAL.md) suggests a *separate, untested* 30-layer alternative: width 2048, FFN 5632, 16/4 heads, same 65K vocabulary and 24 local plus 6 global layers. With the same Peri-LN and value residual its analytic count is **1,487,136,314** (0.89% fewer parameters). This also changes width, FFN allocation and GQA ratio, so even a matched trial is a geometry comparison, not a pure depth ablation.

| Optimistic H100 NVL service bound, per valid token | 25-layer target | Proposed 30-layer stack |
| --- | ---: | ---: |
| FP8 projection forward/backward GFLOPs | 8.0953 | 8.1160 |
| BF16 tied head GFLOPs | 0.9060 | 0.8053 |
| BF16 attention GFLOPs, exact valid pairs | 0.7787 | 0.8306 |
| Tensor-core-only minimum | 6.8624 µs | 6.8164 µs |

These numbers use the repository's [physics calculator](../../speed_research/physics.py) and dense H100 NVL rates. The near-equal floors **do not predict measured throughput**: both omit attention recompute, FP8 conversion, HBM traffic, optimizer, tile waste and launches. Thirty blocks create 20% more serial layer boundaries, but narrower GEMMs and a smaller tied head may recover some time. Only a full-update benchmark can decide speed; only matched learning and downstream quality can decide whether its extra depth is useful.

The [historical 30-layer 96K config](../../configs/1.5b_h100_throughput.json) is not that comparator: it predates Peri-LN/value residual and uses a different tokenizer, optimizer and update profile. Its [42.2K synthetic tokens/s](../terminus/H100_OPTIMIZATION.md) cannot be ranked against the older 25-layer 46.4K result as a depth experiment. Likewise, the stage-one stack's 2.5B-token English win was against a **25-layer baseline**, not a 30-layer candidate.

## What can transfer to 32 GPUs

For 1,500,456,496 FP32 gradients, each data-parallel rank has about 6.002 GB of gradient values. A 32-rank ring all-reduce moves approximately `2 × 31/32 × 6.002 = 11.63 GB/rank/update`; an effective per-rank bandwidth `b` GB/s gives a communication service bound of `11.63/b` seconds, before latency and fabric contention. This bound may overlap with backward only after each gradient bucket is ready. Megatron's overlap controls address that scheduling problem; no single-GPU trace can certify that it will work on the user's actual 32-GPU topology.

If the earlier 2,097,152-token global update is retained, 32-way data parallelism assigns **65,536 tokens/rank/update**. At 4096 context and microbatch eight, that is two local accumulation microsteps; timing 64 microsteps on one GPU is a different optimizer-amortization workload. Use the production local profile for speed transfer and keep the full-global single-GPU profile separately if needed for optimizer/quality study. Start with replicated data parallelism, then measure whether memory or exposed network time justifies ZeRO/FSDP, TP or PP. Their use in production codebases is evidence they are available tools, not that they win for a 1.5B custom stack.

## Decision for this speed pass

Keep **25 layers**. It is the actual selected stack and the cleanest way to isolate execution gains during the three-hour dense-only run. It is **not** proven the best depth for a 1.5B model trained for 1.5–2T tokens. If depth is reopened later, build the proposed 30-layer stack with the same 65K tokenizer, Peri-LN, value residual, dense numerical policy and data; retune depth-sensitive initialization; compare whole-update rate, memory and matched-token **and** matched-GPU-hour quality. Preserve both candidates as a Pareto comparison. The 32-GPU communication/overlap result and later QAT remain separate acceptance gates for either depth.
