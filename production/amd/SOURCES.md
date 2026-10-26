# Checked sources and evidence boundaries

Accessed 2026-10-24. Links to executable vendor code use the inspected commit
where possible. These establish APIs, advertised capabilities and proposed
build inputs, **not** CI-1 GPU performance. Raw downloaded pages and API
responses were inspected outside the repository; none is a trial result.

| ID | Primary source | What was checked / implication |
| --- | --- | --- |
| S1 | [AMD MI325X product specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html) | 256 GB, 6 TB/s, 256 MB LLC, 304 CUs, dense FP8/BF16 vs sparse rates, OAM, 1000 W TBP. Does not specify a cloud server's actual sustained clocks or networking. |
| S2 | [PyTorch HIP semantics, v2.13 source](https://github.com/pytorch/pytorch/blob/v2.13.0/docs/source/notes/hip.rst) | ROCm intentionally uses `torch.cuda`, CUDA device strings/events, HIP detection and caching allocator. No global cuda→hip string replacement. This document is not a binary compatibility certification for our selected 2.12 image. |
| S3 | [AITER attention, pinned `ddf072f`](https://github.com/ROCm/aiter/blob/ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3/aiter/ops/mha.py) | `flash_attn_func` signature, bottom-right causal/window semantics, GQA support, RTNA default and RTNE selector, gfx942 assembly dispatch excludes SWA, CK fallback. Commit timestamp observed: 2026-10-24 02:56:09 UTC. |
| S4 | [AITER GEMM dispatch, same pin](https://github.com/ROCm/aiter/blob/ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3/aiter/ops/gemm_op_a8w8.py) | `gemm_a8w8_blockscale`, FP32 128×128 weight scales, BF16/FP16 output for this path, configuration-first CK/CKTile/Triton selection. This API is not our FP32 1D×1D wgrad. |
| S5 | [AITER Triton block GEMM](https://github.com/ROCm/aiter/blob/ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3/aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py) | Logical activation scale shape `[M,K/128]`, weight scale shape `[N/128,K/128]`, scale strides passed to kernels, distinct tile/split choices. Used to avoid a mistaken shape transpose at the adapter boundary. |
| S6 | [ROCm TE v2.17 README](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/README.rst), [block tensor quantizer](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py), [storage wrapper](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/transformer_engine/pytorch/tensor/storage/float8_blockwise_tensor_storage.py), [GEMM API](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/transformer_engine/pytorch/cpp_extensions/gemm.py), [exact GEMM tests](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/tests/pytorch/test_float8_blockwise_gemm_exact.py) | Tag resolves to `130099ce48dc806398611de9960f8e0c486ad5f1`. Documents HipKittens gfx942/gfx950 block FP8 1D×2D/1D×1D and FP32 output. Quantizer defaults to power-of-two scales; ROCm physical 1D scale layout is `[K/128,M]`, 2D `[N/128,K/128]`, without NVIDIA padding. Adapter preserves our own quantization. |
| S7 | [ROCm 7.14 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html), [ROCm releases API](https://api.github.com/repos/ROCm/ROCm/releases?per_page=4) | Verified the 7.14 release family exists. Unversioned `/latest` documentation can describe a different newer release; it is not a lockfile. |
| S8 | [RCCL usage/tuning](https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html), [rccl-tests](https://github.com/ROCm/rccl-tests) | RCCL profiling/topology and channel/partitioning concerns. MI350X/MI355X 2.30.4 channel defaults are hardware-specific; do not copy them as MI325X recommendations. Existing public performance thresholds are not measured CI-1 results. |
| S9 | [Docker Hub ROCm 7.14 tags API](https://hub.docker.com/v2/repositories/rocm/pytorch/tags?page_size=100&name=rocm7.14) | Observed `rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0`, digest `sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1`, updated 2026-09-01. Pin used in Dockerfile. Image was not pulled or rebuilt locally. |
| S10 | [ROCm PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html) | Vendor image workflow; documentation examples and current tags differ, so choose a real tag/digest rather than combining independently plausible torch/ROCm versions. |
| S11 | [FlashAttention AMD documentation](https://github.com/Dao-AILab/flash-attention#amd-rocm-support), [pinned v2.8.3.post2 setup](https://github.com/Dao-AILab/flash-attention/blob/49774b31c8d1172a5b28f72e9aed0f322bf4ccba/setup.py) | CK forward/backward BF16 support, build flags, head-dimension pruning. The README calls sliding windows work-in-progress in its Triton backend. Build explicitly selects CK with backward, not a Triton attention fallback. |
| S12 | [AITER README at pin](https://github.com/ROCm/aiter/blob/ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3/README.md), [requirements](https://github.com/ROCm/aiter/blob/ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3/requirements.txt), [setup](https://github.com/ROCm/aiter/blob/ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3/setup.py) | gfx942 support, `AITER_USE_SYSTEM_TRITON`, pinned Python dependencies/FlyDSL, `PREBUILD_KERNELS`, worker caps. `PREBUILD_KERNELS=0` avoids GPU-executing build pretests; kernels still need native preflight/JIT. |
| S13 | [AITER dtype utilities](https://github.com/ROCm/aiter/blob/ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3/aiter/utility/dtypes.py) | Architecture-dependent FP8 dtype selection. New local kernels target FNUZ explicitly and reject gfx950 instead of pretending it has the same FP8 encoding. |
| S14 | [TE setup](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/setup.py), [build utilities](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/build_tools/utils.py), [CMake](https://github.com/ROCm/TransformerEngine/blob/130099ce48dc806398611de9960f8e0c486ad5f1/transformer_engine/common/CMakeLists.txt) | `NVTE_USE_ROCM`, `NVTE_FRAMEWORK`, `NVTE_CMAKE_EXTRA_ARGS`, `USE_HIPKITTENS_GEMM`, `CMAKE_HIP_ARCHITECTURES`. TE attention is disabled in the optional build since the model uses AITER/FA-CK directly. |
| S15 | [hipBLASLt offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html) | Shape/algorithm tuning is separate from switching precision. Use for BF16 head/optimizer paths only after profiling; freeze tuned artifacts and account for tuning time. |
| S16 | [rocprofv3 usage](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) | HIP/kernel tracing and profiling workflow. Counter access on a cloud VM must be tested; profiled timings are separate from acceptance timing. |
| S17 | [Pinned Muon](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806), [local spectral derivation](../../docs/MUON_SPECTRAL_IMPLEMENTATION.md) | Exact upstream NS5 and local spectral coefficient/decay/Adam rules. No optimizer replacement in this port. |
| S18 | [Selected production recipe](../recipe.py), [architecture review](../ARCHITECTURE.md), [original parallelism handoff](../PARALLELISM.md), [H100 transfer record](../../docs/speed/TRANSFER_32GPU.md) | Authoritative model and optimizer choices; identifies original shard-validation scale issue and the scope of historical H100 timing. |
| S19 | [PyTorch releases API](https://api.github.com/repos/pytorch/pytorch/releases?per_page=3) | 2.13 and later releases exist. Existence of an upstream release does not verify the corresponding ROCm image/wheel/compiler/native-extension combination. |

Additional configuration-selection research, accessed **2026-10-25**:

| ID | Primary source | What was checked / implication |
| --- | --- | --- |
| S20 | [SmolLM2-1.7B training YAML, commit `438be18`](https://github.com/huggingface/smollm/blob/438be18dae658a638f3a6976fdf5190bb6d68619/text/pretraining/smollm2/config_smollm2_1B.yaml), [SmolLM2 paper §4 and Appendix A](https://arxiv.org/html/2502.02737v1) | Despite the filename, YAML run name is `smollm2-1700M`: DP 256 × microbatch 4 × sequence 2048 × accumulation 1 = 2,097,152 tokens/update. 2,000-step warmup; 524,000 decay steps out of 5,240,000; AdamW 5e-4, linear decay to zero. Supports our batch scale and 10% decay as starting choices; does not establish Muon transfer, a 4096-context optimum or AMD speed. |
| S21 | [MiniCPM paper, §3.2 and §4](https://arxiv.org/html/2404.06395v2) | Batch selection trades update efficiency against data efficiency; WSD separates exploration from final decay. Its empirical scaling experiments and optimizer do not yield a validated CI-1/Muon batch law. Its flexible-horizon theory does not override our exact-resume contract. |
| S22 | [McCandlish et al., An Empirical Model of Large-Batch Training](https://arxiv.org/html/1812.06162v1) | Gradient noise scale predicts a largest useful batch, which varies over training. This supports testing quality at matched tokens/compute instead of choosing global batch from HBM capacity. Not a measured CI-1 gradient noise scale or Muon-specific guarantee. |
| S23 | [Archived local HPO profile](../../hpo/MODEL_PROFILE.md), [width bridge](../../hpo/bridge.py) | 393,216 tokens/update (96 × 4096); ~250M warmup in the archived schedule. Provides the batch-transfer control. These files describe prior research and do not govern the selected production geometry or replace `production/recipe.py`. |
| S24 | [PyTorch DDP implementation/docs at v2.12.0](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/parallel/distributed.py) | 25 MiB default bucket cap, overlap purpose, gradient bucket-view behavior. Matches the pinned image's torch release. Does not demonstrate that 25, 64 or 128 MiB wins on the provider's eight-GPU topology. |

The current [TinyLlama script at `bf12224`](https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/pretrain/tinyllama.py)
was also inspected. Its checked-in global sequence batch differs from widely
quoted training descriptions; it was not used to assert a final TinyLlama
production batch. No third-party throughput number was transferred to CI-1.

## Dense/QAT allocation research

Follow-up sources inspected **2026-10-25**:

| ID | Primary source | What was checked / implication |
| --- | --- | --- |
| S25 | [ParetoQ §2.1](https://arxiv.org/html/2502.02631v1) | MobileLLM-125M fixed 100B-token allocation experiment favors roughly 90/10; separate LR cycles. Its quantizers and gradients differ from our absmean/identity-STE recipe. |
| S26 | [Continual QAT §4–5](https://arxiv.org/html/2502.11895v1) | OLMo-1B, 10k updates at 4,194,304 tokens/update; 2k-step switch best among tested 2k/4k/6k switches. State carry/ramp reduce shock without guaranteeing a final advantage. |
| S27 | [MiniCPM4 / BitCPM4 §3.3](https://arxiv.org/html/2506.07900v2) | 5M / 2B-token pilot, 150M verification; QAT share exceeding ~40% approaches ternary-from-scratch loss. Weight-only QAT, LR re-warmup and 350B-token final QAT. Not proof of dense parity or a universal “twice decay” law. |
| S28 | [BitNet b1.58 2B4T](https://arxiv.org/html/2504.12285v1) | Native ternary training from scratch on 4T tokens; does not establish the best CI-1 phase split or GPU training speed. |
| S29 | [Archived Lean 150M dense/QAT review](../../experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md) | Reports incomplete quality recovery after 10B dense + 2B QAT; different architecture/LRs and changed QAT mixture. Summary inspected, raw training not rerun; no causal conclusion that duration alone caused the gap. |

These motivate the initial [60/40 planning policy](QAT_ALLOCATION.md); none
certifies the selected model's AMD QAT convergence or an optimal split.

## Batch-transfer research

Follow-up sources inspected **2026-10-26**:

| ID | Primary source | What was checked / implication |
| --- | --- | --- |
| S30 | [Dang, Wen and Malladi, *The Best Optimizer Depends on Batch Size*, v2](https://arxiv.org/pdf/2610.08975v2), §§3.2 and E.3, Tables 4–7 | Their 12-layer, width-768, 1024-context FineWeb model uses MuonW for hidden matrices and AdamW for other parameters. Among 216 transfer rules from 131,072 to 2,097,152 tokens/update, the best common language-model rule kept both LRs and Muon momentum fixed, scaled matrix weight decay with batch, and transported Adam second-moment retention. The best rule changed with target batch and task; even the best 2M rule did not erase the reference-batch loss difference at 1.7B tokens. Different architecture, Newton–Schulz iterations, auxiliary rates, data and LR schedule prevent copying its numerical result into CI-1. |
| S31 | [Naganuma et al., *Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales*, v2](https://arxiv.org/html/2602.03001v2), §§4–6 | Its Muon experiments on a 160M Llama model support optimizer-aware batch diagnostics, but §6 explicitly leaves stateful momentum and mixed Muon/AdamW extensions open. A Euclidean gradient-noise estimate alone cannot certify CI-1's 2M batch. |
| S32 | [Malladi et al., *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*, v3](https://arxiv.org/html/2205.10287v3), §5 | The square-root LR and coordinated moment/epsilon rule is derived for RMSprop/Adam under its assumptions, not for Muon plus AdamW with CI-1's spectral update and token-based WSD. It is not an automatic batch-transfer formula for this recipe. |
| S33 | [Sato et al., *Convergence Bound and Critical Batch Size of Muon Optimizer*, v5](https://arxiv.org/html/2507.01598v5), §4 | Muon's critical batch depends on gradient variance, target precision and effective rank; the paper does not provide an absolute critical batch for our workload. |

See [the bounded transfer decision](BATCH_STABILITY.md) for how these sources
affect the selected 393k, 1M and 2M comparisons.

## Claims from the pasted dossier that were not promoted to facts

- Its 43,873.9 tok/s H100 receipt, power/clock curves and per-phase timings
  need the external raw records. Only the consolidated writeup was attached.
- 45–55k AMD, 60–65k stretch and MI355X 85–110k are modeled scenarios,
  not measured CI-1 rates. No such estimate is embedded in a launch decision.
- Library support does not prove every production shape, GQA layout, cached
  offset or backward passes; the native preflight tests those explicitly.
- Vendor/third-party MLPerf, LoRA, large MoE and inference speedups concern
  different workloads. They do not predict this 1.446B dense model's result.
- Provider pricing, server inventory, cooling and NIC topology were not
  needed to implement this handoff and have not been independently verified.
- “All kernels compile,” “no feasibility blocker,” “32-GPU overlap is 57 ms,”
  and a particular training launch date remain unproven until native tests.
- MI355X is a different gfx950/OCP target; it is not silently enabled by this
  gfx942 migration. The user's current plan is MI325X with eight GPUs.

The useful conclusion is a concrete, source-audited set of runnable candidates
and falsifiable tests. CPU checks can verify wiring and arithmetic accounting;
they cannot certify GPU code generation, library linking, speed or convergence.
