# Research library and reusable sources

The sources below were inspected as research context. Paper PDFs are not distributed with this release; use the canonical links. They are supporting evidence, **not drop-in implementations** or claims that published hyperparameters transfer to Terminus.

| Source | Relevance |
| --- | --- |
| [MiniCPM4 / BitCPM4](https://arxiv.org/abs/2506.07900) | Sections 2.3 and 3.3: ModelTunnelv2 task-loss signal and the FP8-to-ternary BitCPM4 pilot. |
| [Densing Law](https://arxiv.org/abs/2412.04315) | ScalingBench construction and task-conditioned loss. |
| [Continual QAT](https://arxiv.org/abs/2502.11895) | Alternative dense/QAT transition and optimizer-state evidence. |
| [ParetoQ](https://arxiv.org/abs/2502.02631) | Ternary quantizer and transition alternatives; compare mechanisms rather than assuming its optimum. |
| [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466) | Conditions and diagnostics for hyperparameter transfer across width. |
| [Scalable Muon](https://arxiv.org/abs/2502.16982) | Muon shape/update scaling and optimizer context. |
| [Step-by-step transfer](https://arxiv.org/abs/2608.20061) | Recent transfer work relevant to Muon; its MoE setting is not direct Terminus validation. |
| [DoReMi](https://arxiv.org/abs/2305.10429) | Proxy-model data-mixture optimization ideas. |
| [RegMix](https://arxiv.org/abs/2407.01492) | Regression-based mixture search and cost-effective candidate ranking. |

Potential reusable software and task sources for the coding agent to inspect later: [ScalingBench](https://huggingface.co/datasets/openbmb/DensingLaw-ScalingBench), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [DataTrove](https://github.com/huggingface/datatrove), [Microsoft μP](https://github.com/microsoft/mup), [ParetoQ reference code](https://github.com/facebookresearch/ParetoQ), and [MILU](https://huggingface.co/datasets/ai4bharat/MILU). Check each source's current revision and license before integrating it.
