# Terminus autoresearch

This is a **separate, private** Terminus architecture-research repository for a Kimi K3 autonomous loop. It adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to the maintained Terminus ModelTunnelv2 code. The agent researches and edits architecture candidates; a protected runner holds the data, tokenizer, training exposure, evaluation, and spending contract fixed. No architecture search has been launched from this repository.

The source snapshot and inspected autoresearch revision are recorded in [SOURCE_BASELINE.md](SOURCE_BASELINE.md). The rationale and readiness gaps are in [docs/AUTORESEARCH_DESIGN.md](docs/AUTORESEARCH_DESIGN.md); exact commands are in [docs/AUTORESEARCH_EXECUTION.md](docs/AUTORESEARCH_EXECUTION.md). Kimi's operating instructions are [program.md](program.md).

The three intended campaigns are **architecture discovery**, then joint architecture confirmation with hyperparameter/μP transfer and longer dense-to-ternary QAT, then kernel optimization for the chosen computation. Only the first campaign is implemented here. It uses English as a cheap architecture proxy and reports quality, learning efficiency, stability, throughput and memory separately. Default trials use dense FP8; a matched short QAT probe is optional for a specific quantization-sensitive mechanism. Its result is a shortlist, because a 200M English proxy and short QAT probe cannot prove the final 1.5B multilingual ternary ranking.

The editable architecture is [candidate.json](candidate.json) and the permitted model files. The [policy](research/policy.json), [preparation wrapper](prepare.py), [trial runner](train.py), and [scorer](research/protocol.py) are protected during a campaign. The runner uses the existing native DeepGEMM/FA3/Muon training stack and requires an H100-class SM90 GPU for paid trials. It does not create or stop cloud machines.

To inspect the interface without launching a run:

```bash
python prepare.py --help
python train.py --help
python -m research.loop --help
python -m unittest tests.test_research_protocol -v
```

The prepared English-only token manifest is an external input; English development tasks are optional. The example campaign and task manifests under `research/` are templates, not ready data. The [scorecard](docs/ARCHITECTURE_SCORECARD.md) defines how the loop compares candidates. The existing `eval/` records are deliberately not imported into Git. The previous source repo README remains at [docs/terminus/ORIGINAL_REPO_README.md](docs/terminus/ORIGINAL_REPO_README.md).
