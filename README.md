# Terminus autoresearch

**Stack-004 speed research:** the new [Kimi K3 program](speed_program.md) and [framework](docs/speed/README.md) target 1.5T–2T tokens within 6,000 production H100-hours, including QAT, on 32 GPUs. Research uses one GPU and a separate budget. This framework pins the winning architecture, provides physics-based targets and an experiment/validation loop, and does not adopt the later HPO/kernel plans. Start there for speed work; the architecture program below remains historical.

This is a **separate, private** Terminus architecture-research repository for a Kimi K3 autonomous loop. It adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to the maintained Terminus ModelTunnelv2 code. The agent researches and edits architecture candidates; a protected runner holds the data, tokenizer, training exposure, evaluation, and spending contract fixed.

**First campaign: `architecture-20261006c` — COMPLETE (2026-10-06 → 2026-10-07).** The stack architecture (Peri-LN branch output norms + ResFormer value residual) beat the baseline by −1.15% at a matched 2.5B-token dense-FP8 horizon and by −0.55% after ternary-weight conversion + 500M-token QAT, while bounding the residual stream the baseline lets migrate. Full evidence: [notes/FINAL_REPORT_ARCHITECTURE_20261006.md](notes/FINAL_REPORT_ARCHITECTURE_20261006.md) and [notes/qat500-probe-comparison.md](notes/qat500-probe-comparison.md). Winner's branch: [`trial/stack/004`](https://github.com/varun29-git/terminus-autoresearch/tree/trial/stack/004).

The source snapshot and inspected autoresearch revision are recorded in [SOURCE_BASELINE.md](SOURCE_BASELINE.md). The rationale and readiness gaps are in [docs/AUTORESEARCH_DESIGN.md](docs/AUTORESEARCH_DESIGN.md); exact commands are in [docs/AUTORESEARCH_EXECUTION.md](docs/AUTORESEARCH_EXECUTION.md). Kimi's operating instructions are [program.md](program.md).

The three intended campaigns are **architecture discovery**, then joint architecture confirmation with hyperparameter/μP transfer and longer dense-to-ternary QAT, then kernel optimization for the chosen computation. The [stage-two hyperparameter autoresearch plan](docs/HYPERPARAMETER_AUTORESEARCH_PLAN.md) is a research and implementation proposal; its campaign is not implemented or launched. The first campaign used English as a cheap architecture proxy and reported quality, learning efficiency, stability, throughput and memory separately. Its selected 200M incumbent and short QAT probe do not establish the final 1.5B multilingual ternary recipe.

The editable architecture is [candidate.json](candidate.json) and the permitted model files. The [policy](research/policy.json), [preparation wrapper](prepare.py), [trial runner](train.py), and [scorer](research/protocol.py) are protected during a campaign. The runner uses the existing native DeepGEMM/FA3/Muon training stack and requires an H100-class SM90 GPU for paid trials. The [Azure cycle](docs/AZURE_EXPERIMENT_CYCLE.md) starts the existing H100 for one committed experiment and deallocates it only after Blob archival and independent hash verification.

To inspect the interface without launching a run:

```bash
python prepare.py --help
python train.py --help
python -m research.loop --help
python -m unittest tests.test_research_protocol -v
```

The prepared English-only token manifest is an external input; English development tasks are optional. The example campaign and task manifests under `research/` are templates, not ready data. The [scorecard](docs/ARCHITECTURE_SCORECARD.md) defines how the loop compares candidates. The existing `eval/` records are deliberately not imported into Git. The previous source repo README remains at [docs/terminus/ORIGINAL_REPO_README.md](docs/terminus/ORIGINAL_REPO_README.md).
