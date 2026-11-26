# Working rules for the public research release

This is the public Terminus Stack 4 research-release tree. Start with `README.md`, `docs/RESULTS.md`, and `docs/REPRODUCIBILITY.md`. The architecture, dense-speed, and 200M HPO campaigns are complete historical records; their `program.md` files do not authorize new work. The broader `production/ci-1` handoff is experimental and is not part of this release.

## Development and verification

- Preserve model and optimizer semantics unless a change is explicitly reviewed. Keep dense-weight FP8 and ternary QAT distinct, and keep legacy campaign configurations on their recorded optimizer rules.
- Run relevant tests for each change. Before release, run the complete CI-equivalent CPU suite in `.github/workflows/ci.yml`: compile `terminus` and `tests`, run `python -m unittest discover -s tests -t . -v`, run the count-only preflight, and syntax-check `scripts/train.sh`.
- Do not weaken tests or treat CUDA skips as native-GPU validation. Record the hardware, commands, test counts, and unrun paths exactly. Do not run cloud or paid GPU campaigns without an explicit human execution decision.

## Data, artifacts, and disclosure

- Never commit credentials, private endpoints or account identifiers, personal local paths, full corpora, token shards, fetched evaluation records, checkpoints, generated run outputs, or cache files.
- Keep external manifests, raw result receipts, and infrastructure configuration outside Git. Distinguish checked-in summaries from independently available raw evidence; source alone does not reproduce a campaign whose inputs are external.
- Do not bundle third-party paper PDFs. Link canonical sources and preserve applicable license and attribution notices. The Apache-2.0 grant does not relicense tokenizer assets, datasets, checkpoints, papers, or installed dependencies.

## Research claims and history

- Keep architecture quality, standalone E3, combined-kernel throughput, HPO, and older external evidence separate. Do not convert a 200M English proxy result into a 1.5B or multilingual quality claim.
- Do not claim matched dense continuation, downstream quality, long-horizon QAT, width transfer, 32-GPU throughput, token-budget feasibility, or production readiness without the corresponding measurements.
- Keep changes reviewable and preserve existing history. Commit locally when requested; push or change repository visibility only when explicitly requested. Document blockers and unrun native-GPU checks honestly.
