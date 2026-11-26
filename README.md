# Terminus autoresearch

Public research code and campaign records for the 25-layer Terminus Stack 4 architecture. This release consolidates the completed architecture, dense-speed, and 200M hyperparameter-search campaigns. It is a research snapshot, not a trained model release or a production launch handoff.

## Evidence

- [Results and evidence boundaries](docs/RESULTS.md) separates the architecture comparison, standalone E3, combined E3/E4/E9 kernel measurements, HPO, rejected experiments, and older external evidence.
- [Reproducibility guide](docs/REPRODUCIBILITY.md) maps retained source snapshots to the reports, lists external inputs and missing receipts, and gives the CPU verification commands.
- [Architecture final report](notes/FINAL_REPORT_ARCHITECTURE_20261006.md): Peri-LN output norms plus learned value residual reduced English held-out loss by 1.15% at a matched 2.5B-token, 200M-scale horizon. A separate 500M-token QAT probe retained a 0.55% advantage.
- [Dense speed record](docs/speed/trials/e2-e3-e4-e9-dense-round.md) and [paired-run summary](docs/speed/TRANSFER_32GPU.md): E3 improved complete-update throughput by 10.0%; the combined E3/E4/E9 stack averaged 40,996 tokens/s versus 36,723 tokens/s for the paired incumbent on one H100 NVL.
- [HPO implementation and final candidate](hpo/README.md): the 200M English proxy search selected spectral-Muon LR 0.017, tied-table AdamW LR 0.006, matrix decay 0.02, and zero embedding decay, with endpoint CE 3.0456. This is not a validated 1.5B recipe.

Later numerical and kernel experiments remain in source history for review. They are not promoted as validated speed or quality winners without the matched evidence described in the results guide.

## Release scope

The selected research architecture has 25 layers, width 2304 at the 1.5B speed-test shape, a tied 65,536-token vocabulary, 4096-token context, Peri-LN, learned value residual, and hybrid local/global attention. The architecture-quality measurements were made at 200M scale; the speed measurements used the 1.5B shape. Neither result establishes long-horizon quality, cross-domain transfer, 32-GPU scaling, or production feasibility.

The broader `production/ci-1` integration is **experimental and excluded from this release**. Its consolidated CPU evaluation did not pass: 22 tests failed, including packed-attention gradient mismatches and unmet production dependencies. No production configuration, manuscript, or paid-run handoff from that branch is included here.

## Repository map

- `terminus/`: model, training, optimizer, tokenizer, and kernel implementations.
- `research/`, `speed_research/`: architecture and speed campaign controllers and protocols.
- `hpo/`: completed 200M HPO runner, fixed reference, final candidate, and width-transfer generators.
- `notes/`, `docs/speed/`: campaign proposals, reviews, and measured summaries.
- `tests/`: CPU correctness and protocol tests; CUDA skips are not native-GPU validation.

`program.md`, `speed_program.md`, and `hpo/program.md` are retained historical protocols for completed campaigns. They do not authorize new cloud or GPU work. Raw corpora, token shards, checkpoints, generated trial records, credentials, and private infrastructure configuration remain outside Git.

## Verify on CPU

```bash
python -m pip install -e '.[muon]'
python -m compileall -q terminus tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -t . -v
python -m terminus.preflight --config configs/1.5b_candidate.json --count-only
bash -n scripts/train.sh
```

The count-only preflight checks configuration and parameter accounting; it does not train a model or certify CUDA kernels.

## Provenance and license

Original project code, documentation, and research reports are Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE), [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md), and [SOURCE_BASELINE.md](SOURCE_BASELINE.md). Third-party tokenizer assets and external dependencies retain their own terms. Paper PDFs are not distributed; [RESEARCH_LIBRARY.md](RESEARCH_LIBRARY.md) links canonical sources instead.
