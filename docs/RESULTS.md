# Results and evidence boundaries

This guide separates campaign results that are often easy to conflate. Numbers below come from checked-in reports and commit records; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) identifies the corresponding source snapshots and external raw artifacts. A checked-in summary is not a substitute for a missing raw receipt.

## Summary

| Campaign | Scale and hardware | Primary result | What it does not establish |
| --- | --- | --- | --- |
| Architecture, `architecture-20261006c` | About 200M parameters; one H100 NVL; English proxy | Peri-LN plus learned value residual: 2.65363 versus 2.68445 held-out loss at 2.5B tokens, a 1.15% relative reduction | 1.5B quality, multilingual quality, downstream tasks, or long-horizon production QAT |
| Standalone E3 | 1.5B shape; one H100 NVL; dense full updates | 40,187 versus 36,527 valid tokens/s in the 20-update confirmation, +10.0% | Quality at a longer horizon, 32-GPU scaling, or the effect of later kernels |
| Combined E3/E4/E9 | 1.5B shape; one H100 NVL; dense full updates | Three paired 20-update means: 40,996 versus 36,723 valid tokens/s, +11.6%; 55.5 GB peak torch allocation | An isolated E4 gain, production-cluster throughput, or validation of later numerical branches |
| 200M HPO | 200M parameters; English proxy; 500,170,752 tokens per ranked candidate | Final selection commit records endpoint CE 3.0456 for spectral Muon 0.017, tied-table AdamW 0.006, matrix decay 0.02, embedding decay 0.0 | Width transfer, a tuned 1.5B recipe, or improved quality relative to the architecture baseline |

## 1. Actual architecture result

The architecture campaign compared mechanisms at matched token exposure using dense FP8, the same English stream, WSD schedule, tokenizer, optimizer, and evaluation contract. The selected stack adds Peri-LN branch output norms and a learned layer-0 value residual. It does **not** include E3, E4, E9, spectral HPO rates, or the later 24-layer production experiment.

### Screen and full dense comparison

| Arm | 0.5B-token screen loss | 2.5B-token final loss | Full-run median throughput | Full-run peak memory |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3.13222 | 2.68445 | 152.4K tokens/s | 76.0 GiB |
| Peri-LN only | 3.13205 | Not continued | — | — |
| Value residual only | 3.09097 | Not continued | — | — |
| Peri-LN + value residual | **3.08531** | **2.65363** | 143.5K tokens/s | 86.9 GiB |

At the full horizon, the stack reduced held-out loss by 1.15% and led at all 26 matched checkpoints. The cost was 5.8% lower throughput and about 11 GiB more peak memory. Baseline layer-24 residual RMS ended at 177.9 with a peak of 293.6; the stack ended at 12.1 with a peak of 15.5. These stability observations are specific to this campaign and probes.

### Separate QAT probe

After dense training, each arm converted to ternary weights with FP8 activations and continued for 500M QAT tokens. Final losses were **2.81095** for the stack and **2.82651** for the baseline, a 0.55% relative advantage. The stack had the larger immediate conversion shock, 9.15 versus 7.78, before recovery. This short probe supports post-recovery stability at 200M scale; it is not a final QAT schedule or a production-scale ternary result.

Sources: [final architecture report](../notes/FINAL_REPORT_ARCHITECTURE_20261006.md), [full-run review](../notes/review-trial-004-full.md), and [QAT comparison](../notes/qat500-probe-comparison.md).

## 2. Standalone E3 speed result

E3 adds packed Q/K normalization and RoPE with learned-gain gradients. The measurement used the 25-layer, 1,500,456,496-parameter Stack 4 shape, dense training, the real staged English stream, and complete 524,288-token optimizer updates. Complete updates include data movement, accumulation, forward, loss, backward, clipping, optimizer work, and FP8 cache invalidation.

| Arm | Confirmation | Valid tokens/s | Peak torch allocation |
| --- | --- | ---: | ---: |
| Incumbent, microbatch 4 / accumulation 32 | 20 updates | 36,527 | 56.8 GB |
| E3 | 20 updates | **40,187** | 54.3 GB |

The confirmed improvement is **10.0%**. A shorter six-update E3 screen reported 41,441 tokens/s, but it is not the confirmation number. Native checks found exact forward and input-gradient agreement and learned-gain gradients within the recorded tolerances.

Source: [dense round record](speed/trials/e2-e3-e4-e9-dense-round.md).

## 3. Combined E3/E4/E9 kernel result

The combined stack adds E4 fused value mixing and E9 batched Muon to E3. It preserves the architecture campaign's optimizer convention; the later spectral-Muon HPO rule is a separate mathematical change.

Three complete paired 20-update receipts are summarized as follows:

| Pair | Combined tokens/s | Incumbent tokens/s |
| --- | ---: | ---: |
| 1 | 40,964 | 36,851 |
| 2 | 40,967 | 36,639 |
| 3 | 41,058 | 36,680 |
| **Mean** | **40,996** | **36,723** |

The relative mean improvement is **11.6%**, with **55.5 GB** peak torch allocation for the combined stack. E9 was approximately a wash in the one-GPU eight-update screen and was retained for parity and possible low-local-accumulation benefit. E4 was kept in the combined stack, but these receipts do not provide an isolated sustained E4 speedup. Native numerical checks are recorded in the trial note.

An archived 80-update log line reports **41,075 tokens/s**, 12.76 seconds/update, and 55.5 GB. Its full JSON receipt was not copied into the available evidence, so the three paired receipts above remain the auditable baseline. The earlier eight-update combined rate of 41,936 tokens/s was optimistic relative to that soak line.

Later N1/N2/E16 and QuACK implementations are optional research code, not additional validated wins in this summary. A matched-token quality gate and complete sustained receipts are required before promoting a numerical branch.

Sources: [dense round record](speed/trials/e2-e3-e4-e9-dense-round.md), [validation state](speed/VALIDATION.md), and [paired-run summary](speed/TRANSFER_32GPU.md).

## 4. HPO result

The HPO campaign used the E3/E4/E9 implementation at 200M scale with spectral Muon fixed across candidates. Each ranked candidate ran 1,272 updates, exactly **500,170,752 tokens**, against the locked English stream; only the endpoint ranked the candidate.

The final selection commit, `5e5f4ab`, records 12 ranked trials and selects:

| Field | Selected value |
| --- | ---: |
| `muon_lr` | 0.017 |
| `adamw_lr` | 0.006 |
| `norm_lr` | 0.0003 |
| `qk_norm_lr` | 0.0001 |
| `weight_decay` | 0.02 |
| `embedding_weight_decay` | 0.0 |
| Endpoint English CE | **3.0456** |

The same commit records a local matrix-LR bracket: 0.012 was about +0.008 CE and 0.024 about +0.003 CE relative to the winner. Matrix decay 0.05 was about +0.021 CE relative to 0.02. Tied-table AdamW 0.003–0.006 was comparatively flat, while 0.0003 was far worse. These are adaptive-search observations, not independent repeated-seed confidence intervals.

The raw result ledger and checkpoints are external, and a complete 12-row results table is not checked in. The final candidate and selection rationale are preserved, but the campaign cannot be independently re-ranked from Git alone. No 700M/1.5B width-transfer trial was completed.

Sources: [final candidate](../hpo/candidate.json), [fixed reference](../hpo/reference_200m.json), [optimizer research](../notes/optimizer-research-20261010.md), and [spectral-Muon implementation note](MUON_SPECTRAL_IMPLEMENTATION.md).

## 5. Negative and external evidence

### Rejected or unresolved architecture/speed work

- Peri-LN alone tied baseline screen loss; its primary independent effect was residual-scale control. Value residual alone improved loss but did not bound residual RMS.
- Gated attention was implemented but never screened because its target layer-0 pathology did not reproduce. It is not a validated improvement.
- E2 residual-norm fusion passed numerical equivalence but was rejected because gain-reduction reads erased the expected speed gain.
- E9 was a one-GPU speed wash in its short screen. Split-K weight gradients, the tested quantizer change, CUDA graphs, and loss-chunk alternatives regressed or washed out in the recorded probes.
- Mid-layer attention entropy concentration remained unresolved and probe-limited. Cross-domain and long-context retrieval behavior were not tested by the architecture screen.

### Older dense/QAT evidence is not a control for Stack 4

The earlier 150M Lean 10B-token dense run and 2B-token QAT continuation used a different architecture and training history. Under the original dense validation weighting, QAT loss was **2.6042 versus 2.4194**, a +0.1848 gap; under the QAT mixture weighting it was **2.4889 versus 2.3154**, +0.1735. Every tracked source worsened despite additional training. A same-data, same-LR dense continuation control was not run, so the gap cannot be assigned solely to ternarization. This is negative external context, not a measurement of the selected Stack 4 QAT probe.

Source: [combined dense/QAT review](../experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md).

### Production and scale limits

The archived 32-GPU document is an analytical scenario, not a cluster measurement. Even assuming 41,075 tokens/s on every GPU with zero overhead, 6,000 GPU-hours yields about **887B dense tokens**, below the stated 1.5T minimum before QAT, communication, evaluation, checkpoints, or downtime. No 32-GPU result supports production feasibility.

The separate `production/ci-1` integration was evaluated on the consolidated tree before this release. Compile and shell syntax checks completed, but the full CPU suite reported **22 failed, 604 passed, 63 skipped, and 215 subtests passed**. Two packed-document attention cases exceeded the existing gradient tolerance; other failures reported missing `orjson`, `duckdb`, and `_terminus_tokenizer_native`. The production branch remains experimental and none of its production handoff is merged into this release.

## 6. Claims this release does not make

There is no checked-in production checkpoint, final multilingual or downstream evaluation, matched long-horizon dense continuation, production-scale dense-to-QAT result, repeated-seed architecture study, measured 32-GPU throughput, validated 1.5B HPO transfer, or proof that the target token budget is feasible. CPU correctness tests do not close any of these gaps.
