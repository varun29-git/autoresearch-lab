# Terminus architecture autoresearch: design and evidence

**Status, 5 October 2026:** framework implemented and CPU-validated; no H100 campaign launched, no architecture optimum claimed. The campaign has a data and development-task readiness gate. The numeric policy is a reviewable first campaign contract, not a recommendation to freeze these choices forever.

## What was copied from autoresearch

At inspected [Karpathy/autoresearch commit `228791f`](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117), `prepare.py` owns fixed data and evaluation, `train.py` is the agent's editable model/trainer, and `program.md` is the human-written agent instruction. Each trial trains for five minutes of training time, reports validation bits per byte, and is kept or discarded. That is an elegant way to make rapid, traceable autonomous experiments. Its own README explicitly says the fixed-time objective finds the best model **for that platform and time budget**, and is not comparable across platforms. The upstream `program.md` tells the agent to commit every trial, log outcomes, and keep going without asking after each trial.

We copied that **research method**, not its Python source. The upstream README calls the code MIT, but the inspected tree contains no `LICENSE` file and GitHub's license field is null. There was no reason to copy its tiny GPT trainer into an existing Terminus implementation. The source baseline and revisions are recorded in [SOURCE_BASELINE.md](../SOURCE_BASELINE.md).

The Terminus adaptation deliberately changes four things:

| Upstream autoresearch | Terminus stage one | Reason |
| --- | --- | --- |
| Five-minute wall-time training | Equal 2.5B-token full horizon; 0.5B interim screen; wall time is only a cap | The owner wants architecture quality first. Equal time would reward a faster candidate for seeing more data. |
| One `val_bpb` on ClimbMix | Ten held-out domains, answer-only tasks, position buckets, paired batch interval | English-only next-token loss cannot define the desired multilingual/code/math model. |
| One editable `train.py` | Editable model files and `candidate.json`; protected trainer, data and evaluator | Terminus's architecture spans several modules, but its training/evaluation contract must not drift between trials. |
| Unbounded local loop | One-GPU 12-hour campaign cap, append-only receipts, no automatic VM creation | The search must respect the owner's cloud budget and preserve failed evidence. |

## Objective and controls

The baseline is the maintained 200M width-family dense-FP8 configuration. A candidate may change architecture and model code within a **190M–210M parameter envelope**. This prevents a trivial quality gain from simply adding many parameters. Width and total parameter count are researched separately in the later transfer stage. The exact parameter envelope can be changed **before** a campaign begins, but not mid-campaign.

The baseline is first trained only to the 0.5B-token screen, so candidate screens can be compared without paying for a 2.5B baseline upfront. Its remaining 2B tokens are spent only when at least one candidate merits a full-horizon comparison. The 2.5B horizon is a proposed extension point, not a proven minimum training duration; it must be reviewed against measured learning curves and the overnight budget before locking a campaign.

The primary score is the predeclared weighted relative improvement in held-out next-token loss at matched exposure:

\[
Q=\sum_{g\in\{\mathrm{en,indic,code,math}\}}w_g
\frac{L_{0,g}-L_{c,g}}{L_{0,g}},\qquad
(w_{en},w_{indic},w_{code},w_{math})=(.35,.35,.15,.15).
\]

Here `indic` is the equal-language mean over Hindi, Marathi, Kannada, Telugu, Tamil, Gujarati and Bengali. A positive value favors the candidate. These proposed weights deliberately give Indic equal aggregate importance to English; they are **not** copied from the 50/30/10/10 training mixture. The owner should review the weights before starting. Every language and domain is reported separately. A >2% relative loss regression in one domain is flagged, not hidden by the aggregate. Differences below 0.5% aggregate are marked uncertain. The runner bootstraps *paired fixed validation batches* for a screening interval; correlated documents and repeated search can still make that interval optimistic. A close decision needs more tokens, a second seed, and later scale transfer.

The development tasks are answer-only conditional losses over fixed prompts and references, with up to 100 items per task at full-stage checkpoints. Task accuracy and executable code tests should be added when task definitions and proxy capability support them. The current task evaluator's teacher-forced loss is useful but has **not** been calibrated against actual downstream accuracy for this model. It must remain a second signal, not an unexamined replacement for quality. Public ScalingBench contains useful English development examples, but its generated rationales cannot be treated as ground-truth short answers. The MILU dataset in the existing evaluation manifest is gated; the actual seven-language development suite is not in this repo. The task manifest gate prevents a false English-only “winner.” Keep a separate untouched final test outside the autonomous loop.

The context-position diagnostics score targets in 0–511, 512–2047 and 2048–4095 token positions on fixed blocks. They reveal position effects; they do **not** prove long-range retrieval. A meaningful local/global architecture conclusion also needs source-disjoint long-context tasks or distance-controlled prediction examples. These should be supplied in the development task manifest, with final counterparts held out.

The fixed training recipe is Muon plus auxiliary AdamW, dense FP8/DeepGEMM/FA3, batch `32 × 3 × 4096 = 393,216` tokens/update on one GPU, and a 10% warmup / 70% stable / 20% decay WSD schedule. This is an experimental control. It is not claimed optimal: the completed 150M run's large norm LR and first-attention suppression show exactly why optimizer interactions must be checked later. A candidate that loses under this one recipe can still be promising, so the loop retains near ties and mechanism-based exceptions. In stage two, the shortlist is revisited with LR/batch/μP transfer and longer dense-to-ternary QAT. A dense-only winner is not automatically the best ternary parent.

Terminus is intended for dense-FP8 pretraining followed by ternary-weight QAT **with FP8 activations throughout**. Most stage-one candidates need only dense-FP8 screens and matched full comparisons. The protected runner now offers a conditional `qat_probe` after a completed 2.5B-token dense parent: about 100M further tokens, fixed half-rate optimizer policy, a new phase-relative WSD warmup/decay, carried weights/moments/RNG/data cursor, zero-update conversion evaluation and a final short-QAT evaluation. Kimi must record a quantization-specific reason before invoking it, and a candidate probe requires a matching baseline probe. This checks whether ternary conversion could change the architecture ranking; it does not establish an optimal QAT schedule or final ternary quality. Its cost is charged to the same 12-hour cap. The prepared data must contain the continued stream past 2.5B tokens if a probe is selected.

## Why the overnight claim is conditional

The completed 146M-parameter dense run processed 10B tokens in **13.11 active H100 hours**, about 212K training tokens/s including validation and checkpoints. Seven billion tokens would take about 9.2 hours at *that* rate, before architecture-dependent slowdown and additional task evaluation. The current 200M learned-Q/K-norm baseline has not been timed on an H100, and its unfused Q/K path may differ. The policy therefore caps actual GPU-hours rather than promising a fixed number of experiments. Kimi may run fewer trials and must stop before the cap. A10G does not execute the present SM90 DeepGEMM/FA3 path, so its cheaper hourly price does not establish cheaper or transferable architecture evidence. No A10G-versus-H100 cost comparison has been measured.

The 200M proxy is closer to the target than a 5M paper pilot, but its tied 65K embedding remains a larger share of parameters than in the 1.5B target. Stage-one rankings need a larger-width bridge. The [combined completed-run review](../experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md) reports a 10B dense run and 2B QAT continuation with no nonfinite incident, first-layer suppression, and worse held-out loss after QAT. Those are motivation for measuring architecture and conversion jointly later, not proof that a particular cut or norm is optimal.

## Agent freedom and reproducibility

Kimi is **not** handed a fixed ablation list. It reads the existing architecture, telemetry, archived tests, MiniCPM4/BitCPM4, attention/norm/quantization research and public implementations, then proposes mechanisms, predicts failure signals, commits one testable design and measures it. It may port licensed state-of-the-art architecture code or invent a coherent redesign; large changes are allowed inside the editable model surface and parameter envelope. A pre-trial [proposal note](../notes/TEMPLATE.md) must specify equations, tensor shapes, scaling and gradients, FP8/QAT interactions, expected cost, a falsifiable prediction and the exact empirical sources or Terminus measurements motivating the change. It can revisit literature after failed trials. The fixed contract protects the comparison; it does not choose the architectural idea.

The exclusion of [previously tried forced-unit normalization](EXCLUDED_UNIT_NORMALIZATION.md) is an evidence-based search boundary, not a prescribed replacement architecture. The `unit_rms` Q/K configuration already has a long Terminus run; the 200M baseline explicitly disables legacy SubLN and uses learned Q/K RMSNorm. The loop should search for new mechanisms and measure quality rather than rerun the old fixed-unit safety stack. A unit-norm step inside standard learned-gain RMSNorm is not what this exclusion means.

The campaign lock stores the source commit, policy hash, tokenizer hash, data-manifest hash and per-task file hashes. The editable surface is `candidate.json`, `terminus/model.py`, new `terminus/architecture_*.py` modules, associated architecture tests and notes. Existing quantization, optimizer, data, training, evaluation and low-level kernel files are protected. The runner rejects uncommitted candidates, changes to protected paths, out-of-envelope parameter counts, missing domains, task-file changes, and a trial that exceeds the budget. Each dense trial has a native numerical preflight, immutable checkpoint receipts, train/stability/incident logs, held-out evaluation, optional full-stage task records, and an append-only ledger. Optional QAT probes use the existing protected transition implementation and record the conversion receipt and matched evaluation. Failed branches remain reviewable. A Kimi session is an agent interface to this repository; no Kimi credentials or local provider configuration are copied into Git.

## Readiness and later stages

Before the first paid trial, stage the prepared 65K-tokenizer corpus on the GPU machine, provide a ten-domain fixed-split manifest with adequate exposure, build and hash an answer-only development suite for all seven Indic languages plus English/code/math, verify contamination exclusions, install the pinned SM90 stack, and pass `prepare.py` plus the baseline native preflight. This repo has neither the full corpus nor that complete task suite. CPU tests do not certify H100 arithmetic.

Stage two should tune hyperparameters and Muon/μP transfer **on a small architecture shortlist**, check at 700M/1.5B, and compare substantially longer matched dense-to-ternary continuations before fixing the final architecture. Stage three profiles that selected computation and autonomously proposes kernels under numerical parity and whole-step throughput tests. Neither later loop is implemented or authorized to run by this repository's stage-one program; the optional short QAT probe belongs to stage one only.

### Primary sources inspected for this adaptation

- [Karpathy/autoresearch README, `program.md`, `prepare.py`, `train.py`](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117): actual fixed-time loop and metric, not a generic “agent framework.”
- [MiniCPM4/BitCPM4 technical report](../papers/minicpm4_2506.07900.pdf): ModelTunnel task-aware proxy search and dense-to-ternary QAT; its schedules are hypotheses for Terminus, not defaults for this architecture screen.
- [Tensor Programs V / μTransfer](../papers/tensor_programs_v_mup_2203.03466.pdf) and [Muon scaling](../papers/muon_scalable_2502.16982.pdf): why width-transfer and optimizer policy need a separate validation stage.
- [Terminus ModelTunnel quality strategy](terminus/TERMINUS_QUALITY_AND_HYPERPARAMETER_STRATEGY_20260927.md), [architecture evidence](terminus/architecture_review_20261005/RESEARCH_EVIDENCE.md), and [completed dense/QAT review](../experiments/terminus_lean_150m_qat_2b/analysis/COMBINED_DENSE_QAT_SCALING_REVIEW.md): local implementation and measured limitations.
