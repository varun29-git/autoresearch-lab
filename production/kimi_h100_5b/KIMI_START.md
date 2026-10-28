# Paste this into Kimi

You are working in the CI-1 standalone production repository. Complete the
engineering and execution preparation for the bounded single-H100 proxy
campaign in `production/kimi_h100_5b/`. Begin with `AGENTS.md`,
`production/README.md`, and this folder's README, RUNBOOK, MONITORING and
EVIDENCE documents. The scientific rationale is in
`production/amd/pretraining_readiness/LR_BATCH_MONITORING.md`.

The user has up to 50 allocated hours on one Azure H100 and reports 200,000
proxy tokens/s. Verify that rate; do not treat it as a new measurement or use
it for the full model. The Azure SKU, usable HBM, actual billing unit, data
paths and credentials remain environment inputs. Gather those from the
available machine/configuration or request only the missing information.
Never print private keys or credentials in logs or commit them.

The user explicitly wants 5B tokens per proxy and width transfer, not a
full-size hyperparameter sweep. Implement exactly five initial proxy trials
H, R0, R1, R2, R3 at the same 2,097,152 global tokens/update. Use the reviewed
24-layer geometry narrowed to width 768, FFN 1920, Q/KV heads 6/2. Do not
copy the archived 25-layer HPO geometry. Retain context 4096, local window
512, global every sixth layer, tied 65,536 vocabulary, head dimension 128,
Peri-LN, learned value residual and the selected FP8 arithmetic.

Use the exact LR matrix and token-aligned complete WSD in RUNBOOK. Transfer
embedding/head AdamW once by 768/2304; keep spectral Muon base eta and
small-group rates unchanged. Historical proxy AdamW .006 maps to production
.002; proxy .009 is the retained production .003 hypothesis, not the
historical winner. Do not add another batch/GPU-count LR scaling rule, change
to RMS-matched Muon, alter decay, or tune every production-size candidate.

The deliverable has two boundaries:

1. **Complete now without a paid launch:** inspect the code, implement a
   separate narrowly allowlisted qualification config builder/validator and
   runner, add the missing observations and fail-closed guards, write CPU
   regression tests, prepare data/runtime/evaluation contracts from actual
   available inputs, and produce the exact launch packet outside the repo.
   If an external input is unavailable, finish independent work and identify
   precisely what remains. Make small reviewable commits, run relevant
   checks, and push completed milestones. Do not change baseline locks or
   mark unrun native tests passed.
2. **When the human launches the bounded campaign:** follow the ordered
   native gates, execute/observe only its authorized scope, maintain the
   shared allocated-hour ledger, compare complete proxy trials, and prepare
   one transferred-winner confirmation. Respect subsequent explicit human
   launch/stop instructions. This prompt itself does not grant permission to
   provision a VM or start a paid sweep; the repository reserves that to the
   human. Prepare concrete commands and receipts before the launch handoff.

The lower-level `scripts/train.sh` is not a ready qualification launcher.
`production/train.sh` rejects the proxy geometry, while
`production/inspect_run.py` also locks baseline optimizer values. Keep those
protections. Reuse the actual `terminus.train` implementation through a
separate reviewed qualification path, with the gates in RUNBOOK. The AMD
launcher requires AMD hardware and is not the H100 launcher.

The current telemetry samples a small subset of values; it does not provide
the full-state, required-gradient, native-FP8 and durable-recovery assurances
this campaign requires. Implement and test those gaps, with explicit coverage
and measured overhead. Observational hooks must not draw from training RNG,
change weights/moments, mutate cached FP8 operands, or perturb the data cursor.
Use isolated state-preserving reference branches. Detect failures; do not
silently lower an LR, reset optimizer state, skip a batch or change a schedule.

All runtime artifacts belong outside Git: manifests, configs, tokenizer/data
identity records, checkpoints, evaluation records, budgets and reports.
Existing HPO evidence is useful prior information, not a substitute for these
new-batch, new-horizon comparisons. Old `hpo/KIMI_START.md`, HPO/speed campaign
permissions, eight-hour caps and research programs do not govern this work.

Enforce the 50-hour allocation across setup, probes, evaluation, checkpointing,
retries and idle time. Reserve the approximately 1B full-size confirmation
before any optional diagnostic. At 200k proxy and 40k full-size tokens/s,
five proxies plus confirmation plus eight hours of reserve total 49.71 h.
If measured costs do not fit, report pending work; do not silently shorten
healthy 5B runs, discard H, expand the cap or claim a winner from incomplete
comparisons. The optional 1M-batch proxy is deferred by default.

Use matched initialization and data within the proxy matrix. Compare fixed
held-out NLL by domain at warmup exit, pre-decay and final cooldown, with
predeclared margins and a reserved final holdout. A tiny single-seed advantage
is not a proven winner. Report a tie when appropriate. A selected candidate
is a provisional transfer choice, not a trillion-token optimum.

After credible proxy selection, transfer hyperparameters, not weights, to
one fresh 1.446B model at the same 2M global batch. The default confirmation
is 1,006,632,960 tokens with its own complete WSD. Keep the user's 5B horizon
for proxies. A cooled-down qualification checkpoint is not a production
continuation or initialization. Do not begin month-long dense training,
ternary conversion or an eight-GPU AMD campaign under this handoff.

Return a clear handback using EVIDENCE.md: commits/tests, exact external
artifact paths and identities, measured versus assumed rates, completed and
pending gates, per-candidate losses and stability, consumed/remaining hours,
selection or tie, the single application of transfer, and what still requires
native MI325X and QAT qualification. Never certify unperformed work.
