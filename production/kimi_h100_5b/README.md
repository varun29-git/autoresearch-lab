# Kimi handoff: five-billion-token proxy qualification

Prepared 2026-10-28. **Start by pasting [KIMI_START.md](KIMI_START.md) into
Kimi, with this repository as its workspace.** This packages the current
[LR/batch protocol](../amd/pretraining_readiness/LR_BATCH_MONITORING.md).
It supersedes older experiment orders, not the reviewed production recipe.

The task is to finish the qualification tooling, then support a human-launched
campaign on **one H100, at most 50 allocated GPU-hours**. Search on the
**194,329,390-parameter proxy**; transfer the selected hyperparameters to one
fresh **1,446,478,126-parameter** confirmation run. Each of the five proxy
candidates receives **4,999,610,368 tokens**, the complete-update equivalent
of 5B. All initial comparisons use **2,097,152 global tokens/update**.

| Read/use | Purpose |
| --- | --- |
| [KIMI_START.md](KIMI_START.md) | Paste-ready agent instructions and scope |
| [RUNBOOK.md](RUNBOOK.md) | Frozen matrix, schedules, engineering backlog and execution order |
| [MONITORING.md](MONITORING.md) | Required observations, failure actions and selection rules |
| [EVIDENCE.md](EVIDENCE.md) | External artifact layout, receipts and handback requirements |
| [plan.py](plan.py) | Executable CPU-only arithmetic and budget forecast |

## What is executable now

From the repository root, using Python 3.10 or newer; no Torch or GPU needed:

```bash
python3 -m production.kimi_h100_5b.plan
python3 -m unittest production.kimi_h100_5b.test_plan
```

To save a planning artifact outside Git:

```bash
python3 -m production.kimi_h100_5b.plan \
  --proxy-tps 200000 --full-tps 40000 --reserve-hours 8 \
  --output "$HOME/ci1-h100-qualification/forecast.json"
```

The output refuses an existing file or a path inside this repository. A
forecast over the cap exits with status 2. This helper **does not generate
trainer configs, validate data, enforce a live budget, allocate a GPU or
launch training**. A forecast that fits is not a launch clearance. Its rates
are inputs: 200k proxy tokens/s is user-reported; 40k full-size tokens/s is an
assumption until measured at the specified geometry and cadence.

## Current readiness

| Component | Status at handoff |
| --- | --- |
| Trial design, transfer and complete WSD horizons | Specified here |
| Planning arithmetic and external forecast writer | Implemented, CPU-testable |
| Production trainer, H100 runtime capture and native preflight | Existing code; native checks must be run on the allocated H100 |
| Candidate-specific preparation/validation and bounded qualification runner | Kimi must implement and test before paid trials |
| Complete safety telemetry, native operand probes, live backup/replay and campaign budget ledger | Required additions; existing sampled monitoring is insufficient |
| Proxy/full-size learning, hardware fit and cadence-inclusive speed | Unmeasured for this campaign |
| Eight-MI325X execution, AMD FP8 parity, QAT and month-long schedule | Separate subsequent gates; not certified by this H100 campaign |

At the forecast rates, five proxies take **34.72 h**; the approximately 1B
full-size confirmation takes **6.99 h**; eight hours for setup/checks/contingency
bring the total to **49.71 h**. This is tight. The optional sixth, 1M-batch
proxy is **deferred**. No automatic larger sweep, longer confirmation or
50-hour overrun is part of this handoff.

The human controls paid launch and stop under [AGENTS.md](../../AGENTS.md).
Creating this folder does not authorize provisioning or training. Kimi should
finish all offline work and prepare the exact reviewable launch packet before
that handoff; it should not stop merely to ask whether to implement the plan.
