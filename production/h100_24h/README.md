# 24-hour H100 qualification handoff

This is the new bounded dense LR study for one human-launched H100 allocation.
Read [RUNBOOK.md](RUNBOOK.md) for the exact science, token budgets, forecast,
native gates, controller command and stop rules. The original 5B/50-hour
campaign remains unchanged in `production/kimi_h100_5b/`.

The five proxy arms run **1,998,585,856 tokens each** at a **2,097,152-token
global batch**. H resumes from its verified 725-update parent checkpoint by
changing only the future WSD horizon. One fresh full-size confirmation runs
**599,785,472 tokens**. Estimated next-allocation use is **22.18 of 24 hours**
including 3 hours for setup and contingency, conditional on a native fork and
measured rates. A 24-hour budget is not a claim of quality or AMD readiness.

The useful entry points are `prepare24.py` (CPU-only external packet),
`fork.py` (prospective checkpoint transition), `auto.py` (automatic arm
handoff), `runner.py` (gated complete updates and live budget), `selection.py`
(frozen comparison/transfer), and `budget.py` (append-only allocation ledger).
The controller never starts or stops a paid VM; the human does that.

CPU checks:

```bash
python -m unittest production.h100_24h.test_plan
CI1_H100_PARENT_PACKET=/external/ci1-h100-qualification \
  python -m pytest tests/test_h100_24h.py -q
```

Native H100 checks are pending: restore payload/contract, new manifest shard
and tokenizer verification, current runtime and FP8 preflight, full-size fit,
S3 restore/upload, fork receipt, sustained wall rates and learning outcomes.
