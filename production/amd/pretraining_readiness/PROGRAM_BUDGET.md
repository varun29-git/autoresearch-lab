# Include posttraining in the fixed allocation

The CPU-only `production.amd.program_budget` command wraps the existing
fixed-QAT forecast with named **remaining** node-hour reserves for SFT,
preference training, qualification, recovery and finalization. It does not
implement those training objectives, automatically orchestrate phases or
authorize a paid launch.

Run `python -m production.amd.program_budget --help`. Supply an external JSON
policy with inputs to `program_forecast`; use the original `--allocation`
ledger for actual remaining time. Without a ledger, the default is a scenario
over 5,760 GPU-hours. Policy rates are **whole-node scored targets/second**
including normal cadence. Policy GPU count must match the ledger.

Only unspent commitments belong in `remaining_reserves_node_hours`. Elapsed
time, including idle time, has already been charged by the ledger. Fund the
remaining absolute QAT budget and finishing chain before expanding dense.
Release reserve early enough for a prospective plateau change; unused time
discovered after committing the ending cannot be retroactively added to dense.

`unique_mix_capacity()` computes a conservative proportional cap from integer
cell weights and eligible unused counts. A missing cell has zero capacity;
no cell borrows another cell's supply. The caller must establish actual
document/prompt uniqueness, quality and packed scored-target inventory.
`apportion()` generates exact final procurement quotas. It is **not** a runtime
sampler: differences between independently rounded largest-remainder budgets
are not guaranteed monotonic.

The current dated decision packet and actual source/data/reserve configuration
are outside Git:

[`allocation-and-curriculum/README.md`](</Users/varundaiya/terminus-pretraining-readiness/2026-11-06/allocation-and-curriculum/README.md>).

The user's subsequent clarification confirms that larger tokenized pools are
already available. The current
[existing-surplus execution steps](</Users/varundaiya/terminus-pretraining-readiness/2026-11-06/allocation-and-curriculum/EXISTING_SURPLUS_STEPS.md>)
supersede exact preservation of every original source-cell percentage as an
extension prerequisite. Blend eligible unused records with the remaining core
data and protect QAT/posttraining. The strict proportional cap applies only
when that particular mixture is selected; it is not proof that surplus tokens
are absent.

That packet distinguishes content budgets from optimizer targets and separates
data phases from LR/quantization semantics. It also records current source
drift, fresh-data shortfalls, posttraining objective gaps and unrun native
acceptance. Preserve the reviewed model/optimizer until any proposed
transition changes have their own accepted contract.
