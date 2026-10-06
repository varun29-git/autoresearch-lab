# Trial diagnostic review: baseline / screen / cycle 20261006T095206Zf07ba1d9 — harness failure (no training)

## Outcome and evidence coverage

- Baseline commit `21d9bb6`, stage screen, cycle `20261006T095206Zf07ba1d9`, 0 tokens trained, 0.0202 GPU-hours, 0.55 VM-on hours. Archive verified in Blob (receipt sha256 `a40bf572…`), VM deallocated by the worker per protocol.
- Read: `train_screen.log`, `run/incidents.jsonl`, `preflight.json`, `campaign_lock.json`, `diagnostic_status_screen.json` (downloaded to controller analysis dir).
- Native preflight **passed**; data gate (`prepare.py`) **passed** — campaign lock written, 200,290,816 baseline parameters, English view accepted. The failure is neither data nor numerics.

## Failure

`build_optimizer` → `LogicalMuonWithAuxAdam.__init__` → pinned upstream `muon.py` line 247 `AssertionError` (exact group-key set). Incident record: `run_failure`, `cuda_oom: false`.

**Root cause (reproduced on the controller):** `terminus/optim.py` attached the telemetry label `name="qk_norm"` to the auxiliary Adam group *before* construction. The pinned Muon (`f98f1cac`) asserts each aux group's keys are exactly `{params, lr, betas, eps, weight_decay, use_muon}`. Every prior exercise of this code path — the archived Lean 150M runs (`qk_norm_mode="none"`) and the whole CPU test suite (`tiny_config` default `unit_rms` has no QK gain parameters) — never created a qk_norm parameter group. The campaign baseline is the first configuration with `learned_rms` QK gains, so the locked code could not run its own locked baseline.

**Why this is a new campaign version, not an in-campaign edit:** `terminus/optim.py` is protected during a campaign; the program routes harness changes through a new campaign version. The fix is behavior-preserving (identical groups, LRs, update math; the cosmetic `name` label is now attached post-construction, exactly where the code already sets `name`/`base_lr` for every other group) and adds the missing regression coverage (`test_learned_qk_norm_group_builds_and_steps`, plus a CPU dev-mode training-loop smoke with `learned_rms`+Muon, both green; full suite 221 passed).

## Judgment

| Finding | Classification | Evidence | Follow-up |
| --- | --- | --- | --- |
| Baseline config cannot build its optimizer under the pinned Muon | negative (harness bug, not architectural) | train_screen.log traceback; controller repro | fix + new tag `architecture-20261006b`; rerun baseline screen |
| Data gate + preflight + archive/verify/deallocate path | positive (worked as designed on first paid cycle) | campaign_lock.json, preflight.json, verified Blob receipt | none |
| Test-suite gap: no learned_rms+Muon construction coverage | negative (process) | tests passed while path broken | regression test added |

**Cost ledger note:** 0.55 VM-on h + 0.0202 GPU-h spent on this failed attempt. Earlier staging: ~0.75 VM-on h over two boots (first boot self-deallocated by a stale boot-enabled finalizer unit from the archived QAT run — `terminus-lean-qat-2b-finalize.service`; all four stale `terminus-lean-*` units were stopped/disabled/masked after confirming the old archives verified; Azure CLI 2.91.0 installed on the VM, which the campaign worker requires). Cumulative VM-on ≈ 1.3 h against the 12 h budget before any training token.
