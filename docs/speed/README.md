# Stack-004 speed autoresearch, five-hour continuation

This dedicated worktree starts from the measured E3/E4/E9 dense stack on commit `1e8c0a4`. The candidate model remains 25 layers and 1,500,456,496 parameters. [speed_program.md](../../speed_program.md) is Kimi's operating program; [PENDING_5H.md](PENDING_5H.md) inventories every unfinished experiment and the largest measured costs. [EXPERIMENTS.md](EXPERIMENTS.md) supplies the full hypothesis atlas, and [PHYSICS.md](PHYSICS.md) derives the H100 and production budget bounds. [KIMI_START.md](../../KIMI_START.md) is the copyable paid-run prompt.

The research budget is **five cumulative H100 VM-on hours** on one H100, separate from the 6,000 aggregate production H100-hours. This pass measures dense training only. The controller in `research/speed_session.py` now caps the new campaign at 18,000 cumulative seconds and installs a VM-side cutoff before the cap. Use the **new** external control root `/Users/varundaiya/terminus-speed-external/stack-004-5h/control`; the old three-hour ledger belongs to a completed campaign. The candidate trials may use the same persistent remote data and results mount with unique `v2-...` names.

The previous best sustained log reports 41,075 valid tokens/s at 55.5 GB allocated. Three complete paired 20-update receipts show a stable ~11.6% gain over the incumbent. The 80-update summary is archived, but its full JSON receipt was not copied locally. The first new winning run should save the complete JSON and its hash immediately. It does not need a long repeat of the old baseline before useful candidate work.

A local config exists outside Git at `/Users/varundaiya/terminus-speed-external/stack-004-5h/control/speed_session.json`. If moving this worktree to another host, start from [the example](../../research/speed_session.example.json) and choose a fresh external `local_control_root`. Commands below are shown for Kimi **after the human's one-time paid-campaign authorization**; reading or local testing does not authorize `open`.

```bash
python3 -m research.speed_session status --config /Users/varundaiya/terminus-speed-external/stack-004-5h/control/speed_session.json
python3 -m research.speed_session open --config /Users/varundaiya/terminus-speed-external/stack-004-5h/control/speed_session.json
python3 -m research.speed_session stage --config /Users/varundaiya/terminus-speed-external/stack-004-5h/control/speed_session.json
python3 -m research.speed_session exec --config /Users/varundaiya/terminus-speed-external/stack-004-5h/control/speed_session.json -- nvidia-smi -L
python3 -m research.speed_session close --config /Users/varundaiya/terminus-speed-external/stack-004-5h/control/speed_session.json --reason budget
```

`open` is paid. `stage` sends a clean committed candidate. `exec` runs the real complete-update timing script on the H100. Continue experiments under the same session instead of rebooting for each trial. If a development gap warrants deallocation, use `close --reason idle`, keep the ledger and resume later. `goal` requires proof of the full production objective, not a dense-only rate. The earlier one-shot `research.azure_speed_worker` is not this loop.

Local checks, without VM allocation:

```bash
python3 -m unittest tests.test_speed_session tests.test_s0_block tests.test_speed_s0 tests.test_speed_research tests.test_hopper_candidates
```

`tests.test_logical_muon` additionally needs the `muon-optimizer` package, which is absent from this local Python; the native H100 environment has its own runtime. No native H100 trial or VM lifecycle command has been run from this new five-hour worktree. The previous campaign's results remain in its separate archived external control tree.
