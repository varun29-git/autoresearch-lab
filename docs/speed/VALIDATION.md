# Local validation, 2026-10-09

The winning stack was previously instantiated on the meta device at the proposed target shape: 1,500,456,496 parameters, 25 layers, 50 Peri-LN branch norms and 48 value-residual scalars. That confirms shape/count, not H100 speed.

The instrumented trainer has a CPU fixture test that compares full model and optimizer states bitwise against the ordinary trainer after matching updates. The speed tests cover a short no-checkpoint timing block, the older S0 worker, and the continuous session's cumulative three-hour ledger, restart accounting, timer installation command and startup-failure cleanup. The ordinary production trainer still requires its preflight report.

No native H100 benchmark or complete Azure boot/cutoff/deallocation cycle was run here. The VM-side cutoff depends on Azure managed identity and systemd and still needs an authorized live test. Azure shutdown latency means the controller cannot promise an exact billed 3.000-hour maximum; it starts the cutoff two minutes before its conservative estimate of that cap. GPU initialization, compilation and VM overhead remain unknown. The single-GPU timing result and 32-GPU scaling must be measured later on authorized hardware.
