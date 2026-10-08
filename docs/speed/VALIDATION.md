# Local validation, 2026-10-09

The winning stack was previously instantiated on the meta device at the proposed target shape: 1,500,456,496 parameters, 25 layers, 50 Peri-LN branch norms and 48 value-residual scalars. That confirms shape/count, not H100 speed.

The instrumented trainer has a CPU fixture test that compares full model and optimizer states bitwise against the ordinary trainer after matching updates. After simplifying the timing path, `python -m unittest tests.test_s0_block tests.test_speed_s0 tests.test_speed_research -q` passed **38 tests**. These include a 60-second trial plan that starts without a ten-minute reserve or preflight, a quick block with no full checkpoint, and worker tests that only request deallocation after a verified archive. The ordinary production trainer still requires its preflight report.

No native H100 benchmark or complete Azure boot/archive/deallocation cycle was run here. GPU initialization, compilation and VM overhead remain unknown. The single-GPU timing result and 32-GPU scaling must be measured later on authorized hardware.
