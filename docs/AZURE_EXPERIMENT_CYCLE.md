# Azure H100 experiment cycle for Kimi

**Historical protocol.** At preparation on 5 October 2026, the designated H100 NVL VM was deallocated. Resource-group, VM, region, and account identifiers are removed from this public record. The completed architecture campaign used the workflow below; it is not authorization to start another VM. Do not start a VM to think, read papers, write code, or review results.

The controller used Azure CLI, SSH, and this repository. Each proposed architecture was researched, implemented, tested, and committed **before** H100 startup. `research.azure_cycle` sends an exact commit as a git bundle, so the VM needs no GitHub credential. The controller refuses an already-running VM, detects older GPU jobs, validates the persistent data/result mount and pinned H100 environment, and rechecks prepared shard hashes before each continuation or candidate trial. It then launches one systemd worker. The worker uses the protected `research.loop` runner and writes preflight, training, evaluation, stability, incident, checkpoint, and ledger records, with sampled `nvidia-smi dmon` telemetry.

After each experiment, including failed or timed-out experiments, the worker makes a SHA-256 inventory of the trial directory plus the campaign lock and ledger. It uploads to a **unique Blob prefix**, lists uploaded names, downloads every file into a second directory, and verifies every hash with `research.artifacts.verify`. **Only the verified path calls `az vm deallocate --no-wait`.** Failed verification leaves the VM and managed disk intact for repair. The controller then confirms the receipt and downloads diagnostics while the H100 is off. Checkpoints remain external; `fetch --include-checkpoints` retrieves them only when explicitly needed.

## One-time prerequisites

The historical setup used a dedicated external Blob container with least-privilege managed-identity access for archive upload and self-deallocation. Account, subscription, role-assignment, and credential details are intentionally not published. No credentials are stored in this repository.

Before any separately authorized future cycle, copy [the Azure config example](../research/azure_cycle.example.json) outside Git and supply new placeholders for the persistent disk, runtime, SSH identity, and archive. Stage the English manifest/shards and campaign JSON there, and verify space for a full checkpoint plus independent download. `/mnt` is Azure's temporary resource disk and is rejected for campaign data/results. The campaign JSON `data_root` and `results_root` must match the staged paths. Keep `remote_python` and `remote_env` pinned for the whole campaign.

The controller requires appropriately scoped Azure and SSH access supplied outside Git. The worker uses managed identity; no SAS, account key, or GitHub credential is copied to the VM. Native preflight records actual Torch/CUDA/DeepGEMM/FA3 identity for each candidate.

Inspect without powering on the VM:

```bash
python -m research.azure_cycle inspect --config /absolute/path/to/azure_cycle.json
```

After the data gate passes **and the human launches the paid campaign**, run the baseline screen from its committed base revision:

```bash
python -m research.azure_cycle run --config /absolute/path/to/azure_cycle.json --trial-id baseline --stage screen
```

Kimi then studies the downloaded `analysis/<cycle-id>/` records, writes the [diagnostic review](../notes/REVIEW_TEMPLATE.md), researches and commits a candidate while the VM is deallocated, and invokes one `run` for that candidate. Full continuation and QAT use the same controller command with `--stage full` or `--stage qat_probe --qat-rationale '...'` on the exact parent commit. There is no research or coding delay on the paid GPU. Every VM boot, data check, trial, upload and verification counts against `max_vm_hours`; the controller reserves 30 minutes by default for the archive. The protected training runner still enforces its own campaign GPU-hour cap. Archive verification may exceed the reserve if storage is unexpectedly slow; data safety takes precedence over the estimate.

If the owner stops an active experiment, use `python -m research.azure_cycle stop --config /absolute/path/to/azure_cycle.json --cycle-id <id>`. This sends SIGTERM to the trial child so Terminus can checkpoint; it leaves the systemd worker alive to archive, verify and deallocate. It is not an immediate VM power-off command. If the worker is already archiving, let it finish.

The controller refuses a new cycle if its append-only `azure_cycles.jsonl` has an unfinished start, a VM that was already running, or a deallocated trial without a confirmed archive. On a **pretrial** failure it deallocates its VM because no trial/checkpoint has begun, except if an older auto-started GPU workload is detected: then it leaves that workload and VM untouched for owner review. Once the systemd worker starts, a failed archive intentionally leaves the VM on. Inspect its `systemctl status`/journal, repair the cause, then run `python -m research.azure_cycle repair --config /absolute/path/to/azure_cycle.json --cycle-id <original-id>`. The repair checks that the old worker and GPU are idle, archives existing files to a **new** Blob prefix without rerunning training, verifies them, and deallocates. Do not deallocate a VM with unverified trial artifacts. If controller connectivity fails during shutdown and the VM is already deallocated, run `python -m research.azure_cycle reconcile --config /absolute/path/to/azure_cycle.json --cycle-id <id>`; it confirms the Blob receipt before closing local bookkeeping. If the VM is still running, inspect the worker or use `repair` after it fails.

The downloaded diagnostic files include per-layer, per-parameter, positional, task and GPU utilization/power telemetry. `nvidia-smi dmon` is sampled hardware context, not a kernel profile or MFU proof. Analyze beneficial and adverse signals before selecting the next architecture. The framework measures training speed and memory directly; inference latency and larger-model transfer still need finalist measurements.
