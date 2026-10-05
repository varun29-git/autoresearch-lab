# Azure H100 experiment cycle for Kimi

**Prepared, not launched.** On 5 October 2026 the existing `H100_GROUP/h100` VM was `PowerState/deallocated`. It is one `Standard_NCC40ads_H100_v5` H100 NVL in Central US. The prepared English-only manifest, campaign JSON and persistent mount still need to be staged before the first trial, and the human starts the paid campaign. Do not start the VM to think, read papers, write code or review results.

Kimi runs on the inexpensive controller with Azure CLI, SSH and this repo. Each proposed architecture is researched, implemented, tested and committed **before** H100 startup. `research.azure_cycle` sends that exact commit as a git bundle; the VM needs no GitHub credential. The controller refuses an already-running VM, detects older GPU jobs, validates the persistent data/result mount and pinned H100 environment, and rechecks prepared shard hashes before each continuation/candidate trial. It then launches one systemd worker. The worker uses the protected `research.loop` runner and writes the normal preflight, training, evaluation, stability, incident, checkpoint and ledger records. It also samples `nvidia-smi dmon` during training.

After the experiment, including a failed or timed-out experiment, the worker makes a SHA-256 inventory of the entire trial directory plus a copy of the campaign lock and ledger. It uploads to a **unique Blob prefix**, lists the exact uploaded names, downloads every file into a second directory, and verifies every hash with `research.artifacts.verify`. **Only the verified path calls `az vm deallocate --no-wait`.** If upload or verification fails, the VM remains running with its managed disk intact for repair. The controller then confirms the Blob receipt and downloads the diagnostic JSON and logs for analysis while the H100 is off. Checkpoints remain in Blob and on the persistent disk; `fetch --include-checkpoints` downloads them to the controller if needed.

## One-time prerequisites

The dedicated private Blob container `terminus-autoresearch` already exists in storage account `t3logse388e1b8d9e3`. The VM managed identity has Storage Blob Data Contributor on that container and Virtual Machine Contributor on **that VM only**, enabling authenticated upload and self-deallocation. The controller user's access is scoped to this container for archive confirmation, download and emergency repair. The VM identity also had broader pre-existing storage roles; this setup did not remove or alter those earlier assignments. Azure RBAC may take a few minutes to propagate. The VM was **not** started for setup.

Before a paid cycle, fill [the Azure config](../research/azure_cycle.example.json) outside Git. Resolve `YOUR_PERSISTENT_DISK` to the actual Azure managed-disk mount, stage the English manifest/shards and campaign JSON there, and verify the results and verification-scratch directories have space for a full checkpoint plus its independent download. `/mnt` is Azure's temporary resource disk and is rejected for campaign data/results. The campaign JSON `data_root` and `results_root` must match the staged paths. The example runtime points to the previously measured H100 venv, CUDA 13.2 JIT compiler and FA3 module path; check they still exist. Keep `remote_python` and `remote_env` pinned for the whole campaign.

The controller needs Azure CLI login with access to this Blob container and permission to start the VM, plus SSH access as `azureuserboss` and passwordless `sudo` on the VM for a transient systemd unit. The worker uses `az login --identity`; no SAS, account key or GitHub credential is copied to the VM. The VM must have Azure CLI and `nvidia-smi`. Its native preflight records actual Torch/CUDA/DeepGEMM/FA3 identity for each candidate.

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
