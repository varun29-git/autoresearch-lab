# Authorized unattended launch

The user instructed Codex to start the 24-hour campaign and keep it automatic
on 2026-11-01. This authorizes the H100 launch, native checks, ordered trials,
logging and deallocation at completion, failure or the allocation deadline.
The scientific matrix and token schedules remain those in RUNBOOK.md.

The launch packet is sealed from a clean Git commit. A systemd service runs
`supervisor.py` through runtime restoration, native gates and `auto.py`.
The service persists independently of SSH and the assistant turn. Its
`supervision/stdout.log` retains complete subprocess output; `events.jsonl`
records supervisor decisions and log-upload receipts; `status.json` contains
the current stage, latest training/validation record, allocation elapsed time
and terminal state. Metadata/log archives are checksum-verified in S3 every
five minutes and on exit. Ambient credentials are never included in archives.

The controller uses independent checkpoint prefixes for H/R0/R1/R2/R3 and
confirmation. Checkpoints larger than 5 GB use multipart S3 with SHA-256
validation for every part and the completed composite; the full payload hash
is tied to object metadata and the durable marker. Restores still verify the
complete downloaded payload hash. Each stage must drain uploads and verify
its latest durable checkpoint before the controller advances.

Independent guards limit allocated cost: the guest supervisor sends SIGTERM
to the whole campaign group at 23.25 hours, preserving time for checkpoint
and uploads; an Azure auto-shutdown schedule is set before the 24-hour cap;
a persistent operator watcher mirrors status and deallocates on terminal
status or deadline. VM deallocation is required to end billing.

Reference replays use the previously frozen 0.005 weight and 0.0005 moment
delta tolerances at initialization/warmup/pre-decay boundaries. Per-update
qualification records retain target counts, all group rates, full finite
state and required gradients; logical role diagnostics run every 20 updates
and at boundaries. Any numerical failure, native-contract mismatch, incomplete
arm, failed durable backup or unaffordable reforecast stops progression.
Transport requests have bounded retries; numerical failures are never
automatically repaired, reranked or silently relabeled.
