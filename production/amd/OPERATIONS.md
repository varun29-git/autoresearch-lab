# Persistent operations for a human-launched MI325X run

These tools never provision hardware. The operator supplies real allocation
and provider information and starts the campaign. CPU checks cover failure
handling; actual MI325X recovery and provider shutdown remain native gates.

## Allocation and provider controls

`python -m production.amd.allocation --output /external/allocation.json
--allocation-id <provider-allocation-id> --started-unix <actual-allocation-time>
--gpu-hours 5760 --gpus 8` records the original billing start and refuses to
reset an existing record. Include setup/idle/recovery in that same allocation.

An external supervisor policy uses `Policy` fields in `supervisor.py`:
`allocation_record`, `provider_stop_command` and `provider_verify_command`
are required. The two commands are argv arrays, never shell strings. They must
identify the actual allocation explicitly and use ambient credentials. Verify
must return zero only when the provider confirms the resource is no longer
billable. A successful shutdown request alone does not prove that. Do not use
placeholder commands for a paid campaign. Commands are hashed, not printed.

The default reserve is 1,800 node-seconds, including up to 1,200 seconds for
checkpoint/upload shutdown and two provider command timeouts. Configure and
measure those bounds for the actual storage route. The independent watchdog
checks the original allocation deadline and controller heartbeat. Numerical,
data, configuration and unknown failures are terminal; only explicit transient
network failures/EX_TEMPFAIL may use the bounded retry allowance. Counters
persist in the run's `supervision/` directory.

Each attempt writes an append-only log. The default 1 GiB attempt-log limit
triggers graceful termination and provider shutdown on a log storm; it is
checked at the polling cadence and can overshoot between polls. This is a
failure bound, not log rotation. Trainer metrics and incidents remain separate
append-only records. Checkpoint disk-floor monitoring covers those files too.

The human starts supervision with an external JSON argv array for the qualified
training command:

```bash
python -m production.amd.supervisor run \
  --policy /external/supervisor-policy.json \
  --command-json /external/training-command.json --run-dir /external/run
```

The command must target that same run directory and pass `--operations-config`.
The supervisor owns `--resume` selection on retries, replacing an initial resume
argument when necessary. A provider-specific hook and independent watchdog must
be tested before unattended rental; writing these tools is not that test.

## Training operations and checkpoint storage

An external `OperationsConfig` JSON needs `allocation_record` and
`backup_destination` (a dedicated S3 prefix or separate durable directory).
Optional fields set upload byte/age bounds, disk floor, timeouts, protected
checkpoint names and local retention (default three; minimum two). The same
policy and allocation identity must survive resume. Do not put credentials in
this file. Use:

```bash
bash production/amd/train.sh --operations-config /external/operations.json
```

Unbounded training requires this configuration. Bounded native benchmarks and
`--stop-after-updates` remain usable without a month-long operating policy.
The supervisor and trainer share the original allocation record; a restarted
trainer cannot obtain another month by resetting `time_limit_seconds`.

`training.async_checkpoint=true` enables one immutable CPU snapshot followed
by background serialization and hashing. GPU→CPU capture is synchronous; no
background worker reads live model/optimizer tensors. One pending snapshot
bounds host memory. Errors propagate to all ranks at configured polling/checkpoint
boundaries, and final checkpoints drain before successful completion. Measure
CPU RAM, GIL impact, capture time and cadence on the node. This is not a claim
that asynchronous checkpointing is free or always faster.

The uploader publishes payload, receipt and then a remote completion marker.
Local durable markers appear only after remote verification. Multipart S3
SHA256 is handled as a composite checksum, not an ETag or whole-file digest.
Active retention on the upload worker checks the remote copy again and preserves recent/protected
snapshots and latest. Shared checkpoint locks protect committed readers during
pruning. Existing stopped-run `production.checkpoints` maintenance is separate.
A dedicated prefix is required; do not point multiple independent writers at
one prefix. Provider lifecycle rules should expire abandoned multipart uploads.

Restore into a fresh external run directory:

```bash
python -m production.amd.checkpoint_service \
  --destination s3://bucket/dedicated-run-prefix \
  --checkpoint step_00001000.pt --output-dir /external/restored-run
```

Restore verifies the remote marker, run contract, payload hash/size and counters
before publishing `latest.json`. Retain the exact source/image, configs,
manifests, data, preflight/runtime/staging receipts separately as well. Restoring
weights alone is not an acceptance test for the complete provider environment.
