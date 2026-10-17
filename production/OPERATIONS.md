# Production engineering handoff

The selected architecture is in `recipe.py` and its reviewed delta is in
`ARCHITECTURE.md`. Data preparation and parallelism are separate workstreams.
All examples below use external artifact directories. Only the human launches
or stops paid GPU work. These commands prepare and inspect artifacts unless
explicitly identified as training or inference.

## Runtime and launch

`build_runtime.sh` provides a candidate Linux x86-64 rebuild using Python 3.10,
PyTorch 2.8.0+cu126, Triton 3.4.0, pinned FA3, pinned DeepGEMM and pinned Muon.
The host needs a C++ build toolchain, Python venv support, Git, CUDA 12.6 for
native extension builds and CUDA 13.2 for DeepGEMM JIT. The driver must support
that toolchain. It creates an isolated environment outside Git; it does not
provision a host. Package dependencies are resolved under the supplied pins,
then the actual installed environment is captured. This is not yet a verified
clean-machine build or a fully locked OCI image.

```bash
RUNTIME_ROOT=/external/runtime PYTHON_BIN=/usr/bin/python3.10 \
  bash production/build_runtime.sh
source /external/runtime/activate-runtime.sh
"$PYTHON_BIN" -m production.runtime --output /external/runtime-identity.json
"$PYTHON_BIN" -m terminus.preflight --config /external/dense.json \
  --output /external/dense-preflight.json
```

Freeze the successfully built environment as an immutable image before the
campaign, retaining its digest externally. `production.runtime` records loaded
native source/binary hashes, all installed package versions, driver, both
compiler versions and binary hashes. Its CPU mode exercises receipt plumbing;
it cannot substitute for a CUDA runtime receipt. Native preflight remains a
separate numerical check and does not certify full-size optimizer stability.

For the human-operated launch, use `production/train.sh` with the existing
CONFIG, MANIFEST, DATA_ROOT, OUTPUT_DIR and PREFLIGHT_REPORT variables plus
RUNTIME_REPORT. The wrapper verifies the runtime and selected model settings
before delegating to `scripts/train.sh`. Keep the source/runtime fixed for exact
recovery. The new 24-layer model requires a fresh training run.

## Checkpoints and operating health

The trainer already commits checksummed checkpoints atomically and records
loss, gradient norm, clipping coefficient, learning rates, memory, sampled
stability probes and incidents. SIGINT/SIGTERM requests a checkpoint after the
current update. The operator controls signals; these tools never send them.

Inspect disk/log health with explicit thresholds chosen for the real checkpoint
size and checkpoint/evaluation duration:

```bash
python -m production.status --run-dir /external/dense-run \
  --min-free-gb <required_free_space> --max-idle-seconds <allowed_log_gap>
python -m production.checkpoints --run-dir /external/dense-run \
  --output /external/checkpoint-audit.json
```

Status returns a nonzero exit code for low disk, stale logs or an incident newer
than the latest event. It is a local health signal, not a distributed watchdog.
The checkpoint audit hashes every saved payload and identifies a recoverable
exact checkpoint, corrupt receipts and orphan payloads. Audit large directories
outside throughput timing windows. It does not deserialize optimizer tensors;
the trainer's normal exact-resume validation still applies.

Retention uses the trainer's exclusive run lock and therefore requires the run
to be stopped. First obtain the plan, then explicitly apply it:

```bash
python -m production.checkpoints --run-dir /external/dense-run \
  --archive /durable/backup/dense-run --keep-last 3 \
  --protect step_00001000.pt --output /external/retention-plan.json
python -m production.checkpoints --run-dir /external/dense-run \
  --archive /durable/backup/dense-run --keep-last 3 \
  --protect step_00001000.pt --apply --output /external/retention-applied.json
```

Use actual checkpoint filenames. All committed payloads are copied, fsynced
and checksum-verified before pruning; the latest checkpoint and protected
milestones are retained. The archive includes run.json, payloads and receipts
and supports the same exact-checkpoint loading path. Provision `/durable` on
independent durable storage. This utility does not create storage accounts or
provide unattended backup during a live run. Budget local disk for the full
interval between maintenance windows.

After an abrupt failure, audit and resume the reported exact `step_*.pt` path
with the original config, source, runtime and world size. Do not edit receipts
to bypass contract failures. Preserve and move orphan/corrupt files aside while
the run is stopped before retrying an update with the same checkpoint filename.
Test storage loss and recovery on the intended cluster before relying on it.

## Evaluation and release comparison

`production.evaluate` verifies the checkpoint contract, tokenizer digest and
every requested task file. A task entry needs status `fetched`, `file`, `sha256`,
a fixed `version` or `revision`, and role `development` or `final`. JSONL rows
contain `input`, `output`, optional `id` and `language`. Requested tasks cannot
be silently skipped. Final-set access requires `--include-final`. Prompts must
be nonempty and complete examples must fit the model context.

```bash
python -m production.evaluate --checkpoint /external/dense-run/checkpoints/step_00001000.pt \
  --tokenizer /external/tokenizer.json --eval-manifest /external/eval/manifest.json \
  --eval-root /external/eval --tasks scalingbench/mmlu,scalingbench/math \
  --device cuda --output /external/dense-eval.json
python -m production.evaluate --bundle /external/ternary-cpu \
  --eval-manifest /external/eval/manifest.json --eval-root /external/eval \
  --tasks scalingbench/mmlu,scalingbench/math --output /external/cpu-eval.json
python -m production.compare_eval --baseline /external/qat-native-eval.json \
  --candidate /external/cpu-eval.json --output /external/export-comparison.json
```

Receipts bind exact checkpoint, task/tokenizer hashes, evaluator source and
per-item records. Comparisons require identical item coverage and token counts
and report paired NLL differences by task and language. Without an explicit
`--max-loss-increase` they produce a comparison only. With a threshold, every
task/language upper bootstrap bound must fit it. Choose release thresholds
before reading final-set results; none have been invented here.

The implemented metric is teacher-forced reference NLL. It is not generated
MMLU accuracy, mathematical-answer accuracy or code pass@1. Freeze generated
benchmark protocols separately if those are release requirements. Defining and
building the multilingual/domain holdouts and contamination audit remains in
the excluded data workstream. The same comparison tool can be used for dense
versus QAT and native-QAT versus CPU export on matched evaluation inputs.

## Dense-to-QAT: deliberately incomplete settings

The operator has not selected QAT rates or the dense/QAT token split. Keep these
unset. Generate an external specification to fill later:

```bash
python -m production.qat_template --output /external/qat-specification.json
```

The resulting specification has null rates, token budgets, schedule, batch,
transition policy and quality threshold. It is intentionally not a training
configuration. No QAT campaign is launch-ready until these values are chosen.

The existing `terminus.transition` command verifies the parent checkpoint and
converts a dense run to the same architecture in ternary mode. It preserves
optimizer moments with explicit child learning rates. Other optimizer changes
require an explicit reset. Same-shape architectural changes (including the
attention window) are rejected. RNG reset requests are honored even when world
size is unchanged, and warmup/decay must fit the remaining QAT phase.

Run conversion on the intended training runtime so its resume contract matches.
Use a fresh QAT directory, an explicit data policy and a fully specified config.
The native DeepGEMM path supports an immediate full-ternary switch; a gradual
quantization curriculum is not validated on that path. CPU transition tests
establish state/cursor mechanics, not successful production quality transfer.

## Packed ternary CPU deployment

```bash
python -m production.export --checkpoint /external/qat-run/checkpoints/step_00002000.pt \
  --tokenizer /external/tokenizer.json --output /external/ternary-cpu
python -m production.infer --bundle /external/ternary-cpu --prompt 'Hello' \
  --max-new-tokens 128 --threads 4
```

Export requires a fully ternary checkpoint. It preserves block scales as FP32,
packs four projection weights per byte and preserves the unquantized state
(including the tied embedding/head) at its stored precision. A selected-size
FP32 checkpoint therefore produces roughly 0.93 GB of tensor storage, including
about 0.60 GB for the unquantized tied embedding/head; two bits applies to the
projection matrices, not the entire model. This excludes KV caches, activations,
temporary workspace, framework overhead and container metadata.

The checksummed bundle has no optimizer state or FP32 latent projection weights.
The CPU loader maps the packed file and unpacks small output-row tiles for
PyTorch matrix multiplication. It retains local/global masks, RoPE, GQA,
Peri-LN, Q/K gains, value residual, activation quantization and tied readout.
CUDA-specific fusion/FA3/DeepGEMM are replaced by the CPU operator path. BF16
residual configurations use CPU BF16 autocast. Floating-point kernel rounding
can differ from the H100 path; comparison on real held-out inputs remains due.

This is a working portable packed-storage runtime. It does not include an
ISA-specific two-bit GEMV kernel, GGUF, bitnet.cpp or vLLM integration. Measure
prefill/decode latency, RSS and quality on the actual edge CPU before setting
device support or performance claims. Full and cached reduced-model tests
exercise the 24-layer 5:1 layout; they do not establish production model quality.

## Remaining acceptance evidence

Local code and CPU tests cover the new geometry, evaluation contracts, checkpoint
backup/retention, QAT state transfer and packed CPU export/inference. The following
remain open: clean Linux runtime rebuild; native H100 preflight and sustained
full-size spectral-Muon updates; actual-cluster recovery and storage behavior;
QAT settings and quality transfer; and production-size CPU quality/latency.
Data and parallelism are intentionally separate. No 32-GPU speed, total token
budget feasibility, trained checkpoint or release quality is certified here.
