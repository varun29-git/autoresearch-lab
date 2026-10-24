#!/usr/bin/env bash
# Human-operated launcher; no provisioning, billing or automatic sweeps.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CONFIG:?Set the external AMD config}"
: "${MANIFEST:?Set the external manifest}"
: "${DATA_ROOT:?Set the immutable staged data root}"
: "${OUTPUT_DIR:?Set the external run directory}"
: "${RUNTIME_REPORT:?Capture the AMD runtime receipt}"
: "${PREFLIGHT_REPORT:?Pass the AMD native preflight}"
: "${STAGING_RECEIPT:?Fully verify this node-local data snapshot on CPU}"
: "${EXPECTED_GLOBAL_BATCH_TOKENS:?Set the reviewed global update size}"
: "${EXPECTED_WORLD_SIZE:?Set the inspected world size}"
exec "${PYTHON_BIN:-python3}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-8}" --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" --module production.amd.train \
  --config "$CONFIG" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" --runtime-report "$RUNTIME_REPORT" \
  --preflight-report "$PREFLIGHT_REPORT" --staging-receipt "$STAGING_RECEIPT" \
  --expected-world-size "$EXPECTED_WORLD_SIZE" \
  --expected-global-batch-tokens "$EXPECTED_GLOBAL_BATCH_TOKENS" "$@"
