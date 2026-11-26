#!/usr/bin/env bash
# Human-operated launch wrapper. Runtime capture does not start training.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RUNTIME_REPORT:?Capture and review the H100 runtime receipt first}"
: "${CONFIG:?Set the production config}"
: "${OUTPUT_DIR:?Set the external run directory}"
"${PYTHON_BIN:-python}" -m production.runtime --verify "$RUNTIME_REPORT"
"${PYTHON_BIN:-python}" - "$CONFIG" "$OUTPUT_DIR" <<'PY'
import sys
from production.artifacts import external_path
from production.recipe import SELECTED_MODEL
from terminus.model import BitNetLlamaConfig
from terminus.train import load_config
external_path(sys.argv[1])
external_path(sys.argv[2])
model, _, _ = load_config(sys.argv[1])
expected = BitNetLlamaConfig(**(SELECTED_MODEL | {'weight_mode': model.weight_mode}))
if model != expected:
    raise ValueError('model differs from the reviewed production geometry and arithmetic')
PY
exec bash scripts/train.sh "$@"
