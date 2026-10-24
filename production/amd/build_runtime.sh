#!/usr/bin/env bash
# Builds dependencies on an existing Linux host; no GPU allocation or training.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${RUNTIME_ROOT:?Choose a fresh external runtime directory}"
PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" - "$RUNTIME_ROOT" <<'PY'
import pathlib, platform, sys, torch
root = pathlib.Path(sys.argv[1]).resolve()
repo = pathlib.Path.cwd()
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
assert sys.version_info >= (3, 10) and torch.version.hip, 'use a ROCm PyTorch image'
assert root != repo and repo not in root.parents, 'runtime must be outside Git'
assert not root.exists(), 'runtime directory must be fresh'
PY
"$PYTHON_BIN" -m venv --system-site-packages "$RUNTIME_ROOT/venv"
PYTHON_BIN="$RUNTIME_ROOT/venv/bin/python"
"$PYTHON_BIN" - "$RUNTIME_ROOT" <<'PY'
import importlib.metadata as m, pathlib, sys
root = pathlib.Path(sys.argv[1])
lines = []
for name in ('torch', 'triton', 'pytorch-triton-rocm'):
    try: lines.append(f'{name}=={m.version(name)}')
    except m.PackageNotFoundError: pass
(root / 'tensor-constraints.txt').write_text('\n'.join(lines) + '\n')
PY
"$PYTHON_BIN" -m pip install -c "$RUNTIME_ROOT/tensor-constraints.txt" -r production/amd/requirements.txt
export GPU_ARCHS=gfx942 PYTORCH_ROCM_ARCH=gfx942 AITER_USE_SYSTEM_TRITON=1
export AITER_JIT_DIR="$RUNTIME_ROOT/cache/aiter" TRITON_CACHE_DIR="$RUNTIME_ROOT/cache/triton"
export AITER_MAX_JOBS=${AITER_MAX_JOBS:-8} MAX_JOBS=${MAX_JOBS:-8}
git clone https://github.com/ROCm/aiter.git "$RUNTIME_ROOT/aiter"
git -C "$RUNTIME_ROOT/aiter" checkout --detach ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3
git -C "$RUNTIME_ROOT/aiter" submodule update --init --recursive
"$PYTHON_BIN" -m pip install -c "$RUNTIME_ROOT/tensor-constraints.txt" -r "$RUNTIME_ROOT/aiter/requirements.txt"
PREBUILD_KERNELS=0 "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation "$RUNTIME_ROOT/aiter"
# CK attention is a separately selectable fallback; do not enable FA Triton SWA.
git clone https://github.com/Dao-AILab/flash-attention.git "$RUNTIME_ROOT/flash-attention"
git -C "$RUNTIME_ROOT/flash-attention" checkout --detach 49774b31c8d1172a5b28f72e9aed0f322bf4ccba
git -C "$RUNTIME_ROOT/flash-attention" submodule update --init --recursive
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE OPT_DIM=64,128 \
  "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps "$RUNTIME_ROOT/flash-attention"
"$PYTHON_BIN" -m pip install --no-deps 'muon-optimizer @ git+https://github.com/KellerJordan/Muon.git@f98f1cacc0263b04290753e32be8d498c1efc806'
if [[ ${INSTALL_TE:-1} == 1 ]]; then
  git clone https://github.com/ROCm/TransformerEngine.git "$RUNTIME_ROOT/TransformerEngine"
  git -C "$RUNTIME_ROOT/TransformerEngine" checkout --detach 130099ce48dc806398611de9960f8e0c486ad5f1
  git -C "$RUNTIME_ROOT/TransformerEngine" submodule update --init --recursive
  NVTE_USE_ROCM=1 NVTE_FRAMEWORK=pytorch NVTE_FUSED_ATTN=0 \
    NVTE_CMAKE_EXTRA_ARGS='-DCMAKE_HIP_ARCHITECTURES=gfx942 -DUSE_HIPKITTENS_GEMM=ON' \
    "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps "$RUNTIME_ROOT/TransformerEngine"
fi
"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps .
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m pip freeze --all > "$RUNTIME_ROOT/packages.freeze.txt"
"$PYTHON_BIN" - "$RUNTIME_ROOT" <<'PY'
from pathlib import Path
import shlex, sys, torch
root = Path(sys.argv[1])
assert torch.version.hip, 'the build replaced ROCm torch'
values = {'PYTHON_BIN': root / 'venv/bin/python', 'GPU_ARCHS': 'gfx942',
          'PYTORCH_ROCM_ARCH': 'gfx942', 'AITER_USE_SYSTEM_TRITON': '1',
          'AITER_JIT_DIR': root / 'cache/aiter', 'TRITON_CACHE_DIR': root / 'cache/triton',
          'NVTE_FP8_BLOCK_SCALING_FP32_SCALES': '1', 'OMP_NUM_THREADS': '4'}
(root / 'activate-runtime.sh').write_text(''.join(f'export {k}={shlex.quote(str(v))}\n' for k, v in values.items())
    + f'export PATH={shlex.quote(str(root / "venv/bin"))}:"$PATH"\n')
PY
