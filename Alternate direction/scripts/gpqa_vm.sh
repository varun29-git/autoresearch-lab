#!/bin/bash
# GPQA-only VM: setup + run + self-shutdown. usage: gpqa_vm.sh <model> <tag> <tfver>
set -x
MODEL=$1
TAG=$2
TFVER=$3
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -q "transformers==$TFVER" "jinja2>=3.1.0" datasets accelerate sentencepiece protobuf safetensors
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
mkdir -p ~/bench/results ~/bench/logs ~/bench/scripts
mv ~/fast_bench.py ~/gpqa_only.py ~/bench/scripts/ 2>/dev/null || true
export HF_ALLOW_CODE_EVAL=1
python3 ~/bench/scripts/gpqa_only.py "$MODEL" "$TAG" ~/bench/results > ~/bench/logs/gpqa_$TAG.log 2>&1
echo "gpqa rc=$?"
touch ~/bench/results/GPQADONE
sudo shutdown -h now
