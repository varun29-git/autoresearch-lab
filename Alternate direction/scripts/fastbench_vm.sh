#!/bin/bash
# Generic fast-bench VM: setup + run + self-shutdown.
# usage: fastbench_vm.sh <model_repo> <tag> <transformers_version>
set -x
MODEL=$1
TAG=$2
TFVER=$3
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -q "transformers==$TFVER" "jinja2>=3.1.0" datasets accelerate sentencepiece protobuf safetensors
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
mkdir -p ~/bench/results ~/bench/logs
mv ~/fast_bench.py ~/bench/ 2>/dev/null || true
export HF_ALLOW_CODE_EVAL=1
python3 ~/bench/fast_bench.py --model "$MODEL" --tag "$TAG" --outdir ~/bench/results > ~/bench/logs/fastbench_$TAG.log 2>&1
echo "fastbench rc=$?"
touch ~/bench/results/FASTDONE
sudo shutdown -h now
