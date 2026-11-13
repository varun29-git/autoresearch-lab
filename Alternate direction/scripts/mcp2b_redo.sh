#!/bin/bash
# MiniCPM5-2B redo VM: setup + fast bench + self-shutdown
set -x
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -q "transformers==5.6.2" "jinja2>=3.1.0" datasets accelerate sentencepiece protobuf safetensors
python3 -c "import torch; print('cuda', torch.cuda.is_available())"
mkdir -p ~/bench/scripts ~/bench/results ~/bench/logs
cp ~/fast_bench.py ~/bench/scripts/ 2>/dev/null || true
export HF_ALLOW_CODE_EVAL=1
python3 ~/bench/scripts/fast_bench.py --model openbmb/MiniCPM5-2B --tag mcp2b --outdir ~/bench/results > ~/bench/logs/fastbench_mcp2b.log 2>&1
echo "fastbench rc=$?"
touch ~/bench/results/FASTDONE
sudo shutdown -h now
