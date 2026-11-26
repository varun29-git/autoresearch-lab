#!/bin/bash
# MiniCPM5 benchmark VM: setup + full run + self-shutdown.
# usage: mcp_vm_setup.sh <model_repo> <tag>
set -x
MODEL=$1
TAG=$2
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -q "transformers==5.6.2" lm_eval==0.4.13 langdetect datasets accelerate sentencepiece protobuf safetensors
python3 -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__, 'cuda', torch.cuda.is_available())"

export HF_ALLOW_CODE_EVAL=1
PY=python3
RES=$HOME/bench/results
LOGS=$HOME/bench/logs
SCR=$HOME/bench/scripts
MA="pretrained=$MODEL,trust_remote_code=True,dtype=bfloat16"
GENKW="temperature=1.0,top_p=0.95,do_sample=true,max_gen_toks=4096"

$PY -m lm_eval run --model hf --model_args "$MA" --tasks gsm8k --device cuda --batch_size 16 --apply_chat_template --gen_kwargs "$GENKW" --output_path $RES/lmeval_${TAG}_gsm8k --log_samples > $LOGS/${TAG}_gsm8k.log 2>&1
echo "gsm8k rc=$?"
$PY $SCR/aime_bench.py --model $MODEL --dataset math-ai/aime25 --split test --samples 4 --batch-size 8 --max-new 8192 --reasoning high --out $RES/aime_aime25_${TAG}.json > $LOGS/${TAG}_aime25.log 2>&1
echo "aime25 rc=$?"
$PY $SCR/gpqa_bench.py --model $MODEL --batch-size 16 --max-new 4096 --reasoning high --out $RES/gpqa_${TAG}.json > $LOGS/${TAG}_gpqa.log 2>&1
echo "gpqa rc=$?"
$PY -m lm_eval run --model hf --model_args "$MA" --tasks ifeval --device cuda --batch_size 16 --apply_chat_template --gen_kwargs "$GENKW" --output_path $RES/lmeval_${TAG}_ifeval --log_samples > $LOGS/${TAG}_ifeval.log 2>&1
echo "ifeval rc=$?"
$PY -m lm_eval run --model hf --model_args "$MA" --tasks humaneval --device cuda --batch_size 16 --apply_chat_template --gen_kwargs "$GENKW" --confirm_run_unsafe_code --output_path $RES/lmeval_${TAG}_humaneval --log_samples > $LOGS/${TAG}_humaneval.log 2>&1
echo "humaneval rc=$?"
$PY $SCR/throughput.py --model $MODEL --out $RES/throughput_${TAG}.json > $LOGS/${TAG}_throughput.log 2>&1
echo "throughput rc=$?"
touch $RES/DONE
echo "ALL DONE $TAG"
sudo shutdown -h now
