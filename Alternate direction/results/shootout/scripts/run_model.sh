#!/bin/bash
# Full benchmark phase for one model.
# usage: run_model.sh <model_repo> <venv_name> <tag>
set -u
MODEL=$1; VENV=$2; TAG=$3
PY=$HOME/$VENV/bin/python
RES=$HOME/shootout/results
LOGS=$HOME/shootout/logs
SCR=$HOME/shootout/scripts
mkdir -p $RES $LOGS

log() { echo "[$(date -u +%FT%TZ)] $TAG: $1" | tee -a $RES/progress.log; }

MODEL_ARGS="pretrained=$MODEL,trust_remote_code=True,dtype=bfloat16"
GENKW="temperature=1.0,top_p=0.95,do_sample=true,max_gen_toks=4096"

log "START model=$MODEL venv=$VENV"

# 1. GSM8K (lm_eval, 5-shot stock task, chat template)
log "gsm8k start"
$PY -m lm_eval run --model hf --model_args "$MODEL_ARGS" \
  --tasks gsm8k --device cuda --batch_size 16 --apply_chat_template \
  --gen_kwargs "$GENKW" \
  --output_path $RES/lmeval_${TAG}_gsm8k --log_samples \
  > $LOGS/${TAG}_gsm8k.log 2>&1
log "gsm8k rc=$?"

# 2. AIME 2025 + AIME 2026 (custom, 4 samples/problem)
for DS in math-ai/aime25 math-ai/aime26; do
  DSNAME=$(basename $DS)
  log "aime $DSNAME start"
  $PY $SCR/aime_bench.py --model $MODEL --dataset $DS --split test \
    --samples 4 --batch-size 8 --max-new 8192 --reasoning high \
    --out $RES/aime_${DSNAME}_${TAG}.json \
    > $LOGS/${TAG}_aime_${DSNAME}.log 2>&1
  log "aime $DSNAME rc=$?"
done

# 3. GPQA Diamond (custom, mirror dataset, 0-shot CoT, chat template)
log "gpqa start"
$PY $SCR/gpqa_bench.py --model $MODEL --batch-size 8 --max-new 8192 --reasoning high \
  --out $RES/gpqa_${TAG}.json \
  > $LOGS/${TAG}_gpqa.log 2>&1
log "gpqa rc=$?"

# 4. IFEval (lm_eval)
log "ifeval start"
$PY -m lm_eval run --model hf --model_args "$MODEL_ARGS" \
  --tasks ifeval --device cuda --batch_size 16 --apply_chat_template \
  --gen_kwargs "$GENKW" \
  --output_path $RES/lmeval_${TAG}_ifeval --log_samples \
  > $LOGS/${TAG}_ifeval.log 2>&1
log "ifeval rc=$?"

# 5. HumanEval (lm_eval, executes code -> confirm flag)
log "humaneval start"
$PY -m lm_eval run --model hf --model_args "$MODEL_ARGS" \
  --tasks humaneval --device cuda --batch_size 16 --apply_chat_template \
  --gen_kwargs "$GENKW" --confirm_run_unsafe_code \
  --output_path $RES/lmeval_${TAG}_humaneval --log_samples \
  > $LOGS/${TAG}_humaneval.log 2>&1
log "humaneval rc=$?"

# 6. Throughput
log "throughput start"
$PY $SCR/throughput.py --model $MODEL --out $RES/throughput_${TAG}.json \
  > $LOGS/${TAG}_throughput.log 2>&1
log "throughput rc=$?"

log "ALL DONE"
