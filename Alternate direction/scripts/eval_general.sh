#!/bin/bash
# General-capability eval suite for Spark-X2.6-1.7B (instruct), lm-evaluation-harness.
cd ~/probe
mkdir -p results/evals
MODEL_ARGS="pretrained=XHToken/Spark-X2.6-1.7B,trust_remote_code=True,dtype=bfloat16"
COMMON="--model hf --model_args $MODEL_ARGS --device cuda --batch_size auto --output_path results/evals"

echo "=== MCQ / loglikelihood tasks (no chat template, comparable to literature) ==="
python3 -m lm_eval $COMMON --tasks mmlu --num_fewshot 5 --log_samples 2>&1 | tail -30
python3 -m lm_eval $COMMON --tasks arc_challenge --num_fewshot 25 2>&1 | tail -15
python3 -m lm_eval $COMMON --tasks hellaswag --num_fewshot 10 2>&1 | tail -15
python3 -m lm_eval $COMMON --tasks winogrande --num_fewshot 5 2>&1 | tail -15
python3 -m lm_eval $COMMON --tasks boolq --num_fewshot 0 2>&1 | tail -15

echo "=== Generative tasks (chat template applied) ==="
python3 -m lm_eval $COMMON --apply_chat_template --tasks gsm8k --num_fewshot 5 --gen_kwargs temperature=0 2>&1 | tail -15
python3 -m lm_eval $COMMON --apply_chat_template --tasks humaneval --num_fewshot 0 --gen_kwargs temperature=0 --confirm_run_unsafe_code 2>&1 | tail -15
python3 -m lm_eval $COMMON --apply_chat_template --tasks ifeval --gen_kwargs temperature=0 2>&1 | tail -15

echo "=== GENERAL EVALS DONE ==="
