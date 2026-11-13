# COMMANDS.md — exact reproduction recipe

VM: AWS EC2 g5.xlarge, 1× NVIDIA A10G 24GB, Ubuntu 22.04, driver 595.112.07 (CUDA 13.2), torch 2.6.0+cu124.

## Environment setup

```bash
python3.11 -m venv ~/venv-spark
~/venv-spark/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/venv-spark/bin/pip install lm_eval==0.4.13 accelerate hf_transfer
~/venv-spark/bin/pip install "transformers==4.57.1"
~/venv-spark/bin/pip install "huggingface_hub==0.36.2" "datasets==4.3.0"   # resolve datasets/hub conflict with tf 4.57.1
~/venv-spark/bin/pip install sentencepiece protobuf tiktoken blobfile

python3.11 -m venv ~/venv-mcp
~/venv-mcp/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/venv-mcp/bin/pip install lm_eval==0.4.13 accelerate hf_transfer
~/venv-mcp/bin/pip install "transformers==5.6.2"
~/venv-mcp/bin/pip install sentencepiece protobuf tiktoken blobfile
# resolves to: datasets 5.2.0, huggingface_hub 1.33.0
```

Why two venvs: `XHToken/Spark-X2.6-1.7B` requires `transformers==4.57.1` (5.x breaks its tied-weight
loading); `openbmb/MiniCPM5-*` ships `transformers_version 5.6.2` and its tokenizer emits
`token_type_ids` that its model rejects under 4.57.1 (`ValueError: model_kwargs not used`).
lm_eval 0.4.13 identical in both venvs; sampling conditions identical at the sampling level.

## One-time model-config patch (Spark)

Spark's `generation_config.json` ships `top_k: -1` (vLLM-style "disabled"), which
transformers 4.57.1 rejects (`ValueError: top_k has to be a strictly positive integer`).
Removed the `top_k` key from the cached snapshot's `generation_config.json` — this is exactly
"top_k unset" per the shootout spec. Original preserved at `results/spark_generation_config.ORIGINAL.json`.
The custom scripts additionally null any non-positive `top_k` at load time.

No chat-template patch needed: Spark's template defaults to `enable_thinking=true,
reasoning_effort='high'` when no kwargs are passed, which is what lm_eval sends.

## Benchmark commands

All lm_eval runs (per model; `<VENV>` = venv-spark for Spark, venv-mcp for MiniCPM5):

```bash
# GSM8K (stock lm_eval task: 5-shot, test split n=1319, exact_match flexible-extract)
<VENV>/bin/python -m lm_eval run --model hf \
  --model_args pretrained=<MODEL>,trust_remote_code=True,dtype=bfloat16 \
  --tasks gsm8k --device cuda --batch_size 16 --apply_chat_template \
  --gen_kwargs temperature=1.0,top_p=0.95,do_sample=true,max_gen_toks=4096 \
  --output_path ~/shootout/results/lmeval_<TAG>_gsm8k --log_samples

# IFEval (lm_eval stock task)
<VENV>/bin/python -m lm_eval run --model hf \
  --model_args pretrained=<MODEL>,trust_remote_code=True,dtype=bfloat16 \
  --tasks ifeval --device cuda --batch_size 16 --apply_chat_template \
  --gen_kwargs temperature=1.0,top_p=0.95,do_sample=true,max_gen_toks=4096 \
  --output_path ~/shootout/results/lmeval_<TAG>_ifeval --log_samples

# HumanEval (lm_eval stock task; executes generated code)
<VENV>/bin/python -m lm_eval run --model hf \
  --model_args pretrained=<MODEL>,trust_remote_code=True,dtype=bfloat16 \
  --tasks humaneval --device cuda --batch_size 16 --apply_chat_template \
  --gen_kwargs temperature=1.0,top_p=0.95,do_sample=true,max_gen_toks=4096 \
  --confirm_run_unsafe_code \
  --output_path ~/shootout/results/lmeval_<TAG>_humaneval --log_samples

# AIME 2025 / AIME 2026 (custom script, 4 samples/problem, mean pass@1)
<VENV>/bin/python ~/shootout/scripts/aime_bench.py --model <MODEL> \
  --dataset math-ai/aime25 --split test --samples 4 --batch-size 8 \
  --max-new 8192 --reasoning high --out ~/shootout/results/aime_aime25_<TAG>.json
<VENV>/bin/python ~/shootout/scripts/aime_bench.py --model <MODEL> \
  --dataset math-ai/aime26 --split test --samples 4 --batch-size 8 \
  --max-new 8192 --reasoning high --out ~/shootout/results/aime_aime26_<TAG>.json

# GPQA Diamond (custom script; mirror dataset hendrydong/gpqa_diamond_mc n=198,
# official Idavidrein/gpqa is gated; 0-shot CoT, boxed-letter exact match)
<VENV>/bin/python ~/shootout/scripts/gpqa_bench.py --model <MODEL> \
  --batch-size 8 --max-new 8192 --reasoning high --out ~/shootout/results/gpqa_<TAG>.json

# Throughput (prefill tok/s; decode tok/s batch 1 and batch 16; peak VRAM)
<VENV>/bin/python ~/shootout/scripts/throughput.py --model <MODEL> \
  --out ~/shootout/results/throughput_<TAG>.json
```

Driver script: `~/shootout/scripts/run_model.sh <MODEL> <VENV> <TAG>` runs the whole sequence.

Sampling everywhere: `temperature=1.0, top_p=0.95, top_k unset, do_sample=true`.
Reasoning: Spark `reasoning_effort=high` (chat-template default); MiniCPM5 has no toggle —
it thinks by default (verified: emits `<think>` blocks); noted in NOTES.md.
