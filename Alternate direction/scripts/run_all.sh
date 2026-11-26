#!/bin/bash
# Orchestrates all probes on the VM. Logs to results/run_all.log
set -x
cd ~/probe
mkdir -p results data models

echo "=== SETUP ==="
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
python3 -m pip install -q "transformers>=4.57.1" tokenizers safetensors huggingface_hub accelerate sentencepiece datasets 2>&1 | tail -1
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi

echo "=== DOWNLOAD MODELS ==="
python3 - <<'EOF'
from huggingface_hub import snapshot_download
for repo, target in [("XHToken/Spark-X2.6-1.7B", None)]:
    snapshot_download(repo, local_dir="models/spark")
    print("downloaded", repo)
EOF
du -sh models/spark

echo "=== PROBE 1: repo facts ==="
python3 scripts/probe1_repo_facts.py

echo "=== PROBE 2: fertility ==="
python3 scripts/probe2_fertility.py

echo "=== PROBE 3: prune dry run ==="
python3 scripts/probe3_prune_dryrun.py

echo "=== PROBE 4: embeddings ==="
python3 scripts/probe4_embeddings.py

echo "=== PROBE 5: perplexity ==="
python3 scripts/probe5_perplexity.py

echo "=== PROBE 6: smoke ==="
python3 scripts/probe6_smoke.py

echo "=== ALL DONE ==="
ls -la results/
