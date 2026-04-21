#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="lwi"

# Create conda environment
conda create -n "$ENV_NAME" python=3.11 -y

# Install packages
conda run -n "$ENV_NAME" pip install torch torchvision torchaudio
conda run -n "$ENV_NAME" pip install mlx mlx-lm
conda run -n "$ENV_NAME" pip install open-clip-torch chromadb
conda run -n "$ENV_NAME" pip install transformers accelerate
conda run -n "$ENV_NAME" pip install gymnasium

# Verify MPS
conda run -n "$ENV_NAME" python -c "
import torch
assert torch.backends.mps.is_available(), 'MPS not available!'
print('MPS available:', torch.backends.mps.is_available())
print('PyTorch version:', torch.__version__)
"

echo "Phase 0 complete. Run model download separately:"
echo "  conda run -n $ENV_NAME python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct -q"
