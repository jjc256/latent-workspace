#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="lwi"

# Create conda environment
conda create -n "$ENV_NAME" python=3.11 -y

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install packages
pip install torch torchvision torchaudio
pip install mlx mlx-lm
pip install open-clip-torch chromadb
pip install transformers accelerate
pip install gymnasium
pip install datasets langdetect

# Verify MPS
python -c "
import torch
assert torch.backends.mps.is_available(), 'MPS not available!'
print('MPS available:', torch.backends.mps.is_available())
print('PyTorch version:', torch.__version__)
"

echo "Phase 0 complete. Activate the env and run model download separately:"
echo "  conda activate $ENV_NAME"
echo "  python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct -q"
