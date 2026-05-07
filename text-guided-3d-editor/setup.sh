#!/usr/bin/env bash
# PRD Phase 0 — pin tool versions in comments; adjust for your CUDA.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> [1/6] Git submodules (from repo root use: git clone --recurse-submodules … ; if not: git submodule update --init --recursive)"
git submodule update --init --recursive

echo "==> [2/6] Create conda env (manual): conda create -n room-editor python=3.10 -y && conda activate room-editor"
echo "==> [3/6] PyTorch (example CUDA 11.8): pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118"

echo "==> [4/6] 3DGS rasterizer extensions"
pip install -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization/ 2>/dev/null || echo "Skip diff-gaussian-rasterization (build from source per 3DGS README)"
pip install -e submodules/gaussian-splatting/submodules/simple-knn/ 2>/dev/null || echo "Skip simple-knn"

echo "==> [5/6] Project requirements"
pip install -r requirements.txt

echo "==> [6/6] Optional: Grounded-SAM-2, DreamGaussian, PhysGaussian (see each README)"
echo "Done. Verify: python -c \"import torch; print(torch.cuda.is_available())\""
