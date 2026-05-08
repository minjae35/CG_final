#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[build] Using python: $(command -v python)"
python -c "import sys; print('[build] Python', sys.version)"

# Build/install CUDA extensions into the *current* Python environment.
# This keeps submodules at pinned commits (no source edits), and makes the repo reproducible.
#
# Notes:
# - We prefer installing from source into site-packages, then ensuring runtime imports
#   pick up the installed extensions (not an unbuilt submodule checkout).
# - Some toolchains require Torch's shared libs on LD_LIBRARY_PATH for subprocesses.

echo "[build] Installing simple-knn (CUDA ext)"
pushd submodules/gaussian-splatting/submodules/simple-knn >/dev/null
python -m pip install -v --no-build-isolation .
popd >/dev/null

echo "[build] Installing diff-gaussian-rasterization (CUDA ext)"
# Some environments fail due to missing <cstdint> in compilation units pulled by CUDA builds.
# We inject the header via compiler flags rather than editing submodule sources.
export CXXFLAGS="${CXXFLAGS:-} -include cstdint"
export TORCH_NVCC_FLAGS="${TORCH_NVCC_FLAGS:-} -Xcompiler -include -Xcompiler cstdint"

pushd submodules/gaussian-splatting/submodules/diff-gaussian-rasterization >/dev/null
python -m pip install -v --no-build-isolation .
popd >/dev/null

echo "[build] Done."
