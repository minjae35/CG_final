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

echo "[build] Installing CUDA extensions used by the pipeline"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "[build] Installing simple-knn (CUDA ext)"
# Build from a temporary copy so we can apply a tiny toolchain-compat patch
# without modifying the git submodule checkout.
cp -a submodules/gaussian-splatting/submodules/simple-knn "${TMP_DIR}/simple-knn"
python - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["TMP_DIR"]) / "simple-knn" / "simple_knn.cu"
txt = p.read_text(encoding="utf-8")
if "<cfloat>" not in txt:
    # Ensure FLT_MAX is defined (nvcc toolchains can miss it otherwise).
    txt = txt.replace("#include <cuda.h>\n", "#include <cuda.h>\n#include <cfloat>\n", 1)
    p.write_text(txt, encoding="utf-8")
PY
pushd "${TMP_DIR}/simple-knn" >/dev/null
python -m pip install -v --no-build-isolation .
popd >/dev/null

echo "[build] Installing diff-gaussian-rasterization (CUDA ext)"
# Build from a temporary copy so we can add missing <cstdint> include on some toolchains
# without modifying the git submodule checkout.
cp -a submodules/gaussian-splatting/submodules/diff-gaussian-rasterization "${TMP_DIR}/diff-gaussian-rasterization"
python - <<'PY'
import os
from pathlib import Path

h = Path(os.environ["TMP_DIR"]) / "diff-gaussian-rasterization" / "cuda_rasterizer" / "rasterizer_impl.h"
txt = h.read_text(encoding="utf-8")
if "<cstdint>" not in txt:
    # Provide uint32_t/uint64_t/std::uintptr_t on strict compilers.
    txt = txt.replace("#include <vector>\n", "#include <vector>\n#include <cstdint>\n", 1)
    h.write_text(txt, encoding="utf-8")
PY
pushd "${TMP_DIR}/diff-gaussian-rasterization" >/dev/null
python -m pip install -v --no-build-isolation .
popd >/dev/null

echo "[build] Done."
