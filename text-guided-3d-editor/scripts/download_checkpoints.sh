#!/usr/bin/env bash
set -euo pipefail

# Download external checkpoints that are intentionally NOT committed to git.
# Places files where configs/pipeline_config.yaml expects them:
#   submodules/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth
#   submodules/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "==> exists: $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  echo "==> download: $url"
  echo "           -> $out"
  if need_cmd curl; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
  elif need_cmd wget; then
    wget -O "$out" "$url"
  else
    echo "ERROR: need curl or wget to download checkpoints." >&2
    exit 1
  fi
}

GDINO_OUT="submodules/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
SAM2_OUT="submodules/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"

# GroundingDINO Swin-B CogCoOr (official filename used by many GSAM2 repos)
GDINO_URL="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"

# SAM2.1 Hiera Large (official public files)
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

download "$GDINO_URL" "$GDINO_OUT"
download "$SAM2_URL" "$SAM2_OUT"

echo
echo "Done."
echo " - $GDINO_OUT"
echo " - $SAM2_OUT"
