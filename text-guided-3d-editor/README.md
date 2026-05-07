# Text-guided 3D indoor editor

Pipeline: **Mip-NeRF 360 (COLMAP) → 3DGS → Grounded SAM 2 → DreamGaussian → PhysGaussian** (see [documents/PRD_v2_kor.md](../documents/PRD_v2_kor.md) if present in your clone).

## Getting the code (read this first)

From the **repository root** (`CG_final/`), always clone or update submodules so these folders are populated (not empty):

```bash
# Preferred one-shot clone
git clone --recurse-submodules <REPO_URL> CG_final

# If you already cloned without submodules:
cd CG_final
git submodule update --init --recursive
```

After `git pull` when submodule revisions change:

```bash
git submodule update --init --recursive
```

Then continue with **Setup** below (`cd text-guided-3d-editor`).

## Submodule pins (record `git rev-parse HEAD` after clone)

| Submodule | URL |
|-----------|-----|
| gaussian-splatting | `text-guided-3d-editor/submodules/gaussian-splatting` |
| dreamgaussian | `text-guided-3d-editor/submodules/dreamgaussian` |
| Grounded-SAM-2 | `text-guided-3d-editor/submodules/Grounded-SAM-2` |
| PhysGaussian | `text-guided-3d-editor/submodules/PhysGaussian` |

## Setup

```bash
cd text-guided-3d-editor
bash setup.sh   # then follow printed conda / torch lines
pip install -r requirements.txt
```

Install **3DGS CUDA extensions** and **Grounded-SAM-2 / DreamGaussian / PhysGaussian** deps per upstream READMEs.

## Checkpoints (Grounded SAM 2)

`configs/pipeline_config.yaml` points at files under `submodules/Grounded-SAM-2/` (Grounding DINO + SAM 2). They are **not** in git: follow [Grounded-SAM-2/README.md](submodules/Grounded-SAM-2/README.md) to download weights into the expected paths before `pipeline.py` segmentation.

## Data (PRD v2)

Run the downloader to fetch **Mip-NeRF 360** (`360_v2.zip`) and materialize `data/mipnerf360/<scene_name>/` with `images/` and `sparse/0/` (defaults: `room`). Override `scene.dataset_url`, `scene.data_root`, or `scene.scene_name` in `configs/pipeline_config.yaml` as needed.

## Run (PYTHONPATH)

```bash
cd text-guided-3d-editor
export PYTHONPATH=src

python src/pipeline.py prepare-dataset
python src/pipeline.py train --smoke          # T4-friendly low iterations
python src/pipeline.py mode-a "a red rubber duck" --smoke
python src/pipeline.py mode-b "the wooden chair" --smoke
```

View rendering for mode B uses CUDA (`render_training_views`) to save RGB, metric depth, and per-view camera intrinsics/extrinsics next to masks.

Shell helpers: `scripts/run_mode_a.sh`, `scripts/run_mode_b.sh`, `scripts/run_evaluation.sh`.

## Tests

```bash
cd text-guided-3d-editor
export PYTHONPATH=src
pytest tests/ -q
```
