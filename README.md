# Text-guided 3D indoor editor

This repository contains a text-guided indoor 3D editing pipeline for COMS3168 Deep Learning for Computer Graphics.

Pipeline:

```text
Mip-NeRF 360 room data + COLMAP -> 3D Gaussian Splatting base scene
-> Grounded SAM 2 object selection
-> DreamGaussian object generation / insertion
-> PhysGaussian simulation and video rendering
```

Mode A inserts a generated or prebuilt object into the room scene and simulates only that inserted object. Mode B selects an existing object in the room from text and applies material-specific physics to the selected Gaussians.

## Repository Structure

- `README.md`: this submission README.
- `text-guided-3d-editor/`: main implementation code.
- `text-guided-3d-editor/src/pipeline.py`: Typer CLI entry point for dataset prep, 3DGS training, Mode A, and Mode B.
- `text-guided-3d-editor/configs/pipeline_config.yaml`: default scene, model, checkpoint, preset, and output settings.
- `text-guided-3d-editor/scripts/`: helper scripts for checkpoint download, CUDA extension build, and example Mode A/B runs.
- `text-guided-3d-editor/submodules/`: pinned upstream code for `gaussian-splatting`, `dreamgaussian`, `Grounded-SAM-2`, and `PhysGaussian`.
- `text-guided-3d-editor/data/`: local dataset directory, ignored by git.
- `text-guided-3d-editor/output/`: generated checkpoints, renders, simulations, and videos, ignored by git.

## Clone

Submodules are required. A plain `git clone` leaves the upstream folders empty.

```bash
git clone --recurse-submodules <YOUR_REPO_URL> CG_final
cd CG_final
```

If the repository was already cloned without submodules:

```bash
cd CG_final
git submodule update --init --recursive
```

After `git pull`, run the same update command if submodule pointers changed:

```bash
git submodule update --init --recursive
```

To verify the checkout, these paths should exist and should not be empty:

- `text-guided-3d-editor/submodules/gaussian-splatting/`
- `text-guided-3d-editor/submodules/dreamgaussian/`
- `text-guided-3d-editor/submodules/Grounded-SAM-2/`
- `text-guided-3d-editor/submodules/PhysGaussian/`

## Environment Setup

Use the `cg_final` conda environment.

```bash
cd /home/bgh1225/CG_final
conda create -n cg_final python=3.10 -y
conda activate cg_final

python -m pip install --upgrade pip setuptools wheel ninja
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

cd text-guided-3d-editor
pip install -r requirements.txt --no-build-isolation
bash scripts/build_cuda_extensions.sh
```

`setup.sh` is also available if you want the helper script to run submodule setup and project requirements after activating the environment:

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
conda activate cg_final
bash setup.sh
```

The manual commands above are the recommended submission setup because they explicitly use the `cg_final` environment and build the CUDA extensions without editing submodules.

Install additional upstream dependencies as needed from the pinned upstream READMEs:

- `text-guided-3d-editor/submodules/Grounded-SAM-2/`
- `text-guided-3d-editor/submodules/dreamgaussian/`
- `text-guided-3d-editor/submodules/PhysGaussian/`

Before running the pipeline, set `PYTHONPATH` from the implementation directory:

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src
```

## Data and Checkpoints

The default config downloads the Mip-NeRF 360 `room` scene into `text-guided-3d-editor/data/mipnerf360/room/`.

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py prepare-dataset
```

Expected data layout:

- `data/mipnerf360/room/images/`
- `data/mipnerf360/room/sparse/0/`

Grounding DINO and SAM 2 weights are not committed to git. Download them into the paths expected by `configs/pipeline_config.yaml`:

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
bash scripts/download_checkpoints.sh
```

Expected checkpoint files:

- `submodules/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth`
- `submodules/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt`

## Train Base 3DGS Scene

The Mode A and Mode B commands below use the smoke 3DGS checkpoint, which is faster and writes iteration `8000`.

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py train --smoke
```

Expected base checkpoint:

- `output/base_scene_smoke/point_cloud/iteration_8000/point_cloud.ply`

For a full 3DGS training run:

```bash
python src/pipeline.py train
```

Expected full checkpoint:

- `output/base_scene_full/point_cloud/iteration_30000/point_cloud.ply`

Use the same smoke/full choice consistently when running later commands.

## Run Mode A

Mode A inserts a generated or prebuilt object into the reconstructed room and simulates only the inserted object.

Fresh run that lets the pipeline generate the object:

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py mode-a "duck" --smoke --full-sds \
  --material rubber --in-place-wobble --no-auto-realism \
  --wobble-amp 0.095 --wobble-frequency 0.65 \
  --wobble-height-weight 0.25 --wobble-bottom-pin 0.15
```

If a prebuilt Gaussian object already exists at `output/generated_objects/sds_run/object_mesh_gaussians.ply`, reuse it with:

```bash
python src/pipeline.py mode-a "duck" --smoke --full-sds \
  --object-gaussians output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material rubber --in-place-wobble --no-auto-realism \
  --wobble-amp 0.095 --wobble-frequency 0.65 \
  --wobble-height-weight 0.25 --wobble-bottom-pin 0.15
```

Other supported material presets include `jelly`, `foam`, `sand`, `metal`, and `rigid`; the output folder changes with the material name.

Expected Mode A outputs for the rubber command:

- `output/generated_objects/sds_run/`: generated or staged object assets.
- `output/generated_objects/sds_run/object_scaled.ply`: object after scene-scale conversion.
- `output/merged_scenes/mode_a_merged.ply`: base scene plus inserted object.
- `output/sim_results/mode_a_rubber/phys_mode_a.json`: generated PhysGaussian config.
- `output/sim_results/mode_a_rubber/mode_a_obj_indices.npy`: inserted-object Gaussian indices.
- `output/sim_results/mode_a_rubber/frames/`: rendered frames.
- `output/sim_results/mode_a_rubber/video/output.mp4`: final Mode A video.

## Run Mode B

Mode B selects an existing object in the room with Grounded SAM 2, projects the mask back to 3D Gaussians, and simulates the selected set. The current CLI requires `--preset`.

Jelly desk preset:

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py mode-b --smoke-3dgs --preset desk_jelly --no-debug-selection
```

Sand chair/footrest preset:

```bash
python src/pipeline.py mode-b --smoke-3dgs --preset footrest_sand --no-debug-selection
```

For a faster physics smoke test, use `--smoke` instead of `--smoke-3dgs`:

```bash
python src/pipeline.py mode-b --smoke --preset footrest_sand --no-reuse-selection
```

Useful Mode B flags:

- `--smoke-3dgs`: use the low-iteration 3DGS checkpoint but keep full physics settings.
- `--smoke`: use the low-iteration 3DGS checkpoint and smaller/shorter Mode B MPM settings.
- `--no-reuse-selection`: recompute the text-to-mask-to-Gaussian selection.
- `--force-rerender-views`: regenerate RGB/depth/camera metadata used for selection.
- `--taichi-memory-gb 4`: override PhysGaussian Taichi memory allocation if needed.
- `--allow-cpu-fallback`: retry segmentation on CPU if CUDA segmentation fails; this is much slower.

Expected Mode B outputs:

- `output/renders_views/`: RGB/depth/camera metadata used for segmentation.
- `output/sim_results/mode_b_jelly/video/output.mp4`: final video for `desk_jelly`.
- `output/sim_results/mode_b_sand/video/output.mp4`: final video for `footrest_sand`.
- `output/sim_results/<mode_b_output>/phys_config.json`: generated PhysGaussian config.
- `output/sim_results/<mode_b_output>/selected_indices.npy`: final selected Gaussian indices.
- `output/sim_results/<mode_b_output>/surface_gaussian_indices.npy`: cached surface selection.
- `output/sim_results/<mode_b_output>/selection_trace.json`: selection audit trail.
- `output/sim_results/<mode_b_output>/camera_usage.txt`: render/selection camera audit.
- `output/sim_results/<mode_b_output>/debug/` and `debug_selection/`: optional selection diagnostics.

## Tests

```bash
cd /home/bgh1225/CG_final/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

pytest tests/ -q
```

## Troubleshooting

- Empty submodule folders: run `git submodule update --init --recursive` from `/home/bgh1225/CG_final`.
- Missing base-scene checkpoint: run `python src/pipeline.py prepare-dataset`, then `python src/pipeline.py train --smoke`. The commands above expect `output/base_scene_smoke/point_cloud/iteration_8000/point_cloud.ply`.
- Missing Grounding DINO or SAM 2 weights: run `bash scripts/download_checkpoints.sh` from `text-guided-3d-editor/`.
- `ImportError: libc10.so` while using GroundingDINO CUDA ops: run:

```bash
export LD_LIBRARY_PATH="$(python -c 'import torch, pathlib; print((pathlib.Path(torch.__file__).resolve().parent/"lib").as_posix())'):$LD_LIBRARY_PATH"
```

- `nvdiffrast` build failures: install PyTorch first, then run `pip install -r requirements.txt --no-build-isolation`.
- CUDA or Taichi out-of-memory: use `--smoke` for a smaller physics run, or pass `--taichi-memory-gb 4` to Mode B.
- Mode B selects the wrong object or reuses stale indices: rerun with `--no-reuse-selection --force-rerender-views`, or remove the relevant folder under `output/sim_results/`.
- Grounded SAM 2 returns a fallback rectangle or no mask: check that both checkpoint files exist, verify CUDA is available, and try a more specific preset/prompt. `--allow-cpu-fallback` can help debug but is slow.

## Git Submission Notes

Pushed to git:

- This repository's source code and config files.
- Fixed submodule commit pointers in `.gitmodules`.

Not pushed to git:

- `text-guided-3d-editor/output/`
- `text-guided-3d-editor/data/`
- Grounding DINO and SAM 2 checkpoint files under `text-guided-3d-editor/submodules/Grounded-SAM-2/`
- local caches and generated artifacts listed in `.gitignore`

You do not need push access to the upstream submodule repositories to use this project. `git submodule update --init --recursive` only downloads the pinned commits.
