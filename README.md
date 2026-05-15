# Text-guided 3D indoor editor

This repository contains a text-guided indoor 3D editing pipeline for COMS3168 Deep Learning for Computer Graphics.

---

| <mark>Start here</mark> — **code structure** |
| :--- |
| For a <mark>**directory tree**</mark> and short **notes on what each major folder does**, <mark>open</mark> **[about: Code structure on Notion](https://regal-tomato-67b.notion.site/about-Code-structure-35f183b9bcdc800e8d4fd3a7ea8e0d76?source=copy_link)**. |

<table>
<tbody>
<tr><td><strong>Final report</strong> – <a href="https://drive.google.com/file/d/1mRwYNZVW6xtlWn1rrQ6t6Z-GqhjFny_B/view?usp=sharing">Google Drive link</a></td></tr>
<tr><td><strong>Project poster</strong> - <a href="https://drive.google.com/file/d/1_3FS-H2I-KMU_D1sA2VV95s_OgfnRLd6/view?usp=sharing">Google Drive link</a></td></tr>
<tr><td><strong>Demo video</strong> — <a href="https://drive.google.com/drive/folders/1xkxrKRifPMdOvlIiuX7t776YjjkO9d3m?usp=sharing">Google Drive link</a></td></tr>
</tbody>
</table>

---

**Pipeline**:

```text
Mip-NeRF 360 room data + COLMAP -> 3D Gaussian Splatting base scene
-> Grounded SAM 2 object selection
-> DreamGaussian object generation / insertion
-> PhysGaussian / kinematic simulation and video rendering
```

**Mode A** inserts a generated or prebuilt object into the room scene and simulates only that inserted object. <br>
**Mode B** selects an existing object in the room from text and applies material-specific physics to the selected Gaussians. <br>

> [!IMPORTANT]
> **Shell paths:** All `cd …`, file, and `--object-gaussians` examples assume the code lives at **`/home/bgh1225/text-guided-3d-editor/`**. If your clone is elsewhere (for example **`…/CG_final/text-guided-3d-editor/`**), replace that prefix with your real absolute path before pasting commands.

Copy-paste commands for **Mode A** and **Mode B** are in **[Run Mode A](#run-mode-a)** and **[Run Mode B](#run-mode-b)** below (after clone, environment, and checkpoints). **Metal** / **jelly** there use **in-place kinematic wobble**; **sand** / **foam** use **full PhysGaussian MPM**. The same examples are also collected in `scripts/run_mode_a.sh` and `scripts/run_mode_b.sh`.

## Project Overview

<table>
  <tr>
    <td align="center"><img src="/home/bgh1225/CG_final/assets/1.png" width="280" alt="3DGS-rendered room scene" /></td>
    <td align="center"><img src="/home/bgh1225/CG_final/assets/2.png" width="280" alt="Clean editing baseline" /></td>
    <td align="center"><img src="/home/bgh1225/CG_final/assets/3.png" width="280" alt="DreamGaussian duck for Mode A" /></td>
  </tr>
  <tr>
    <td align="center"><img src="/home/bgh1225/CG_final/assets/4.png" width="280" alt="Duck merged into base scene (Mode A)" /></td>
    <td align="center"><img src="/home/bgh1225/CG_final/assets/5.png" width="280" alt="Grounded SAM 2 selection on desk (Mode B)" /></td>
    <td align="center"><img src="/home/bgh1225/CG_final/assets/6.png" width="280" alt="Grounded SAM 2 selection on armchair (Mode B)" /></td>
  </tr>
</table>

<strong>Text-Guided 3D Indoor Scene Editing.</strong> A single 3DGS reconstruction of the Mip-NeRF 360 room supports two language-driven editing modes. <br>
Top: the 3DGS-rendered scene, the same scene as a clean editing baseline, and a DreamGaussian-generated duck for Mode A insertion. <br>
Bottom: the generated duck merged into the base scene (Mode A), and Grounded SAM 2-driven Gaussian selection on the desk and on the armchair (Mode B) before physics is applied.

## Clone Repository (with Submodules)

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

- `/home/bgh1225/text-guided-3d-editor/submodules/gaussian-splatting/`
- `/home/bgh1225/text-guided-3d-editor/submodules/dreamgaussian/`
- `/home/bgh1225/text-guided-3d-editor/submodules/Grounded-SAM-2/`
- `/home/bgh1225/text-guided-3d-editor/submodules/PhysGaussian/`

## Environment Setup

Use the `cg_final` conda environment.

```bash
conda create -n cg_final python=3.10 -y
conda activate cg_final

python -m pip install --upgrade pip setuptools wheel ninja
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

cd /home/bgh1225/text-guided-3d-editor   # implementation directory (see Shell paths callout under **Pipeline**)
pip install -r requirements.txt --no-build-isolation
bash scripts/build_cuda_extensions.sh
```

`setup.sh` is also available if you want the helper script to run submodule setup and project requirements after activating the environment:

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
bash setup.sh
```

The manual commands above are the recommended submission setup because they explicitly use the `cg_final` environment and build the CUDA extensions without editing submodules.

Install additional upstream dependencies as needed from the pinned upstream READMEs (same four submodules as **Clone Repository** above):

- `/home/bgh1225/text-guided-3d-editor/submodules/gaussian-splatting/` — 3DGS training/rendering; CUDA rasterizer extensions are built by `bash scripts/build_cuda_extensions.sh` above; see this README if `train` or rasterizer setup fails.
- `/home/bgh1225/text-guided-3d-editor/submodules/dreamgaussian/`
- `/home/bgh1225/text-guided-3d-editor/submodules/Grounded-SAM-2/`
- `/home/bgh1225/text-guided-3d-editor/submodules/PhysGaussian/`

Before running the pipeline, set `PYTHONPATH` from the implementation directory:

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src
```

## Pre-trained base scenes (recommended; Google Drive `Pretrained_3DGS/`)

This is the **default path for instructor and TAs**: it avoids a long 3DGS `train` run while still matching the Mode A / Mode B commands in this README (with the correct `--smoke` / `--smoke-3dgs` flags).

Training `train` from scratch is slow, so we ship **pre-built base checkpoints** outside Git.

1. Open the shared Google Drive folder [**`Pretrained_3DGS/`**](https://drive.google.com/drive/folders/126R-A_W69BE_vLjMZnl3jVNGyEo2m5GB?usp=sharing) and download both archives:

   - `base_scene_smoke.tar.gz` — low-iteration run (`iteration_8000`), matches the `--smoke` / `--smoke-3dgs` examples below.
   - `base_scene_full.tar.gz` — full run (`iteration_30000`), matches non-smoke `train` and commands without `--smoke`.

2. Place the archives under `/home/bgh1225/text-guided-3d-editor/output/` (create that folder if needed), then extract:

```bash
cd /home/bgh1225/text-guided-3d-editor/output
tar -xzf base_scene_smoke.tar.gz
tar -xzf base_scene_full.tar.gz
```

3. After extraction you should have:

   - `/home/bgh1225/text-guided-3d-editor/output/base_scene_smoke/point_cloud/iteration_8000/point_cloud.ply` (and the rest of that run directory).
   - `/home/bgh1225/text-guided-3d-editor/output/base_scene_full/point_cloud/iteration_30000/point_cloud.ply` (and the rest of that run directory).

4. **Skip `python src/pipeline.py train ...`** for the bundle you installed. When running Mode A / Mode B, keep flags consistent: use `--smoke` / `--smoke-3dgs` only if you rely on **`base_scene_smoke`**; omit smoke flags if you rely only on **`base_scene_full`**.

**Still required:** `prepare-dataset` (or an existing `room` scene at your config’s `data_root`) if you need `images/` and COLMAP `sparse/` for view rendering or Mode B selection; `bash scripts/download_checkpoints.sh` for Grounding DINO + SAM 2. The Drive archives replace **only** the long **3DGS `train`** step, not data or segmentation weights.

**Train from scratch instead:** If you want to **run 3DGS `train` yourself** (not only the pre-trained checkpoints), prepare the Mip-NeRF 360 bundle in **Data and Checkpoints** (official download or the optional Drive zip below), then follow **Train Base 3DGS Scene**.

## Data and Checkpoints

Mode B view rendering and segmentation need the **`room`** scene under `data/mipnerf360/room/` (`images/`, COLMAP `sparse/0/`). You can get there in either of two ways.

### Option A — official download (default)

`prepare-dataset` fetches **`360_v2.zip`** from the URL in `configs/pipeline_config.yaml` (Google Research Mip-NeRF 360 bundle), extracts it, and stages **`room`** for training.

### Option B — Google Drive zip (skip the long HTTP download)

To save download time (or if the official host is slow from your network), use the same archive we host on the course Google Drive:

1. Open the shared Drive folder **`Data/mipnerf360_raw/`** (link in the final submission / course materials) and download **`360_v2.zip`**.
2. Place it at **`/home/bgh1225/text-guided-3d-editor/data/mipnerf360_raw/360_v2.zip`** (create `data/mipnerf360_raw/` if needed).
3. Run **`prepare-dataset`** as below — the pipeline **does not re-download** when that file already exists; it only extracts and copies **`room`** into `data/mipnerf360/`.
4. Continue with **Train Base 3DGS Scene** if you are training a base scene from scratch (not using **Pre-trained base scenes** only).

### Prepare the `room` scene (either option)

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py prepare-dataset
```

Expected layout after `prepare-dataset`:

- `/home/bgh1225/text-guided-3d-editor/data/mipnerf360/room/images/`
- `/home/bgh1225/text-guided-3d-editor/data/mipnerf360/room/sparse/0/`

Grounding DINO and SAM 2 weights are not committed to git. Download them into the paths expected by `configs/pipeline_config.yaml`:

```bash
cd /home/bgh1225/text-guided-3d-editor
bash scripts/download_checkpoints.sh
```

Expected checkpoint files:

- `/home/bgh1225/text-guided-3d-editor/submodules/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth`
- `/home/bgh1225/text-guided-3d-editor/submodules/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt`

## Train Base 3DGS Scene

Use this section only if you are **not** using the pre-trained `/home/bgh1225/text-guided-3d-editor/output/base_scene_*` trees from **Pre-trained base scenes** above.

The Mode A and Mode B commands below use the smoke 3DGS checkpoint, which is faster and writes iteration `8000`.

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py train --smoke
```

Expected base checkpoint:

- `/home/bgh1225/text-guided-3d-editor/output/base_scene_smoke/point_cloud/iteration_8000/point_cloud.ply`

For a full 3DGS training run:

```bash
python src/pipeline.py train
```

Expected full checkpoint:

- `/home/bgh1225/text-guided-3d-editor/output/base_scene_full/point_cloud/iteration_30000/point_cloud.ply`

Use the same smoke/full choice consistently when running later commands.

## Pre-built Mode A object (Google Drive `Generated_objects/`)

The **Run Mode A** examples below reuse a pre-generated yellow duck Gaussian PLY via **`--object-gaussians`**. That asset lives under `output/generated_objects/` and is **not** committed to git. To match the README commands without running DreamGaussian first, use our Drive bundle:

1. Open the shared Google Drive folder [**`Generated_objects/`**](https://drive.google.com/drive/folders/1GURei8bgnKKj-pTkB_UX_f8sQ8BzHsJF?usp=sharing) and download **`generated_objects.tar.gz`** (required for the `--object-gaussians` examples below). The same folder also has optional **`merged_scenes.tar.gz`** — see **Optional: pre-merged Mode A scene** below.
2. Place the archive under `/home/bgh1225/text-guided-3d-editor/output/` (create `output/` if needed), then extract:

```bash
cd /home/bgh1225/text-guided-3d-editor/output
tar -xzf generated_objects.tar.gz
```

3. After extraction you should have at least:

   - `/home/bgh1225/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply`

4. Continue with **Run Mode A** below — the `--object-gaussians` paths already match this layout.

**Not required for Mode B.** **Alternative (slower):** omit `--object-gaussians` and let the pipeline generate the object (see **Other example (jelly)** under **Run Mode A**). You still need **Pre-trained base scenes** (or `train`) for the room checkpoint.

## Optional: pre-merged Mode A scene (`merged_scenes.tar.gz`)

The same Google Drive folder [**`Generated_objects/`**](https://drive.google.com/drive/folders/1GURei8bgnKKj-pTkB_UX_f8sQ8BzHsJF?usp=sharing) also hosts **`merged_scenes.tar.gz`** — a snapshot of `output/merged_scenes/` (room + inserted duck merge and PhysGaussian staging files). **Not required:** a normal **Run Mode A** command rebuilds `mode_a_merged.ply` each time. Use this bundle if you want the exact merged tree without re-running the merge step, or for inspection.

1. From that Drive folder, download **`merged_scenes.tar.gz`**.
2. Place the archive under `/home/bgh1225/text-guided-3d-editor/output/` (create `output/` if needed), then extract:

```bash
cd /home/bgh1225/text-guided-3d-editor/output
tar -xzf merged_scenes.tar.gz
```

3. After extraction you should have paths such as:

   - `/home/bgh1225/text-guided-3d-editor/output/merged_scenes/mode_a_merged.ply`
   - `/home/bgh1225/text-guided-3d-editor/output/merged_scenes/mode_a_phys_input/` (MPM staging; includes `point_cloud/iteration_0/point_cloud.ply` and `cameras.json` when present in the archive)

4. Then run **Run Mode A** below (with **Pre-built Mode A object** / `--object-gaussians` as needed). A new Mode A run may **overwrite** files under `merged_scenes/` when it merges again.

## Run Mode A

Install **Pre-built Mode A object** above before copy-pasting the `--object-gaussians` examples (unless you plan to generate the duck yourself).

### How to run `mode-a` (metal)

Same as `scripts/run_mode_a.sh` **lines 3–5**. Output: **`/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_metal/video/output.mp4`** (kinematic wobble, not full MPM).

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material metal --in-place-wobble --no-auto-realism
```

**Example output (metal):** `text-guided-3d-editor/output/sim_results/mode_a_metal/video/demo.gif` (local path after a successful run; `output/` is not in git).

<img src="text-guided-3d-editor/output/sim_results/mode_a_metal/video/demo.gif" width="400" alt="Mode A metal demo" />

### How to run `mode-a` (jelly)

Same as `scripts/run_mode_a.sh` **lines 14–17**. Output: **`/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_jelly/video/output.mp4`** (kinematic wobble, not full MPM).

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material jelly --in-place-wobble --no-auto-realism \
  --wobble-amp 0.095 --wobble-frequency 0.65 --wobble-height-weight 0.25 --wobble-bottom-pin 0.15
```

**Example output (jelly):** `text-guided-3d-editor/output/sim_results/mode_a_jelly/video/demo.gif` (local path after a successful run; `output/` is not in git).

<img src="text-guided-3d-editor/output/sim_results/mode_a_jelly/video/demo.gif" width="400" alt="Mode A jelly demo" />

### How to run `mode-a` (sand)

Same as `scripts/run_mode_a.sh` **lines 27–29**. Output directory: **`/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_sand/`** (PhysGaussian MPM; main clip under **`video/output.mp4`** in typical runs).

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material sand --no-auto-realism
```

**Example output (sand):** `text-guided-3d-editor/output/sim_results/mode_a_sand/video/demo.gif` (local path after a successful run; `output/` is not in git).

<img src="text-guided-3d-editor/output/sim_results/mode_a_sand/video/demo.gif" width="400" alt="Mode A sand demo" />

### How to run `mode-a` (foam)

Same as `scripts/run_mode_a.sh` **lines 32–34**. Output directory: **`/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_foam/`** (PhysGaussian MPM; main clip under **`video/output.mp4`** in typical runs).

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material foam --no-auto-realism
```

**Example output (foam):** `text-guided-3d-editor/output/sim_results/mode_a_foam/video/demo.gif` (local path after a successful run; `output/` is not in git).

<img src="text-guided-3d-editor/output/sim_results/mode_a_foam/video/demo.gif" width="400" alt="Mode A foam demo" />

### Expected outputs (Mode A)

Paths below are under `/home/bgh1225/text-guided-3d-editor/` (or use the same layout relative to your clone). The run directory name is `mode_a_<material>`. Rerunning the same material overwrites that run folder (for example **`mode_a_jelly/`** for the jelly examples above and in **Other example (jelly)**).

- `/home/bgh1225/text-guided-3d-editor/output/generated_objects/sds_run/`: staged assets when reusing `--object-gaussians`, or intermediate outputs when the pipeline generates the object.
- `/home/bgh1225/text-guided-3d-editor/output/generated_objects/sds_run/object_scaled.ply`: object after scene-scale conversion (when the pipeline generates or rescales the object).
- `/home/bgh1225/text-guided-3d-editor/output/merged_scenes/mode_a_merged.ply`: base scene plus inserted object.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_metal/video/output.mp4`: **metal** + in-place wobble (`run_mode_a.sh` lines 3–5).
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_jelly/video/output.mp4`: **jelly** + in-place wobble (`run_mode_a.sh` lines 14–17); wobble runs also write **`/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_jelly/frames/`**. If you later run **full PhysGaussian MPM** for jelly, the same **`mode_a_jelly`** tree may also contain `phys_mode_a.json`, `mode_a_obj_indices.npy`, etc.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_sand/video/output.mp4`: **sand** + MPM (`run_mode_a.sh` lines 27–29); you may also see companion artifacts under `mode_a_sand/` such as `phys_mode_a.json` and `mode_a_obj_indices.npy`.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_a_foam/video/output.mp4`: **foam** + MPM (`run_mode_a.sh` lines 32–34); same pattern under `mode_a_foam/`.


## Run Mode B

Mode B selects an existing object in the room with Grounded SAM 2, projects the mask back to 3D Gaussians, and simulates the selected set. The current CLI requires `--preset`.

### How to run `mode-b` (desk jelly)

This runs the **`desk_jelly`** preset (wooden desk in the forefront, jelly-style physics / kinematic path per config):

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

python src/pipeline.py mode-b --smoke-3dgs --preset desk_jelly --no-debug-selection
```

**Example output (desk jelly):** `text-guided-3d-editor/output/sim_results/mode_b_desk_jelly/video/demo.gif` (local path after a successful run; `output/` is not in git).

<img src="text-guided-3d-editor/output/sim_results/mode_b_desk_jelly/video/demo.gif" width="400" alt="Mode B desk jelly demo" />

**Output directory:** `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_b_desk_jelly/` (main clip is typically **`video/output.mp4`** under that folder).

### How to run `mode-b` (armchair jelly)

Same `cd` / `conda activate` / `export PYTHONPATH=src` as **desk jelly** above, then:

```bash
python src/pipeline.py mode-b --smoke-3dgs --preset armchair_jelly --no-debug-selection
```

**Example output (armchair jelly):** `text-guided-3d-editor/output/sim_results/mode_b_armchair_jelly/video/demo.gif` (local path after a successful run; `output/` is not in git).

<img src="text-guided-3d-editor/output/sim_results/mode_b_armchair_jelly/video/demo.gif" width="400" alt="Mode B armchair jelly demo" />

**Output directory:** `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_b_armchair_jelly/` (main clip is typically **`video/output.mp4`** under that folder).

### Other presets (sand)

From the same working directory with `conda activate cg_final` and `export PYTHONPATH=src` as above:

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

- `/home/bgh1225/text-guided-3d-editor/output/renders_views/`: RGB/depth/camera metadata used for segmentation.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_b_desk_jelly/video/output.mp4`: final video for **`desk_jelly`** (`--preset desk_jelly`).
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_b_armchair_jelly/video/output.mp4`: final video for **`armchair_jelly`** (`--preset armchair_jelly`).
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/mode_b_sand/video/output.mp4`: final video for `footrest_sand` (preset-dependent folder name; see `configs/pipeline_config.yaml`).
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/<mode_b_output>/phys_config.json`: generated PhysGaussian config.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/<mode_b_output>/selected_indices.npy`: final selected Gaussian indices.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/<mode_b_output>/surface_gaussian_indices.npy`: cached surface selection.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/<mode_b_output>/selection_trace.json`: selection audit trail.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/<mode_b_output>/camera_usage.txt`: render/selection camera audit.
- `/home/bgh1225/text-guided-3d-editor/output/sim_results/<mode_b_output>/debug/` and `debug_selection/`: optional selection diagnostics.

## Tests

```bash
cd /home/bgh1225/text-guided-3d-editor
conda activate cg_final
export PYTHONPATH=src

pytest tests/ -q
```

## Troubleshooting

- Empty submodule folders: run `git submodule update --init --recursive` from your repository root (the directory that contains `text-guided-3d-editor/`).
- Missing base-scene checkpoint: either extract **`Pretrained_3DGS/`** archives into `/home/bgh1225/text-guided-3d-editor/output/` (see **Pre-trained base scenes** above), or run `python src/pipeline.py prepare-dataset`, then `python src/pipeline.py train --smoke`. Smoke examples expect `/home/bgh1225/text-guided-3d-editor/output/base_scene_smoke/point_cloud/iteration_8000/point_cloud.ply`.
- Slow or failed Mip-NeRF download: place **`360_v2.zip`** from Drive **`Data/mipnerf360_raw/`** at `/home/bgh1225/text-guided-3d-editor/data/mipnerf360_raw/360_v2.zip`, then rerun `prepare-dataset` (see **Data and Checkpoints** — Option B).
- Mode A fails on **`--object-gaussians`**: download **`generated_objects.tar.gz`** from Drive [**`Generated_objects/`**](https://drive.google.com/drive/folders/1GURei8bgnKKj-pTkB_UX_f8sQ8BzHsJF?usp=sharing), extract under `/home/bgh1225/text-guided-3d-editor/output/` (see **Pre-built Mode A object** above), or rerun without `--object-gaussians` to generate the duck.
- Missing Grounding DINO or SAM 2 weights: run `bash scripts/download_checkpoints.sh` from `/home/bgh1225/text-guided-3d-editor/`.
- `ImportError: libc10.so` while using GroundingDINO CUDA ops: run:

```bash
export LD_LIBRARY_PATH="$(python -c 'import torch, pathlib; print((pathlib.Path(torch.__file__).resolve().parent/"lib").as_posix())'):$LD_LIBRARY_PATH"
```

- `nvdiffrast` build failures: install PyTorch first, then run `pip install -r requirements.txt --no-build-isolation`.
- CUDA or Taichi out-of-memory: use `--smoke` for a smaller physics run, or pass `--taichi-memory-gb 4` to Mode B.
- Mode B selects the wrong object or reuses stale indices: rerun with `--no-reuse-selection --force-rerender-views`, or remove the relevant folder under `/home/bgh1225/text-guided-3d-editor/output/sim_results/`.
- Grounded SAM 2 returns a fallback rectangle or no mask: check that both checkpoint files exist, verify CUDA is available, and try a more specific preset/prompt. `--allow-cpu-fallback` can help debug but is slow.

## Git Submission Notes

Pushed to git:

- This repository's source code and config files.
- Fixed submodule commit pointers in `.gitmodules`.

Not pushed to git:

- `/home/bgh1225/text-guided-3d-editor/output/`
- `/home/bgh1225/text-guided-3d-editor/data/`
- Grounding DINO and SAM 2 checkpoint files under `/home/bgh1225/text-guided-3d-editor/submodules/Grounded-SAM-2/`
- local caches and generated artifacts listed in `.gitignore`
- Pre-trained 3DGS base runs (`base_scene_smoke.tar.gz`, `base_scene_full.tar.gz`): distributed separately in Google Drive folder **`Pretrained_3DGS/`** (see **Pre-trained base scenes** above).
- Mip-NeRF 360 bundle (`360_v2.zip`): optional mirror in Google Drive folder **`Data/mipnerf360_raw/`** (see **Data and Checkpoints** — Option B).
- Pre-built Mode A duck assets (`generated_objects.tar.gz` → `output/generated_objects/sds_run/`): Google Drive folder [**`Generated_objects/`**](https://drive.google.com/drive/folders/1GURei8bgnKKj-pTkB_UX_f8sQ8BzHsJF?usp=sharing) (see **Pre-built Mode A object** above).
- Optional merged Mode A scene (`merged_scenes.tar.gz` → `output/merged_scenes/`): same Drive folder (see **Optional: pre-merged Mode A scene** above).

You do not need push access to the upstream submodule repositories to use this project. `git submodule update --init --recursive` only downloads the pinned commits.
