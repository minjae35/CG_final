# CG_final

(COMS3168) Deep learning for Computer Graphics — text-guided indoor 3D editing pipeline lives in **`text-guided-3d-editor/`**.

## Clone (submodules required)

Submodules are **not** filled by a plain `git clone`. Use one of the following from this directory:

```bash
git clone --recurse-submodules <YOUR_REPO_URL> CG_final
cd CG_final
```

If you already cloned without submodules:

```bash
cd CG_final
git submodule update --init --recursive
```

Then open **`text-guided-3d-editor/README.md`** for conda, `setup.sh`, dataset download, checkpoints, and how to run `pipeline.py`.

### After `git pull`

If submodule pointers changed:

```bash
git submodule update --init --recursive
```

### Verify submodules

Each submodule directory should contain a real checkout (e.g. `text-guided-3d-editor/submodules/dreamgaussian/main.py` exists). If a folder is empty, you skipped the submodule step above.

## What gets pushed vs local only

- **Pushed:** this repo + fixed submodule **commits** (see `.gitmodules` URLs on GitHub).
- **Not pushed (ignored):** `text-guided-3d-editor/output/`, local `data/`, large caches — see root `.gitignore`.
- **Weights you must download:** Grounding DINO + SAM2 paths in `configs/pipeline_config.yaml` (see Grounded-SAM-2 upstream README). Without those files, segmentation steps will fail.

## Submodule writes

You do **not** need push access to the upstream submodule repos to **use** this project: `git submodule update` only **downloads** the pinned commits. You only need your own fork + URL changes if you **modify** submodule code and want those commits on GitHub.
