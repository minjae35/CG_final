# Recompute selection from scratch (slow).
python src/pipeline.py mode-b "wooden desk" --smoke --no-reuse-selection --kinematic-wobble

# < kinematic-wobble >
# Not PhysGaussian MPM; wobbles Gaussian positions directly.
# Skips redoing selection and reuses cached selection (faster).
python src/pipeline.py mode-b "wooden table in the foreground." \
  --smoke --kinematic-wobble

# Want everything at full quality?
conda activate cg_final
cd text-guided-3d-editor
export PYTHONPATH=src
python src/pipeline.py mode-b "wooden table in the foreground" --no-debug-selection

# ===================================================================================================
### < preset >
### < wooden desk in the forefront > - Kinematic wobble 
# Only 3DGS smoke, but physics/video at full settings?
python src/pipeline.py mode-b --smoke-3dgs --preset desk_jelly --no-debug-selection
: <<'TAG'
Internally the preset uses the following.
prompt = "wooden desk in the forefront"
physics_type = "jelly"
output = output/sim_results/mode_b_jelly
TAG

### < blue armchair in the back center > - Kinematic wobble
# Only 3DGS smoke, but physics/video at full settings?
python src/pipeline.py mode-b --smoke-3dgs --preset armchair_jelly --no-debug-selection
python src/pipeline.py mode-b --smoke-3dgs --preset armchair_jelly
python src/pipeline.py mode-b --smoke-3dgs --preset armchair_jelly --no-reuse-selection
python src/pipeline.py mode-b --smoke-3dgs --preset armchair_jelly --reuse-selection --no-debug-selection
: <<'TAG'
Internally the preset uses the following.
prompt = "blue armchair in the back center"
physics_type = "jelly"
output = output/sim_results/mode_b_sand
TAG

# ============================================================================================
