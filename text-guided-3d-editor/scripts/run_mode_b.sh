# selection을 처음부터 다시 함 (오래걸림)
python src/pipeline.py mode-b "wooden desk" --smoke --no-reuse-selection --kinematic-wobble

# < kinematic-wobble >
# PhysicsGaussian MPM이 아니라, Gaussian 위치를 직접 흔드는 방식임.
# selection을 하지 않고, 기존거를 이용함 (좀 더 빠름)
python src/pipeline.py mode-b "wooden table in the foreground." \
  --smoke --kinematic-wobble

# 모든 것을 full로 하고 싶으면?
conda activate cg_final
cd text-guided-3d-editor
export PYTHONPATH=src
python src/pipeline.py mode-b "wooden table in the foreground" --no-debug-selection

# 3DGS만 smoke, 물리/영상은 full로 하고 싶으면? (Mode-b "wooden desk in the forefront" – Jelly Physics)
python src/pipeline.py mode-b --smoke-3dgs --no-debug-selection --preset desk_jelly
# 내부적으로 자동으로 아래를 사용함.
# prompt = "wooden desk in the forefront"
# physics_type = "jelly"
# output = output/sim_results/mode_b_jelly


# 3DGS만 smoke, 물리/영상은 full로 하고 싶으면? (Mode-b "blue armchair in the back center" – Sand Physics)
python src/pipeline.py mode-b --smoke-3dgs --preset footrest_sand --no-debug-selection
python src/pipeline.py mode-b --smoke-3dgs --preset footrest_sand
python src/pipeline.py mode-b --smoke-3dgs --preset footrest_sand --no-reuse-selection
python src/pipeline.py mode-b --smoke-3dgs --preset footrest_sand --reuse-selection --no-debug-selection
# 내부적으로 자동으로 아래를 사용함.
# prompt = "blue armchair in the back center"
# physics_type = "sand"
# output = output/sim_results/mode_b_sand

# ============================================================================================
