### < Kinematic wobble >
# duck이 그냥 정지해있음. - 이거를 metal로 하는게 좋을 듯.
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material metal --in-place-wobble --no-auto-realism 

# duck이 jelly처럼 꿈틀거리기는 하는데, 부자연스러움. (output_duck5.mp4)
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material jelly --in-place-wobble --no-auto-realism \
  --wobble-amp 0.10 --wobble-frequency 1.0 --wobble-height-weight 2.0 --wobble-bottom-pin 0.28

# duck이 이상하게 꿈틀거림 (output_duck9.mp4) - 이거를 jelly로 하는게 좋을 듯.
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material jelly --in-place-wobble --no-auto-realism \
  --wobble-amp 0.095 --wobble-frequency 0.65 --wobble-height-weight 0.25 --wobble-bottom-pin 0.15
# ============================================================================================
### < PhysGaussian > — MPM (Material Point Method)
# MPM을 이용해서 시뮬레이션 하는 방식이라고 생각하면 됨.
# < metal > - 값을 너무 단단하게 하면 자꾸 터져서, 포기함.
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material metal --no-auto-realism

# < Sand >
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material sand --no-auto-realism

# < Foam >
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material foam --no-auto-realism

# < Jelly >
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material jelly --no-auto-realism