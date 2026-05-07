# duck이 그냥 정지해있음.
python src/pipeline.py mode-a "duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material rubber --in-place-wobble --no-auto-realism \

# duck이 jelly처럼 꿈틀거리기는 하는데, 부자연스러움. (output_duck5.mp4)
python src/pipeline.py mode-a "duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material rubber --in-place-wobble --no-auto-realism \
  --wobble-amp 0.10 --wobble-frequency 1.0 --wobble-height-weight 2.0 --wobble-bottom-pin 0.28

# duck이 이상하게 꿈틀거림 (output_duck9.mp4)
python src/pipeline.py mode-a "duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material rubber --in-place-wobble --no-auto-realism \
  --wobble-amp 0.095 --wobble-frequency 0.65 --wobble-height-weight 0.25 --wobble-bottom-pin 0.15