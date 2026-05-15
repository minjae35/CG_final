### < Kinematic wobble >
# duck just stays still. - "Use this as the metal result".
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material metal --in-place-wobble --no-auto-realism 

# duck wriggles like "jelly". (output_duck5.mp4)
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material jelly --in-place-wobble --no-auto-realism \
  --wobble-amp 0.10 --wobble-frequency 1.0 --wobble-height-weight 2.0 --wobble-bottom-pin 0.28

# duck wriggles like "jelly" (output_duck9.mp4) - "Use this as the jelly result"
python src/pipeline.py mode-a "yellow rubber duck" --smoke --full-sds \
  --object-gaussians /home/bgh1225/CG_final/text-guided-3d-editor/output/generated_objects/sds_run/object_mesh_gaussians.ply \
  --material jelly --in-place-wobble --no-auto-realism \
  --wobble-amp 0.095 --wobble-frequency 0.65 --wobble-height-weight 0.25 --wobble-bottom-pin 0.15
# ============================================================================================
### < PhysGaussian > — MPM (Material Point Method)
# Think of it as simulating with MPM.
# < metal > 
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