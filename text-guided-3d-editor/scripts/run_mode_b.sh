# selection을 처음부터 다시 함 (오래걸림)
python src/pipeline.py mode-b "wooden desk" --smoke --no-reuse-selection --kinematic-wobble


# < kinematic-wobble >
# PhysicsGaussian MPM이 아니라, Gaussian 위치를 직접 흔드는 방식임.
# selection을 하지 않고, 기존거를 이용함 (좀 더 빠름)
python src/pipeline.py mode-b "wooden table in the foreground." \
  --smoke --kinematic-wobble
