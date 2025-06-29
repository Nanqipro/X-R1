
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_0dot5B_config.yaml \
> ./output/x_r1_0dot5B_sampling.log 2>&1



ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_1dot5B_config.yaml \
> ./output/x_r1_1dot5B_sampling.log 2>&1



PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/grpo.py \
--config recipes/X_R1_zero_3B_config.yaml \
> ./output/x_r1_3B_sampling.log 2>&1


# 如果上面的配置仍然显存不足，使用极端低内存版本：
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACCELERATE_LOG_LEVEL=info accelerate launch \
# --config_file recipes/zero3.yaml \
# --num_processes=2 src/x_r1/grpo.py \
# --config recipes/X_R1_zero_3B_config_lowmem.yaml \
# > ./output/x_r1_3B_lowmem_sampling.log 2>&1
