# 使用2个进程的X-R1训练脚本 (配合num_generations=2)

# 3B LoRA微调配置 (使用vLLM - 2进程版)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_usevllm_2gen_config.yaml \
> ./output/x_r1_3b_lora_2proc_sampling.log 2>&1


# 3B LoRA微调配置 (不使用vLLM - 2进程版) [推荐]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=2 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_novllm_2gen_config.yaml \
> ./output/x_r1_3b_lora_novllm_2proc_sampling.log 2>&1


# 3B LoRA微调配置 (极致内存优化 - 1进程版)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_ultra_low_memory.yaml \
> ./output/x_r1_3b_ultra_low_memory.log 2>&1


# 3B LoRA微调配置 (4GPU高性能版)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_4gpu_config.yaml \
> ./output/x_r1_3b_4gpu.log 2>&1 