# X-R1 2GPU版本训练脚本
# 2GPU LoRA微调配置 (使用vLLM - 优化版)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero2.yaml \
--num_processes=2 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_usevllm_config_2gpu.yaml \
> ./output/x_r1_3b_lora_2gpu_sampling.log 2>&1