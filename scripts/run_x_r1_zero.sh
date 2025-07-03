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


# 3B 
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/X_R1_zero_3B_config.yaml \
> ./output/x_r1_3B_sampling.log 2>&1


# 3B LoRA微调配置 (使用vLLM - 优化版)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_usevllm_config.yaml \
> ./output/x_r1_3b_lora_sampling.log 2>&1

# 3B LoRA微调配置-改进奖励函数 (使用vLLM - 优化版)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_usevllm_config_advanced.yaml \
> ./output/x_r1_3b_lora_advanced_sampling_bespokelabs.log 2>&1

# 3B LoRA微调配置-改进奖励函数 (使用vLLM - 快速版)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_usevllm_config_advanced_fast.yaml \
> ./output/x_r1_3b_lora_advanced_fast_sampling_bespokelabs.log 2>&1

# 3B LoRA微调配置 (不使用vLLM - 更低内存)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_novllm_config.yaml \
> ./output/x_r1_3b_lora_novllm_sampling.log 2>&1

# 7B LoRA微调配置 (使用vLLM)
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \ 
--config recipes/examples/X_R1_zero_7B_peft_usevllm_config.yaml \
> ./output/test_7b_lora_sampling.log 2>&1
