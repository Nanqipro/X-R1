#!/bin/bash

# X-R1 3B LoRA微调配置-稳定版本
# 优化内存使用和分布式训练稳定性

echo "开始X-R1 3B LoRA稳定训练..."

# 设置环境变量优化内存和通信
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=7200000  # 增加NCCL超时时间到2小时
export NCCL_IB_DISABLE=1     # 禁用InfiniBand，使用以太网
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# 清理GPU内存
echo "清理GPU内存..."
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print('GPU内存已清理')
"

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi

# 启动训练
echo "启动训练..."
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_usevllm_config_advanced.yaml \
> ./output/x_r1_3b_lora_advanced_sampling_generated_x_r1_dataset02_stable.log 2>&1

echo "训练完成，检查日志: ./output/x_r1_3b_lora_advanced_sampling_generated_x_r1_dataset02_stable.log" 