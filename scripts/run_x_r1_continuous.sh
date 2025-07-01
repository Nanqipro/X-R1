#!/bin/bash

# X-R1 训练脚本 - 使用改进的连续奖励函数
# 预期效果：显著减少0奖励，增加训练信号密度

echo "=========================================="
echo "开始X-R1训练 - 连续奖励函数版本"
echo "配置: accuracy_continuous + format_continuous"
echo "预期改进: 奖励密度从25%提升到80%+"
echo "=========================================="

# 确保输出目录存在
mkdir -p ./output

# 设置CUDA内存配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动训练
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 src/x_r1/grpo.py \
--config recipes/examples/X_R1_zero_3B_peft_usevllm_config.yaml \
> ./output/x_r1_3b_lora_continuous.log 2>&1

echo "=========================================="
echo "训练完成！日志文件: ./output/x_r1_3b_lora_continuous.log"
echo "检查奖励改进："
echo "  grep 'accuracy rewards (continuous)' ./output/x_r1_3b_lora_continuous.log | tail -5"
echo "  grep 'format rewards (continuous)' ./output/x_r1_3b_lora_continuous.log | tail -5"
echo "==========================================" 