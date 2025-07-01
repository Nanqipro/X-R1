#!/bin/bash

# X-R1 奖励函数对比脚本
# 依次运行三种配置，便于对比效果

echo "==========================================="
echo "X-R1 奖励函数对比测试"
echo "将依次运行三种配置："
echo "1. 原始二元奖励 (accuracy + format)"
echo "2. 基础连续奖励 (accuracy_continuous + format_continuous)"  
echo "3. 高级连续奖励 (+ reasoning_steps)"
echo "==========================================="

# 确保输出目录存在
mkdir -p ./output

# 设置CUDA内存配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 函数：运行训练并分析奖励
run_training() {
    local config_file=$1
    local log_file=$2
    local description=$3
    
    echo ""
    echo "-------------------------------------------"
    echo "开始训练: $description"
    echo "配置文件: $config_file"
    echo "日志文件: $log_file"
    echo "-------------------------------------------"
    
    # 启动训练
    ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/zero3.yaml \
    --num_processes=3 src/x_r1/grpo.py \
    --config $config_file \
    > $log_file 2>&1
    
    echo "训练完成: $description"
    
    # 分析奖励分布
    echo "奖励分析:"
    if grep -q "accuracy rewards:" $log_file; then
        echo "  原始精度奖励样本:"
        grep "accuracy rewards:" $log_file | tail -3
    fi
    
    if grep -q "accuracy rewards (continuous):" $log_file; then
        echo "  连续精度奖励样本:"
        grep "accuracy rewards (continuous):" $log_file | tail -3
    fi
    
    echo "-------------------------------------------"
}

# 1. 运行原始二元奖励版本
run_training \
    "recipes/examples/X_R1_zero_3B_peft_usevllm_config_original.yaml" \
    "./output/x_r1_3b_lora_original.log" \
    "原始二元奖励"

# 2. 运行基础连续奖励版本
run_training \
    "recipes/examples/X_R1_zero_3B_peft_usevllm_config.yaml" \
    "./output/x_r1_3b_lora_continuous.log" \
    "基础连续奖励"

# 3. 运行高级连续奖励版本
run_training \
    "recipes/examples/X_R1_zero_3B_peft_usevllm_config_advanced.yaml" \
    "./output/x_r1_3b_lora_advanced.log" \
    "高级连续奖励"

echo ""
echo "==========================================="
echo "所有训练完成！对比结果分析："
echo "==========================================="

# 生成对比报告
echo ""
echo "📊 奖励密度对比（非零奖励比例）："
echo "原始版本："
if [ -f "./output/x_r1_3b_lora_original.log" ]; then
    grep "accuracy rewards:" "./output/x_r1_3b_lora_original.log" | tail -10 | grep -o "\[.*\]" | tr ',' '\n' | grep -v "0\.0" | wc -l | awk '{print "  非零奖励约 " $1 " 个样本"}'
fi

echo "连续版本："
if [ -f "./output/x_r1_3b_lora_continuous.log" ]; then
    grep "accuracy rewards (continuous):" "./output/x_r1_3b_lora_continuous.log" | tail -10 | grep -o "\[.*\]" | tr ',' '\n' | grep -v "0\.000" | wc -l | awk '{print "  非零奖励约 " $1 " 个样本"}'
fi

echo ""
echo "📈 详细分析："
echo "查看原始奖励: grep 'accuracy rewards:' ./output/x_r1_3b_lora_original.log | tail -5"
echo "查看连续奖励: grep 'accuracy rewards (continuous):' ./output/x_r1_3b_lora_continuous.log | tail -5"
echo "查看高级奖励: grep 'accuracy rewards (continuous):' ./output/x_r1_3b_lora_advanced.log | tail -5"

echo ""
echo "✅ 对比测试完成！"
echo "===========================================" 