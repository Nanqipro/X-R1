#!/bin/bash

# ====================================
# X-R1 3B模型 LoRA训练启动脚本
# ====================================
#
# 功能说明:
#   使用LoRA技术训练Qwen2.5-3B模型，大幅降低硬件要求和训练成本
#
# 硬件要求:
#   - GPU: 4×RTX 3090/4090 (24GB each) - 1个vLLM + 3个训练
#   - 显存: ~72GB (vs 全量微调需要96GB)
#   - 训练时间: ~1-1.5小时 (vs 全量微调2-3小时)
#
# 成本对比:
#   - 全量微调: 4×GPU, 2-3小时, ~$12-18
#   - LoRA微调: 3×GPU, 1-1.5小时, ~$6-9 (节省50-60%)
#
# 使用方法:
#   chmod +x scripts/run_x_r1_3B_lora.sh
#   ./scripts/run_x_r1_3B_lora.sh
#
# ====================================

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU: 1个vLLM + 3个训练
export WANDB_PROJECT="X-R1-3B-LoRA"
export WANDB_RUN_NAME="x_r1_3b_lora_$(date +%Y%m%d_%H%M%S)"

# 检查必要文件
CONFIG_FILE="recipes/X_R1_zero_3B_lora_config.yaml"
ZERO_CONFIG="recipes/zero3.yaml"
SCRIPT_FILE="src/x_r1/grpo.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

if [ ! -f "$ZERO_CONFIG" ]; then
    echo "❌ 错误: DeepSpeed配置文件 $ZERO_CONFIG 不存在"
    exit 1
fi

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ 错误: 训练脚本 $SCRIPT_FILE 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p output
mkdir -p logs

# 输出训练信息
echo "🚀 开始X-R1 3B模型LoRA训练"
echo "📊 配置文件: $CONFIG_FILE"
echo "🔧 DeepSpeed配置: $ZERO_CONFIG"
echo "💻 使用GPU: $CUDA_VISIBLE_DEVICES"
echo "📝 日志文件: ./output/x_r1_3B_lora_training.log"
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=====================================\n"

# 启动训练
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file $ZERO_CONFIG \
--num_processes=3 \
$SCRIPT_FILE \
--config $CONFIG_FILE \
> ./output/x_r1_3B_lora_training.log 2>&1

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "\n✅ 训练完成成功！"
    echo "📁 输出目录: output/X-R1-3B-LoRA"
    echo "📋 日志文件: output/x_r1_3B_lora_training.log"
    echo "⏰ 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 显示LoRA适配器信息
    if [ -d "output/X-R1-3B-LoRA" ]; then
        echo "\n📈 训练统计:"
        echo "🔸 适配器大小: $(du -sh output/X-R1-3B-LoRA | cut -f1)"
        echo "🔸 检查点数量: $(ls -1 output/X-R1-3B-LoRA/ | wc -l)"
    fi
    
    echo "\n🎯 使用训练好的模型:"
    echo "1. 推理脚本中指定: adapter_name_or_path='output/X-R1-3B-LoRA'"
    echo "2. 合并权重: 可使用PEFT的merge_and_unload()方法"
    echo "3. 部署建议: LoRA适配器可以动态切换，支持多任务部署"
    
else
    echo "\n❌ 训练失败，请检查日志文件: output/x_r1_3B_lora_training.log"
    echo "📋 常见问题排查:"
    echo "1. 检查GPU显存是否足够 (需要~48-60GB, 3×GPU)"
    echo "2. 检查数据集路径是否正确"
    echo "3. 检查模型路径是否存在"
    echo "4. 查看详细错误信息: tail -50 output/x_r1_3B_lora_training.log"
    exit 1
fi 