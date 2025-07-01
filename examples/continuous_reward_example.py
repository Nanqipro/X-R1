#!/usr/bin/env python3
"""
连续奖励函数使用示例

这个脚本展示了如何使用改进的连续奖励函数替代原来的二元奖励函数，
从而获得更细粒度的训练信号。

运行示例：
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/continuous_reward_example.py \
    --model_name_or_path "xiaodongguaAIGC/X-R1-3B" \
    --dataset_name "HuggingFaceH4/MATH-500" \
    --reward_funcs "accuracy_continuous" "format_continuous" "reasoning_steps" \
    --output_dir "./output/continuous_reward_training" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --save_steps 500

对比不同奖励组合的训练效果：

1. 原始二元奖励（总是0或1）：
   --reward_funcs "accuracy" "format"

2. 连续奖励组合1（更细粒度）：
   --reward_funcs "accuracy_continuous" "format_continuous"

3. 连续奖励组合2（包含推理步骤）：
   --reward_funcs "accuracy_continuous" "format_continuous" "reasoning_steps"

4. 连续奖励组合3（包含长度惩罚）：
   --reward_funcs "accuracy_continuous" "format_continuous" "length"

5. 高级连续奖励组合（余弦缩放 + 重复惩罚）：
   --reward_funcs "cosine" "repetition_penalty" "format_continuous"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/x_r1'))

from grpo import main
from configs import GRPOConfig
from grpo import GRPOScriptArguments
from trl import ModelConfig
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="连续奖励函数训练示例")
    
    # 模型和数据集参数
    parser.add_argument("--model_name_or_path", type=str, default="xiaodongguaAIGC/X-R1-3B",
                       help="模型路径或名称")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500",
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="./output/continuous_reward_training",
                       help="输出目录")
    
    # 奖励函数参数
    parser.add_argument("--reward_funcs", nargs="+", 
                       choices=["accuracy", "accuracy_continuous", "format", "format_continuous", 
                               "reasoning_steps", "cosine", "repetition_penalty", "length"],
                       default=["accuracy_continuous", "format_continuous"],
                       help="要使用的奖励函数列表")
    
    # 训练参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="每个设备的批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                       help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日志步数")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="保存步数")
    
    # 奖励函数特定参数
    parser.add_argument("--cosine_min_value_correct", type=float, default=0.5,
                       help="余弦奖励：正确答案最小奖励")
    parser.add_argument("--cosine_max_value_correct", type=float, default=1.0,
                       help="余弦奖励：正确答案最大奖励")
    parser.add_argument("--cosine_min_value_wrong", type=float, default=0.0,
                       help="余弦奖励：错误答案最小奖励")
    parser.add_argument("--cosine_max_value_wrong", type=float, default=-0.5,
                       help="余弦奖励：错误答案最大奖励")
    parser.add_argument("--cosine_max_len", type=int, default=1000,
                       help="余弦奖励：最大长度")
    
    return parser.parse_args()

def create_training_configs(args):
    """创建训练配置"""
    
    # 脚本参数
    script_args = GRPOScriptArguments(
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        dataset_config=None,
        reward_funcs=args.reward_funcs,
        cosine_min_value_correct=args.cosine_min_value_correct,
        cosine_max_value_correct=args.cosine_max_value_correct,
        cosine_min_value_wrong=args.cosine_min_value_wrong,
        cosine_max_value_wrong=args.cosine_max_value_wrong,
        cosine_max_len=args.cosine_max_len,
    )
    
    # 训练参数
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="no",
        save_strategy="steps",
        report_to=["wandb"],
        run_name=f"continuous_reward_{'+'.join(args.reward_funcs)}",
        logging_dir=f"{args.output_dir}/logs",
    )
    
    # 模型参数
    model_args = ModelConfig(
        model_name_or_path=args.model_name_or_path,
    )
    
    return script_args, training_args, model_args

def print_reward_comparison():
    """打印不同奖励函数的对比说明"""
    print("=" * 80)
    print("奖励函数对比分析：")
    print("=" * 80)
    
    comparisons = [
        {
            "name": "原始二元奖励",
            "funcs": ["accuracy", "format"],
            "特点": [
                "✗ 只有0和1两种奖励值",
                "✗ 无法区分'接近正确'和'完全错误'",
                "✗ 训练信号稀疏，容易陷入局部最优",
                "✓ 计算简单，训练稳定"
            ]
        },
        {
            "name": "连续精度奖励",
            "funcs": ["accuracy_continuous", "format_continuous"],
            "特点": [
                "✓ 提供0-1之间的连续奖励",
                "✓ 部分匹配可获得30%奖励",
                "✓ 包含长度惩罚，避免过长答案",
                "✓ 更细粒度的训练信号"
            ]
        },
        {
            "name": "推理步骤奖励",
            "funcs": ["accuracy_continuous", "format_continuous", "reasoning_steps"],
            "特点": [
                "✓ 鼓励步骤化推理",
                "✓ 识别'Step 1:', '1.', 'First,'等模式",
                "✓ 3步或以上推理得满分",
                "✓ 提升答案可解释性"
            ]
        },
        {
            "name": "长度优化奖励",
            "funcs": ["accuracy_continuous", "format_continuous", "length"],
            "特点": [
                "✓ 鼓励简洁正确的答案",
                "✓ 惩罚过度冗长的回答",
                "✓ 基于Kimi 1.5技术报告",
                "✓ 提高token效率"
            ]
        },
        {
            "name": "高级连续奖励",
            "funcs": ["cosine", "repetition_penalty", "format_continuous"],
            "特点": [
                "✓ 余弦缩放：短正确答案奖励更高",
                "✓ 重复惩罚：避免n-gram重复",
                "✓ 最精细的奖励控制",
                "✓ 适合高级训练场景"
            ]
        }
    ]
    
    for comp in comparisons:
        print(f"\n【{comp['name']}】")
        print(f"函数组合: {comp['funcs']}")
        print("特点:")
        for feature in comp['特点']:
            print(f"  {feature}")
    
    print("\n" + "=" * 80)
    print("建议使用顺序：")
    print("1. 先用 accuracy_continuous + format_continuous 验证基础效果")
    print("2. 再加入 reasoning_steps 提升推理能力")
    print("3. 最后尝试 cosine + repetition_penalty 精细调优")
    print("=" * 80)

if __name__ == "__main__":
    # 打印对比说明
    print_reward_comparison()
    
    # 解析参数
    args = parse_args()
    
    # 创建配置
    script_args, training_args, model_args = create_training_configs(args)
    
    print(f"\n开始训练，使用奖励函数: {args.reward_funcs}")
    print(f"输出目录: {args.output_dir}")
    
    # 开始训练
    try:
        main(script_args, training_args, model_args)
        print("\n训练完成！")
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise 