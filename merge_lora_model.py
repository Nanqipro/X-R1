#!/usr/bin/env python3
"""
合并LoRA适配器与基础模型的脚本

此脚本将LoRA适配器权重与基础模型合并，生成vLLM可直接使用的完整模型。
"""

import os
import argparse
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def merge_lora_model(
    base_model_path: str, 
    lora_model_path: str, 
    output_path: str,
    device: str = "cuda"
) -> None:
    """
    合并LoRA适配器与基础模型
    
    Parameters
    ----------
    base_model_path : str
        基础模型路径
    lora_model_path : str 
        LoRA适配器模型路径
    output_path : str
        合并后模型的输出路径
    device : str, optional
        计算设备, 默认为 "cuda"
    """
    print(f"正在加载基础模型: {base_model_path}")
    
    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
    if not os.path.exists(lora_model_path):
        raise FileNotFoundError(f"LoRA模型路径不存在: {lora_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"正在加载LoRA适配器: {lora_model_path}")
    
    # 加载LoRA模型
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16
    )
    
    print("正在合并LoRA权重...")
    
    # 合并LoRA权重到基础模型
    merged_model = lora_model.merge_and_unload()
    
    print(f"正在保存合并后的模型到: {output_path}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存合并后的模型
    merged_model.save_pretrained(
        output_path,
        save_function=merged_model.save_pretrained,
        max_shard_size="2GB"
    )
    
    # 复制tokenizer相关文件
    print("正在复制tokenizer文件...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ 模型合并完成! 合并后的模型保存在: {output_path}")
    print(f"现在可以在vLLM中使用路径: {output_path}")

def main():
    """主函数，处理命令行参数并执行合并操作"""
    parser = argparse.ArgumentParser(
        description="合并LoRA适配器与基础模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python merge_lora_model.py --base ./LLM-models-datasets/Qwen2.5-3B \\
                            --lora ./LLM-models-datasets/X-R1-3B-LoRA-Advanced \\
                            --output ./LLM-models-datasets/X-R1-3B-Merged
        """
    )
    
    parser.add_argument(
        "--base", 
        type=str, 
        default="./LLM-models-datasets/Qwen2.5-3B",
        help="基础模型路径 (默认: ./LLM-models-datasets/Qwen2.5-3B)"
    )
    
    parser.add_argument(
        "--lora", 
        type=str, 
        default="./LLM-models-datasets/X-R1-3B-LoRA-Advanced-Fast/checkpoint-60",
        help="LoRA适配器模型路径 (默认: ./LLM-models-datasets/X-R1-3B-LoRA-Advanced-Fast/checkpoint-60)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./LLM-models-datasets/X-R1-3B-Merged-Fast",
        help="合并后模型输出路径 (默认: ./LLM-models-datasets/X-R1-3B-Merged-Fast)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (默认: cuda)"
    )
    
    args = parser.parse_args()
    
    try:
        merge_lora_model(
            base_model_path=args.base,
            lora_model_path=args.lora, 
            output_path=args.output,
            device=args.device
        )
    except Exception as e:
        print(f"❌ 合并过程中发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 