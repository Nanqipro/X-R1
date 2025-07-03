#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X-R1 数据集生成工具使用示例

展示如何使用data-generate.py脚本生成训练数据集
"""

import subprocess
import os
import json
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """
    运行命令并显示结果
    
    Parameters
    ----------
    command : str
        要执行的命令
    description : str
        命令描述
        
    Returns
    -------
    bool
        是否成功执行
    """
    print(f"\n🚀 {description}")
    print(f"执行命令: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 命令执行成功")
            if result.stdout:
                print(f"输出: {result.stdout}")
            return True
        else:
            print("❌ 命令执行失败")
            if result.stderr:
                print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False

def check_file_exists(file_path: str) -> bool:
    """
    检查文件是否存在
    
    Parameters
    ----------
    file_path : str
        文件路径
        
    Returns
    -------
    bool
        文件是否存在
    """
    exists = Path(file_path).exists()
    status = "✅ 存在" if exists else "❌ 不存在"
    print(f"文件检查 {file_path}: {status}")
    return exists

def preview_dataset(file_path: str, num_samples: int = 1) -> None:
    """
    预览数据集内容
    
    Parameters
    ----------
    file_path : str
        数据集文件路径
    num_samples : int
        预览样本数量
    """
    print(f"\n📖 预览数据集: {file_path}")
    print("-" * 50)
    
    try:
        if not Path(file_path).exists():
            print("❌ 文件不存在")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    print(f"\n样本 {i + 1}:")
                    print(f"系统提示词: {data['system'][:100]}...")
                    
                    conversations = data.get('conversations', [])
                    for j, conv in enumerate(conversations):
                        role = conv.get('from', 'unknown')
                        content = conv.get('value', '')
                        print(f"  {role}: {content[:200]}...")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ 第{i+1}行JSON解析错误: {e}")
                    
    except Exception as e:
        print(f"❌ 预览失败: {e}")

def main():
    """
    主函数 - 演示完整的使用流程
    """
    print("=" * 60)
    print("🎯 X-R1 数据集生成工具使用示例")
    print("=" * 60)
    
    # 1. 检查输入文件
    input_file = "A-data/A-data.jsonl"
    print(f"\n📋 第1步: 检查输入文件")
    if not check_file_exists(input_file):
        print(f"⚠️  输入文件不存在: {input_file}")
        print("请确保A-data.jsonl文件在正确位置")
        return
    
    # 2. 生成小样本数据集（仅处理前5条）
    print(f"\n🔧 第2步: 生成小样本数据集")
    sample_output = "/home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/sample_dataset.jsonl"
    
    # 创建临时的小样本输入文件
    temp_input = "temp_sample_input.jsonl"
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(temp_input, 'w', encoding='utf-8') as f_out:
                for i, line in enumerate(f_in):
                    if i >= 5:  # 只处理前5条
                        break
                    f_out.write(line)
        
        sample_command = f"python data-generate.py --input {temp_input} --output {sample_output} --delay 0.5"
        success = run_command(sample_command, "生成小样本数据集")
        
        # 清理临时文件
        if os.path.exists(temp_input):
            os.remove(temp_input)
        
        if not success:
            print("⚠️  小样本生成失败，请检查API配置和网络连接")
            return
            
    except Exception as e:
        print(f"❌ 创建临时文件失败: {e}")
        return
    
    # 3. 验证生成的数据集
    print(f"\n🔍 第3步: 验证数据集格式")
    validate_command = f"python test_dataset_format.py {sample_output}"
    run_command(validate_command, "验证数据集格式")
    
    # 4. 预览数据集内容
    print(f"\n👀 第4步: 预览数据集内容")
    preview_dataset(sample_output, num_samples=2)
    
    # 5. 显示完整数据集生成命令
    print(f"\n🎯 第5步: 生成完整数据集")
    full_output = "/home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/generated_x_r1_dataset.jsonl"
    full_command = f"python data-generate.py --input {input_file} --output {full_output}"
    
    print("要生成完整数据集，请运行以下命令:")
    print(f"```bash")
    print(f"{full_command}")
    print(f"```")
    
    # 6. 显示训练配置示例
    print(f"\n⚙️  第6步: X-R1训练配置")
    print("数据集生成完成后，可以使用以下配置进行训练:")
    
    config_example = f"""
# 创建训练配置文件: recipes/custom_config.yaml
model_name_or_path: Qwen/Qwen2.5-1.5B
dataset_name: ./LLM-models-datasets/generated_x_r1_dataset.jsonl
dataset_configs:
- train
num_train_epochs: 3
output_dir: output/X-R1-custom

# 启动训练
python src/x_r1/grpo.py recipes/custom_config.yaml
"""
    print(config_example)
    
    print("\n" + "=" * 60)
    print("✨ 示例运行完成！")
    print("💡 提示: 根据需要调整延迟时间和批次大小以优化API使用")
    print("=" * 60)

if __name__ == "__main__":
    main() 