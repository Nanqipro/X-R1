#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集格式验证脚本

用于验证生成的数据集是否符合X-R1框架的格式要求
"""

import json
import argparse
import re
from typing import Dict, List, Any
from pathlib import Path

def validate_response_format(assistant_value: str) -> tuple[bool, str]:
    """
    验证assistant回答是否符合要求的格式
    
    Parameters
    ----------
    assistant_value : str
        assistant的回答内容
        
    Returns
    -------
    tuple[bool, str]
        (是否有效, 错误信息)
    """
    # 检查是否包含think和answer标签
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(pattern, assistant_value, re.DOTALL)
    
    if not match:
        return False, "assistant回答必须包含<think>...</think>和<answer>...</answer>标签"
    
    think_content = match.group(1).strip()
    answer_content = match.group(2).strip()
    
    if not think_content:
        return False, "<think>标签内容不能为空"
    
    if not answer_content:
        return False, "<answer>标签内容不能为空"
    
    return True, ""

def validate_x_r1_format(data_item: Dict[str, Any]) -> tuple[bool, str]:
    """
    验证单个数据项是否符合X-R1格式
    
    Parameters
    ----------
    data_item : Dict[str, Any]
        待验证的数据项
        
    Returns
    -------
    tuple[bool, str]
        (是否有效, 错误信息)
    """
    # 检查必需字段
    if "system" not in data_item:
        return False, "缺少 'system' 字段"
    
    if "conversations" not in data_item:
        return False, "缺少 'conversations' 字段"
    
    # 检查system字段类型
    if not isinstance(data_item["system"], str):
        return False, "'system' 字段必须是字符串"
    
    # 检查conversations字段类型
    conversations = data_item["conversations"]
    if not isinstance(conversations, list):
        return False, "'conversations' 字段必须是列表"
    
    if len(conversations) < 2:
        return False, "'conversations' 必须至少包含2个对话（用户和助手）"
    
    # 检查对话格式
    expected_roles = ["user", "assistant"]
    for i, conv in enumerate(conversations[:2]):  # 只检查前两个对话
        if not isinstance(conv, dict):
            return False, f"对话 {i} 必须是字典格式"
        
        if "from" not in conv:
            return False, f"对话 {i} 缺少 'from' 字段"
        
        if "value" not in conv:
            return False, f"对话 {i} 缺少 'value' 字段"
        
        if conv["from"] != expected_roles[i]:
            return False, f"对话 {i} 的 'from' 字段应为 '{expected_roles[i]}'"
        
        if not isinstance(conv["value"], str):
            return False, f"对话 {i} 的 'value' 字段必须是字符串"
        
        if not conv["value"].strip():
            return False, f"对话 {i} 的 'value' 字段不能为空"
        
        # 特别检查assistant回答的格式
        if conv["from"] == "assistant":
            format_valid, format_error = validate_response_format(conv["value"])
            if not format_valid:
                return False, f"对话 {i} (assistant): {format_error}"
    
    return True, ""

def validate_dataset_file(file_path: str) -> Dict[str, Any]:
    """
    验证整个数据集文件
    
    Parameters
    ----------
    file_path : str
        数据集文件路径
        
    Returns
    -------
    Dict[str, Any]
        验证结果统计
    """
    if not Path(file_path).exists():
        return {
            "success": False,
            "error": f"文件不存在: {file_path}",
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors": []
        }
    
    total_count = 0
    valid_count = 0
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                
                try:
                    data_item = json.loads(line)
                    is_valid, error_msg = validate_x_r1_format(data_item)
                    
                    if is_valid:
                        valid_count += 1
                    else:
                        errors.append(f"第 {line_num} 行: {error_msg}")
                        
                except json.JSONDecodeError as e:
                    errors.append(f"第 {line_num} 行: JSON解析错误 - {str(e)}")
                
                # 限制错误数量，避免输出过长
                if len(errors) >= 20:
                    errors.append("... (更多错误被省略)")
                    break
    
    except Exception as e:
        return {
            "success": False,
            "error": f"读取文件时发生错误: {str(e)}",
            "total": total_count,
            "valid": valid_count,
            "invalid": total_count - valid_count,
            "errors": errors
        }
    
    return {
        "success": True,
        "total": total_count,
        "valid": valid_count,
        "invalid": total_count - valid_count,
        "errors": errors
    }

def print_validation_results(results: Dict[str, Any]) -> None:
    """
    打印验证结果
    
    Parameters
    ----------
    results : Dict[str, Any]
        验证结果
    """
    print("=" * 60)
    print("数据集格式验证结果")
    print("=" * 60)
    
    if not results["success"]:
        print(f"❌ 验证失败: {results['error']}")
        return
    
    total = results["total"]
    valid = results["valid"]
    invalid = results["invalid"]
    
    if total == 0:
        print("⚠️  警告: 文件中没有找到有效数据")
        return
    
    success_rate = (valid / total) * 100 if total > 0 else 0
    
    print(f"📊 统计信息:")
    print(f"   总数据条数: {total}")
    print(f"   有效条数: {valid}")
    print(f"   无效条数: {invalid}")
    print(f"   有效率: {success_rate:.2f}%")
    print()
    
    if invalid == 0:
        print("✅ 所有数据项格式都正确!")
    else:
        print("❌ 发现格式错误:")
        for error in results["errors"]:
            print(f"   - {error}")
    
    print("=" * 60)

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="验证X-R1数据集格式")
    parser.add_argument("file", help="要验证的数据集文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="显示详细信息")
    
    args = parser.parse_args()
    
    print(f"正在验证文件: {args.file}")
    
    results = validate_dataset_file(args.file)
    print_validation_results(results)
    
    # 如果需要详细信息且有有效数据，显示第一个样本
    if args.verbose and results.get("valid", 0) > 0:
        print("\n📝 第一个有效数据样本:")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data_item = json.loads(line)
                        is_valid, _ = validate_x_r1_format(data_item)
                        if is_valid:
                            print(json.dumps(data_item, ensure_ascii=False, indent=2))
                            break
        except Exception as e:
            print(f"显示样本时出错: {e}")

if __name__ == "__main__":
    main() 