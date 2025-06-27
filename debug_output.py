#!/usr/bin/env python3
"""
调试脚本：检查模型输出和reward计算
"""

import sys
import os
sys.path.append('src/x_r1')

from rewards import accuracy_reward, format_reward
import re

# 模拟一些可能的模型输出
test_outputs = [
    # 格式正确的输出
    [{"content": "<think>Let me solve this step by step. 2+3=5</think><answer>5</answer>"}],
    
    # 格式不正确的输出 
    [{"content": "The answer is 5"}],
    
    # 只有数学内容
    [{"content": "2 + 3 = 5"}],
    
    # 空输出
    [{"content": ""}],
]

test_solutions = ["5", "5", "5", "5"]

print("=== 测试模型输出格式 ===")
for i, (output, solution) in enumerate(zip(test_outputs, test_solutions)):
    print(f"\n测试 {i+1}:")
    print(f"模型输出: {output[0]['content']}")
    print(f"期望答案: {solution}")
    
    # 测试格式奖励
    format_rew = format_reward([output])
    print(f"格式奖励: {format_rew}")
    
    # 测试准确率奖励  
    try:
        acc_rew = accuracy_reward([output], [solution])
        print(f"准确率奖励: {acc_rew}")
    except Exception as e:
        print(f"准确率奖励失败: {e}")

print("\n=== 检查实际日志中的解析问题 ===")
# 从日志中可以看到的实际模式
actual_failed_cases = [
    "answer_parsed: [] \ngold_parsed: [30, '30'] \nreward: 0.0",
    "answer_parsed: [Eq(8/(3*c), 8/((3*c))), '\\frac{8}{3} \\div c = \\frac{8}{3c}'] \ngold_parsed: [2/3, '\\frac{2}{3}'] \nreward: 0.0"
]

for case in actual_failed_cases:
    print(case)
    print() 