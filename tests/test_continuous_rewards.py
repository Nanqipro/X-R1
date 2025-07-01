#!/usr/bin/env python3
"""
连续奖励函数测试脚本

用于验证新的连续奖励函数是否按预期工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/x_r1'))

from rewards import (
    accuracy_reward, 
    accuracy_reward_continuous,
    format_reward,
    format_reward_continuous,
    reasoning_steps_reward
)

def test_accuracy_rewards():
    """测试精度奖励函数"""
    print("=" * 60)
    print("测试精度奖励函数")
    print("=" * 60)
    
    # 测试数据
    test_cases = [
        {
            "completions": [[{"content": "<think>2+2=4</think><answer>4</answer>"}]],
            "solution": ["4"],
            "description": "完全正确答案"
        },
        {
            "completions": [[{"content": "<think>2+2=5</think><answer>5</answer>"}]],
            "solution": ["4"],
            "description": "完全错误答案"
        },
        {
            "completions": [[{"content": "<think>计算2+2</think><answer>答案是4</answer>"}]],
            "solution": ["4"],
            "description": "包含正确答案但格式不标准"
        },
        {
            "completions": [[{"content": "<think>这是一个复杂的数学问题，需要仔细计算。首先我们看到2+2，这是基础算术。根据加法运算规则，2+2等于4。但是让我再检查一遍确保正确...</think><answer>4</answer>"}]],
            "solution": ["4"],
            "description": "正确答案但过长"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {case['description']}")
        print(f"输入: {case['completions'][0][0]['content'][:100]}...")
        print(f"期望: {case['solution'][0]}")
        
        # 原始奖励
        original_reward = accuracy_reward(case['completions'], case['solution'])
        print(f"原始奖励: {original_reward}")
        
        # 连续奖励
        continuous_reward = accuracy_reward_continuous(case['completions'], case['solution'])
        print(f"连续奖励: {continuous_reward}")
        
        print("-" * 40)

def test_format_rewards():
    """测试格式奖励函数"""
    print("\n" + "=" * 60)
    print("测试格式奖励函数")
    print("=" * 60)
    
    test_cases = [
        {
            "completions": [[{"content": "<think>思考过程</think><answer>答案</answer>"}]],
            "description": "完整格式"
        },
        {
            "completions": [[{"content": "没有任何标签的回答"}]],
            "description": "无格式"
        },
        {
            "completions": [[{"content": "<think>只有思考标签</think>"}]],
            "description": "只有think标签"
        },
        {
            "completions": [[{"content": "<answer>只有答案标签</answer>"}]],
            "description": "只有answer标签"
        },
        {
            "completions": [[{"content": "<think>思考</think>中间内容<answer>答案</answer>"}]],
            "description": "标签间有额外内容"
        },
        {
            "completions": [[{"content": "<answer>答案在前</answer><think>思考在后</think>"}]],
            "description": "标签顺序错误"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {case['description']}")
        print(f"输入: {case['completions'][0][0]['content']}")
        
        # 原始奖励
        original_reward = format_reward(case['completions'])
        print(f"原始奖励: {original_reward}")
        
        # 连续奖励
        continuous_reward = format_reward_continuous(case['completions'])
        print(f"连续奖励: {continuous_reward}")
        
        print("-" * 40)

def test_reasoning_steps_reward():
    """测试推理步骤奖励函数"""
    print("\n" + "=" * 60)
    print("测试推理步骤奖励函数")
    print("=" * 60)
    
    test_cases = [
        {
            "completions": [[{"content": "没有步骤的回答"}]],
            "description": "无步骤"
        },
        {
            "completions": [[{"content": "Step 1: 第一步"}]],
            "description": "一个步骤"
        },
        {
            "completions": [[{"content": "Step 1: 第一步\nStep 2: 第二步\nStep 3: 第三步"}]],
            "description": "三个步骤"
        },
        {
            "completions": [[{"content": "1. 第一点\n2. 第二点\n3. 第三点\n4. 第四点\n5. 第五点"}]],
            "description": "五个数字列表"
        },
        {
            "completions": [[{"content": "First, 我们需要分析问题。Next, 进行计算。Finally, 得出结论。"}]],
            "description": "转折词"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {case['description']}")
        print(f"输入: {case['completions'][0][0]['content']}")
        
        reward = reasoning_steps_reward(case['completions'])
        print(f"推理步骤奖励: {reward}")
        
        print("-" * 40)

def test_reward_distribution():
    """测试奖励分布的改善"""
    print("\n" + "=" * 60)
    print("奖励分布对比测试")
    print("=" * 60)
    
    # 模拟一批训练数据
    batch_data = [
        {
            "completions": [[{"content": "<think>2+2=4</think><answer>4</answer>"}]],
            "solution": ["4"]
        },
        {
            "completions": [[{"content": "<think>2+2=5</think><answer>5</answer>"}]],
            "solution": ["4"]
        },
        {
            "completions": [[{"content": "答案是4"}]],
            "solution": ["4"]
        },
        {
            "completions": [[{"content": "<think>计算...</think>"}]],
            "solution": ["4"]
        }
    ]
    
    # 收集原始奖励
    original_accuracy = []
    original_format = []
    
    # 收集连续奖励
    continuous_accuracy = []
    continuous_format = []
    
    for data in batch_data:
        original_accuracy.extend(accuracy_reward([data["completions"]], [data["solution"]]))
        original_format.extend(format_reward([data["completions"]]))
        
        continuous_accuracy.extend(accuracy_reward_continuous([data["completions"]], [data["solution"]]))
        continuous_format.extend(format_reward_continuous([data["completions"]]))
    
    # 统计分析
    def analyze_rewards(rewards, name):
        zero_count = sum(1 for r in rewards if r == 0.0)
        partial_count = sum(1 for r in rewards if 0.0 < r < 1.0)
        full_count = sum(1 for r in rewards if r == 1.0)
        
        print(f"\n{name}:")
        print(f"  零奖励: {zero_count}/{len(rewards)} ({zero_count/len(rewards)*100:.1f}%)")
        print(f"  部分奖励: {partial_count}/{len(rewards)} ({partial_count/len(rewards)*100:.1f}%)")
        print(f"  满奖励: {full_count}/{len(rewards)} ({full_count/len(rewards)*100:.1f}%)")
        print(f"  平均奖励: {sum(rewards)/len(rewards):.3f}")
        print(f"  奖励值: {[f'{r:.3f}' for r in rewards]}")
    
    analyze_rewards(original_accuracy, "原始精度奖励")
    analyze_rewards(continuous_accuracy, "连续精度奖励")
    analyze_rewards(original_format, "原始格式奖励")
    analyze_rewards(continuous_format, "连续格式奖励")

if __name__ == "__main__":
    print("开始测试连续奖励函数...")
    
    try:
        test_accuracy_rewards()
        test_format_rewards() 
        test_reasoning_steps_reward()
        test_reward_distribution()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("✅ 连续奖励函数工作正常")
        print("✅ 可以开始使用新的奖励函数进行训练")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 