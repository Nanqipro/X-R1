from datasets import load_dataset
import json

# 加载数据集
try:
    dataset = load_dataset('./LLM-models-datasets/OpenThoughts-114k', split='train[:3]')  # 只加载前3个样本
    
    # 查看数据集的基本信息
    print('数据集大小:', len(dataset))
    print('字段名称:', dataset.column_names)
    print('数据集特征:', dataset.features)
    
    # 查看前几个样例
    print('\n=== 前几个样例 ===')
    for i in range(len(dataset)):
        print(f'\n样例 {i}:')
        example = dataset[i]
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 200:
                print(f'  {key}: {value[:200]}...')
            elif isinstance(value, list):
                print(f'  {key}: {value[:2]}...')  # 只显示前2个元素
            else:
                print(f'  {key}: {value}')
except Exception as e:
    print(f'加载数据集出错: {e}') 