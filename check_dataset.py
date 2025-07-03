from datasets import load_dataset
import json

# 加载数据集
try:
    dataset = load_dataset('./LLM-models-datasets/Bespoke-Stratos-17k', split='train[:3]')  # 只加载前3个样本
    
    # 查看数据集的基本信息
    print('数据集类型:', type(dataset))
    
    # 安全地获取数据集大小
    try:
        dataset_len = len(dataset)  # type: ignore
        print('数据集大小:', dataset_len)
    except Exception as e:
        print(f'获取数据集大小时出错: {e}')
    
    # 安全地获取字段名称
    try:
        column_names = getattr(dataset, 'column_names', None)
        if column_names is not None:
            print('字段名称:', column_names)
        else:
            print('字段名称: 无法获取')
    except Exception as e:
        print(f'获取字段名称时出错: {e}')
    
    # 安全地获取数据集特征
    try:
        features = getattr(dataset, 'features', None)
        if features is not None:
            print('数据集特征:', features)
        else:
            print('数据集特征: 无法获取')
    except Exception as e:
        print(f'获取数据集特征时出错: {e}')
    
    # 查看前几个样例
    print('\n=== 前几个样例 ===')
    try:
        # 尝试通过索引访问前几个样例
        for i in range(3):
            try:
                print(f'\n样例 {i}:')
                example = dataset[i]  # type: ignore
                # 处理字典类型的样例
                if hasattr(example, 'items'):
                    for key, value in example.items():  # type: ignore
                        if isinstance(value, str) and len(value) > 200:
                            print(f'  {key}: {value[:200]}...')
                        elif isinstance(value, list):
                            print(f'  {key}: {value[:2]}...')  # 只显示前2个元素
                        else:
                            print(f'  {key}: {value}')
                else:
                    print(f'  数据: {example}')
            except (IndexError, TypeError):
                # 如果索引访问失败，尝试迭代器
                if i == 0:  # 只在第一次失败时尝试迭代器
                    print('索引访问失败，尝试使用迭代器...')
                    try:
                        for j, example in enumerate(dataset):  # type: ignore
                            if j >= 3:  # 只显示前3个样例
                                break
                            print(f'\n样例 {j}:')
                            if hasattr(example, 'items'):
                                for key, value in example.items():  # type: ignore
                                    if isinstance(value, str) and len(value) > 200:
                                        print(f'  {key}: {value[:200]}...')
                                    elif isinstance(value, list):
                                        print(f'  {key}: {value[:2]}...')  # 只显示前2个元素
                                    else:
                                        print(f'  {key}: {value}')
                            else:
                                print(f'  数据: {example}')
                        break
                    except Exception as iter_e:
                        print(f'迭代器访问也失败: {iter_e}')
                        break
                break
    except Exception as e:
        print(f'遍历样例时出错: {e}')
        
except Exception as e:
    print(f'加载数据集出错: {e}') 