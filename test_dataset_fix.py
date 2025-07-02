from datasets import load_dataset

def test_openthoughts_dataset():
    """测试OpenThoughts-114k数据集的问题和答案提取"""
    
    # 加载少量样本进行测试
    dataset = load_dataset('./LLM-models-datasets/OpenThoughts-114k', split='train[:3]')
    
    print(f"数据集大小: {len(dataset)}")
    print(f"原始字段: {dataset.column_names}")
    
    # 定义提取函数（与grpo.py中相同的逻辑）
    def extract_conversation_content(example):
        """从conversations格式中提取用户问题和助手答案"""
        conversations = example.get("conversations", [])
        result = {"problem": "", "solution": ""}
        
        if conversations:
            # 查找用户消息作为问题
            for conv in conversations:
                if conv.get("from") == "user" or conv.get("from") == "human":
                    result["problem"] = conv.get("value", "")
                    break
            
            # 查找助手消息作为答案
            for conv in conversations:
                if conv.get("from") == "assistant" or conv.get("from") == "gpt":
                    result["solution"] = conv.get("value", "")
                    break
            
            # 如果没有找到用户消息，使用第一个消息作为问题
            if not result["problem"] and conversations:
                result["problem"] = conversations[0].get("value", "")
            
            # 如果没有找到助手消息，使用最后一个消息作为答案
            if not result["solution"] and len(conversations) > 1:
                result["solution"] = conversations[-1].get("value", "")
        
        return result
    
    # 应用转换
    transformed_dataset = dataset.map(extract_conversation_content)
    
    print(f"转换后字段: {transformed_dataset.column_names}")
    
    # 检查前几个样例
    for i in range(min(3, len(transformed_dataset))):
        print(f"\n=== 样例 {i} ===")
        example = transformed_dataset[i]
        
        # 检查是否成功提取了问题和答案
        has_problem = bool(example.get("problem", "").strip())
        has_solution = bool(example.get("solution", "").strip())
        
        print(f"有问题字段: {has_problem}")
        print(f"有答案字段: {has_solution}")
        
        if has_problem:
            problem = example["problem"][:200] + "..." if len(example["problem"]) > 200 else example["problem"]
            print(f"问题预览: {problem}")
        
        if has_solution:
            solution = example["solution"][:200] + "..." if len(example["solution"]) > 200 else example["solution"]
            print(f"答案预览: {solution}")
    
    return transformed_dataset

if __name__ == "__main__":
    try:
        test_openthoughts_dataset()
        print("\n✅ 数据集处理测试成功！")
    except Exception as e:
        print(f"\n❌ 数据集处理测试失败: {e}") 