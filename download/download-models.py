from modelscope import snapshot_download
import os
from typing import Optional

def download_model_from_modelscope(model_name: str, local_dir: str) -> bool:
    """
    从魔塔社区下载模型
    
    Parameters
    ----------
    model_name : str
        模型名称，格式为 "组织名/模型名"
    local_dir : str
        本地存储目录路径
        
    Returns
    -------
    bool
        下载是否成功
    """
    try:
        print(f"从魔塔社区下载模型 {model_name} 到 {local_dir}...")
        # ModelScope的snapshot_download使用不同的参数
        snapshot_download(
            model_id=model_name,
            cache_dir=local_dir,
            # revision='master'  # 可选：指定版本
        )
        print(f"模型下载成功到 {local_dir}")
        return True
    except Exception as e:
        print(f"下载模型时出错: {e}")
        return False

def main() -> None:
    """
    主函数：执行模型下载流程
    """
    # 魔塔社区上的模型名称（对应Qwen2.5-3B-Instruct）
    model_name = "xiaodongguaAIGC/X-R1-3B"
    # 备选模型：model_name = "AI-ModelScope/gpt-neo-1.3B"
    
    # 定义本地存储路径
    base_dir = "../LLM-models-datasets"
    local_model_dir = os.path.join(base_dir, model_name.split('/')[-1])
    
    # 确保基础目录存在
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"创建目录: {base_dir}")
    
    # 执行下载
    success = download_model_from_modelscope(
        model_name=model_name,
        local_dir=local_model_dir
    )
    
    if success:
        print("✅ 模型下载完成！")
    else:
        print("❌ 模型下载失败，请检查网络连接和模型名称")

if __name__ == "__main__":
    main()
