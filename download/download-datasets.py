#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMLU数据集下载器

该脚本用于从Hugging Face下载cais/mmlu数据集到本地指定目录
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_download_directory(base_path: str) -> Path:
    """
    设置并创建下载目录
    
    Parameters
    ----------
    base_path : str
        基础路径字符串
        
    Returns
    -------
    Path
        创建的目录路径对象
    """
    download_dir = Path(base_path)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"下载目录已准备: {download_dir}")
    return download_dir


def download_mmlu_dataset(
    dataset_name: str = "cais/mmlu", 
    config_name: str = "all",
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True
) -> Dict[str, Dataset]:
    """
    下载MMLU数据集
    
    Parameters
    ----------
    dataset_name : str, optional
        数据集名称，默认为 "cais/mmlu"
    config_name : str, optional
        配置名称，默认为 "all" (包含所有学科)
    cache_dir : str, optional
        缓存目录路径
    trust_remote_code : bool, optional
        是否信任远程代码，默认为True
        
    Returns
    -------
    Dict[str, Dataset]
        包含各个分割的数据集字典
        
    Raises
    ------
    Exception
        当下载失败时抛出异常
    """
    try:
        logger.info(f"开始下载数据集: {dataset_name}")
        logger.info(f"配置: {config_name}")
        logger.info(f"目标路径: {cache_dir}")
        
        # 下载数据集 (指定配置名称)
        dataset = load_dataset(
            dataset_name,
            config_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        
        logger.info("数据集下载完成!")
        
        # 显示数据集信息
        for split_name, split_data in dataset.items():
            logger.info(f"  - {split_name}: {len(split_data)} 条记录")
            
        return dataset
        
    except Exception as e:
        logger.error(f"数据集下载失败: {str(e)}")
        raise


def verify_dataset(dataset: Dict[str, Dataset]) -> bool:
    """
    验证下载的数据集完整性
    
    Parameters
    ----------
    dataset : Dict[str, Dataset]
        已下载的数据集
        
    Returns
    -------
    bool
        验证是否通过
    """
    try:
        logger.info("验证数据集完整性...")
        
        if not dataset:
            logger.error("数据集为空")
            return False
            
        # 检查基本分割
        expected_splits = ['test', 'validation', 'dev']
        available_splits = list(dataset.keys())
        
        logger.info(f"可用的数据分割: {available_splits}")
        
        # 检查每个分割是否有数据
        for split_name, split_data in dataset.items():
            if len(split_data) == 0:
                logger.warning(f"分割 {split_name} 没有数据")
                return False
                
        logger.info("数据集验证通过!")
        return True
        
    except Exception as e:
        logger.error(f"验证过程出错: {str(e)}")
        return False


def main() -> None:
    """
    主函数：执行MMLU数据集下载流程
    """
    # 目标下载路径
    target_path = "../LLM-models"
    dataset_name = "cais/mmlu"
    
    try:
        logger.info("=" * 50)
        logger.info("MMLU数据集下载器启动")
        logger.info("=" * 50)
        
        # 1. 设置下载目录
        download_dir = setup_download_directory(target_path)
        
        # 2. 下载数据集
        dataset = download_mmlu_dataset(
            dataset_name=dataset_name,
            config_name="all",  # 下载所有学科的完整数据集
            cache_dir=str(download_dir)
        )
        
        # 3. 验证数据集
        if verify_dataset(dataset):
            logger.info("✅ 数据集下载并验证成功!")
            logger.info(f"📁 保存位置: {download_dir}")
        else:
            logger.error("❌ 数据集验证失败!")
            sys.exit(1)
            
        # 4. 显示数据集样例
        logger.info("\n📋 数据集样例预览:")
        for split_name, split_data in dataset.items():
            if len(split_data) > 0:
                logger.info(f"\n{split_name} 分割样例:")
                sample = split_data[0]
                for key, value in sample.items():
                    logger.info(f"  - {key}: {str(value)[:100]}...")
                break  # 只显示第一个分割的样例
                
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)
    
    logger.info("\n🎉 任务完成!")


if __name__ == "__main__":
    main()
