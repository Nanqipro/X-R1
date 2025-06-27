import os
import asyncio
from typing import Optional, List, Dict, Any
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    ModelScope数据集下载器
    
    提供从ModelScope平台下载各种类型数据集的功能，支持批量下载和自定义配置
    """
    
    def __init__(self, base_dir: str = "../LLM-models-datasets") -> None:
        """
        初始化数据集下载器
        
        Parameters
        ----------
        base_dir : str
            数据集存储的基础目录路径
        """
        self.base_dir = base_dir
        self._ensure_base_dir_exists()
        
    def _ensure_base_dir_exists(self) -> None:
        """
        确保基础目录存在，不存在则创建
        """
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info(f"创建基础目录: {self.base_dir}")
    
    async def download_dataset_async(
        self, 
        dataset_name: str, 
        subset_name: Optional[str] = None,
        split: str = "train",
        download_mode: str = "force_redownload"
    ) -> bool:
        """
        异步下载指定数据集
        
        Parameters
        ----------
        dataset_name : str
            数据集名称，格式为 "组织名/数据集名"
        subset_name : Optional[str]
            子数据集名称（可选）
        split : str
            数据集分割类型，默认为 "train"
        download_mode : str
            下载模式，默认为 "force_redownload"
            
        Returns
        -------
        bool
            下载是否成功
        """
        try:
            logger.info(f"开始异步下载数据集: {dataset_name}")
            
            # 构建本地存储路径
            dataset_local_name = dataset_name.split('/')[-1]
            if subset_name:
                dataset_local_name += f"_{subset_name}"
            
            local_path = os.path.join(self.base_dir, dataset_local_name)
            
            # 使用ModelScope数据集API下载
            dataset = MsDataset.load(
                dataset_name=dataset_name,
                subset_name=subset_name,
                split=split,
                download_mode=getattr(DownloadMode, download_mode.upper(), DownloadMode.FORCE_REDOWNLOAD)
            )
            
            # 记录数据集信息（ModelScope数据集通常自动缓存）
            logger.info(f"数据集已缓存，可通过 MsDataset.load() 访问")
            
            logger.info(f"✅ 数据集 {dataset_name} 下载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载数据集 {dataset_name} 时出错: {e}")
            return False
    
    def download_dataset_sync(
        self, 
        dataset_name: str, 
        subset_name: Optional[str] = None,
        split: str = "train"
    ) -> bool:
        """
        同步下载指定数据集
        
        Parameters
        ----------
        dataset_name : str
            数据集名称，格式为 "组织名/数据集名"
        subset_name : Optional[str]
            子数据集名称（可选）
        split : str
            数据集分割类型，默认为 "train"
            
        Returns
        -------
        bool
            下载是否成功
        """
        try:
            logger.info(f"开始同步下载数据集: {dataset_name}")
            
            # 使用ModelScope数据集API下载
            dataset = MsDataset.load(
                dataset_name=dataset_name,
                subset_name=subset_name,
                split=split
            )
            
            # 构建本地存储路径
            dataset_local_name = dataset_name.split('/')[-1]
            if subset_name:
                dataset_local_name += f"_{subset_name}"
            
            local_path = os.path.join(self.base_dir, dataset_local_name)
            
            # 记录数据集信息（ModelScope数据集通常自动缓存）
            logger.info(f"数据集已缓存，可通过 MsDataset.load() 访问")
            
            logger.info(f"✅ 数据集 {dataset_name} 下载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载数据集 {dataset_name} 时出错: {e}")
            return False
    
    async def batch_download_datasets(self, dataset_configs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        批量异步下载多个数据集
        
        Parameters
        ----------
        dataset_configs : List[Dict[str, Any]]
            数据集配置列表，每个配置包含数据集下载参数
            
        Returns
        -------
        Dict[str, bool]
            数据集名称到下载结果的映射
        """
        logger.info(f"开始批量下载 {len(dataset_configs)} 个数据集")
        
        # 创建异步任务列表
        tasks = []
        dataset_names = []
        
        for config in dataset_configs:
            dataset_name = config['dataset_name']
            dataset_names.append(dataset_name)
            
            task = self.download_dataset_async(
                dataset_name=dataset_name,
                subset_name=config.get('subset_name'),
                split=config.get('split', 'train'),
                download_mode=config.get('download_mode', 'force_redownload')
            )
            tasks.append(task)
        
        # 并发执行所有下载任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 构建结果字典
        download_results = {}
        for name, result in zip(dataset_names, results):
            if isinstance(result, Exception):
                logger.error(f"数据集 {name} 下载异常: {result}")
                download_results[name] = False
            else:
                download_results[name] = result
        
        # 统计下载结果
        success_count = sum(download_results.values())
        logger.info(f"批量下载完成: {success_count}/{len(dataset_configs)} 个数据集下载成功")
        
        return download_results


async def main() -> None:
    """
    主函数：演示数据集下载功能
    """
    # 初始化下载器
    downloader = DatasetDownloader()
    
    # 定义要下载的数据集配置
    dataset_configs = [
        {
            'dataset_name': 'modelscope/alpaca-gpt4-data-zh',  # 中文指令数据集
            'split': 'train'
        },
        {
            'dataset_name': 'AI-ModelScope/AdvertiseGen',  # 广告生成数据集
            'split': 'train'
        },
        {
            'dataset_name': 'ticoAg/Chinese-medical-dialogue',  # 中文医疗对话数据集
            'split': 'train'
        },
        {
            'dataset_name': 'AI-MO/NuminaMath-CoT',  # 数学推理思维链数据集
            'split': 'train'
        }
    ]
    
    logger.info("=== ModelScope 数据集下载工具 ===")
    
    # 单个数据集同步下载示例
    logger.info("\n1. 单个数据集同步下载示例:")
    success = downloader.download_dataset_sync(
        dataset_name="AI-MO/NuminaMath-CoT",
        split="train"
    )
    
    if success:
        logger.info("✅ 单个数据集下载完成")
    else:
        logger.error("❌ 单个数据集下载失败")
    
    # 批量数据集异步下载示例
    logger.info("\n2. 批量数据集异步下载示例:")
    batch_results = await downloader.batch_download_datasets(dataset_configs)
    
    # 显示下载结果
    logger.info("\n=== 下载结果汇总 ===")
    for dataset_name, result in batch_results.items():
        status = "✅ 成功" if result else "❌ 失败"
        logger.info(f"{dataset_name}: {status}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
