import os
import asyncio
from typing import Optional
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    ModelScope数据集下载器
    
    提供从ModelScope平台下载各种类型数据集的功能
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
            
            # 使用ModelScope数据集API下载到指定目录
            dataset = MsDataset.load(
                dataset_name=dataset_name,
                subset_name=subset_name,
                split=split,
                download_mode=getattr(DownloadMode, download_mode.upper(), DownloadMode.FORCE_REDOWNLOAD),
                cache_dir=local_path  # 指定缓存目录为我们的目标路径
            )
            
            logger.info(f"数据集已下载到: {local_path}")
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
            
            # 构建本地存储路径
            dataset_local_name = dataset_name.split('/')[-1]
            if subset_name:
                dataset_local_name += f"_{subset_name}"
            
            local_path = os.path.join(self.base_dir, dataset_local_name)
            
            # 使用ModelScope数据集API下载到指定目录
            dataset = MsDataset.load(
                dataset_name=dataset_name,
                subset_name=subset_name,
                split=split,
                cache_dir=local_path  # 指定缓存目录为我们的目标路径
            )
            
            logger.info(f"数据集已下载到: {local_path}")
            logger.info(f"✅ 数据集 {dataset_name} 下载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载数据集 {dataset_name} 时出错: {e}")
            return False


async def main() -> None:
    """
    主函数：演示数据集下载功能
    """
    # 初始化下载器
    downloader = DatasetDownloader()
    
    logger.info("=== ModelScope 数据集下载工具 ===")
    
    # 单个数据集同步下载示例
    logger.info("\n单个数据集下载示例:")
    success = downloader.download_dataset_sync(
        dataset_name="open-thoughts/OpenThoughts-114k",
        split="train"  # 使用实际存在的分割
    )
    
    if success:
        logger.info("✅ 数据集下载完成")
    else:
        logger.error("❌ 数据集下载失败")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
