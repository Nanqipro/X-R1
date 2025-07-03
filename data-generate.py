#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集生成脚本 - 调用硅基流动API生成X-R1训练数据集

功能描述
-------
此脚本从A-data.jsonl读取题目，调用硅基流动(SiliconFlow)API生成答案，
并转换为X-R1框架可用的Bespoke-Stratos格式数据集。

使用要求
-------
- Python >= 3.8
- 需要安装: requests, tqdm, datasets
- 需要有效的硅基流动 API密钥

数据格式
-------
输出格式符合X-R1框架要求，包含：
- system: 系统提示词
- conversations: 对话列表，包含user和assistant的交互

作者: AI Assistant
版本: 1.0.0
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import requests
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import os
import re
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API配置 - 从环境变量获取密钥，确保安全性（使用硅基流动API）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("SILICONFLOW_API_KEY environment variable must be set")

SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 系统提示词配置
SYSTEM_PROMPTS = {
    "code-generate": """你是一位经验丰富的Python编程专家。请根据给定的函数签名和需求描述，编写完整的函数实现。

要求：
1. 代码必须符合Python规范，包含完整的类型注解
2. 提供详细的中文注释说明算法思路
3. 确保代码效率和可读性
4. 处理边界情况和异常情况
5. 遵循题目中的约束条件

输出格式要求：
请严格按照以下格式输出，包含思考过程和最终答案：

<think>
[在这里详细描述你的思考过程，包括：
- 问题分析和理解
- 算法选择和设计思路
- 实现步骤和注意事项
- 时间复杂度和空间复杂度分析]
</think>

<answer>
[在这里提供完整的Python代码实现，包含详细注释]
</answer>""",

    "math": """你是一位数学专家。请仔细分析数学问题，提供详细的解题步骤和最终答案。

要求：
1. 详细展示每个计算步骤
2. 说明使用的数学公式和定理
3. 确保计算过程准确无误
4. 最终答案用明确的格式表示
5. 如需要，提供多种解法

输出格式要求：
请严格按照以下格式输出，包含思考过程和最终答案：

<think>
[在这里详细描述你的思考过程，包括：
- 题目理解和分析
- 解题思路和方法选择
- 详细的计算步骤
- 公式和定理的应用]
</think>

<answer>
[在这里提供最终答案，格式清晰明确]
</answer>""",

    "choice": """你是一位逻辑分析专家。请仔细分析选择题，运用严密的逻辑推理得出正确答案。

要求：
1. 逐一分析每个选项的合理性
2. 说明选择或排除某选项的原因
3. 运用相关的知识点和理论
4. 确保推理过程逻辑严密
5. 明确指出正确答案及理由

输出格式要求：
请严格按照以下格式输出，包含思考过程和最终答案：

<think>
[在这里详细描述你的思考过程，包括：
- 题目理解和关键信息提取
- 各选项的逐一分析
- 相关知识点的应用
- 逻辑推理过程]
</think>

<answer>
[在这里明确指出正确答案，如：答案是A/B/C/D，并简述主要理由]
</answer>""",

    "generic-generate": """你是一位知识渊博的AI助手。请根据问题提供准确、详细的回答。

要求：
1. 回答内容准确可靠，有理有据
2. 语言表达清晰简洁
3. 如涉及计算，展示详细过程
4. 提供相关背景知识和解释
5. 确保回答完整性和实用性

输出格式要求：
请严格按照以下格式输出，包含思考过程和最终答案：

<think>
[在这里详细描述你的思考过程，包括：
- 问题理解和分析
- 相关知识点梳理
- 逻辑推理过程
- 答案组织思路]
</think>

<answer>
[在这里提供完整、准确的答案]
</answer>"""
}

class SiliconFlowAPIClient:
    """
    硅基流动 API客户端
    
    用于调用硅基流动(SiliconFlow) API生成回答
    
    Attributes
    ----------
    api_key : str
        硅基流动 API密钥
    api_url : str
        API端点URL
    headers : Dict[str, str]
        HTTP请求头
    """
    
    def __init__(self, api_key: str, api_url: str = SILICONFLOW_API_URL) -> None:
        """
        初始化API客户端
        
        Parameters
        ----------
        api_key : str
            DeepSeek API密钥
        api_url : str
            API端点URL
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         max_tokens: int = 2048, temperature: float = 0.1) -> Optional[str]:
        """
        调用API生成回答
        
        Parameters
        ----------
        prompt : str
            用户输入的问题
        system_prompt : str
            系统提示词
        max_tokens : int
            最大token数量
        temperature : float
            生成温度参数
            
        Returns
        -------
        Optional[str]
            生成的回答，失败时返回None
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": "deepseek-ai/DeepSeek-R1",  # 硅基流动支持的模型名称
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            # 设置合理的超时时间
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=120  # 120秒超时
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                if content and content.strip():  # 确保内容不为空
                    return content
                else:
                    logger.warning("API返回内容为空")
                    return None
            else:
                logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求异常: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"API响应格式错误: {e}")
            return None
    
    def validate_and_fix_response_format(self, response: str) -> Optional[str]:
        """
        验证并修正API响应格式，确保包含思考过程和答案
        
        现在改为：如果格式符合要求就返回原格式，否则直接返回原始响应
        
        Parameters
        ----------
        response : str
            API返回的原始响应
            
        Returns
        -------
        Optional[str]
            格式化后的响应或原始响应，只有在响应为空时才返回None
        """
        if not response:
            return None
        
        # 检查是否已经包含正确格式
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        if re.search(pattern, response, re.DOTALL):
            logger.info("响应格式符合要求，直接使用")
            return response
        
        # 格式不符合要求，但直接使用原始响应而不进行格式化
        logger.info("响应格式不符合要求，但保留原始响应")
        return response

class DatasetGenerator:
    """
    数据集生成器
    
    负责读取原始数据，调用API生成答案，并转换为目标格式
    
    Attributes
    ----------
    api_client : SiliconFlowAPIClient
        API客户端实例
    success_count : int
        成功生成的数据项数量
    failure_count : int
        失败的数据项数量
    """
    
    def __init__(self, api_client: SiliconFlowAPIClient) -> None:
        """
        初始化数据集生成器
        
        Parameters
        ----------
        api_client : SiliconFlowAPIClient
            API客户端实例
        """
        self.api_client = api_client
        self.success_count = 0
        self.failure_count = 0
    
    def load_source_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载源数据文件
        
        Parameters
        ----------
        file_path : str
            源数据文件路径
            
        Returns
        -------
        List[Dict[str, Any]]
            加载的数据列表
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析错误: {e}")
                        continue
            logger.info(f"成功加载 {len(data)} 条数据")
            return data
        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return []
        except Exception as e:
            logger.error(f"加载数据文件时发生错误: {e}")
            return []
    
    def convert_to_x_r1_format(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        转换为X-R1框架所需的数据格式
        
        Parameters
        ----------
        item : Dict[str, Any]
            原始数据项
        response : str
            API生成的回答
            
        Returns
        -------
        Dict[str, Any]
            转换后的数据格式
        """
        # 获取问题类型对应的系统提示词
        question_type = item.get('type', 'generic-generate')
        system_prompt = SYSTEM_PROMPTS.get(question_type, SYSTEM_PROMPTS['generic-generate'])
        
        # 构造X-R1格式的数据
        x_r1_data = {
            "system": system_prompt,
            "conversations": [
                {
                    "from": "user",
                    "value": item['prompt']
                },
                {
                    "from": "assistant", 
                    "value": response
                }
            ]
        }
        
        return x_r1_data
    
    def generate_single_item(self, item: Dict[str, Any], retry_count: int = 3) -> Optional[Dict[str, Any]]:
        """
        生成单个数据项
        
        Parameters
        ----------
        item : Dict[str, Any]
            原始数据项
        retry_count : int
            重试次数
            
        Returns
        -------
        Optional[Dict[str, Any]]
            生成的数据项，失败时返回None
        """
        question_type = item.get('type', 'generic-generate')
        system_prompt = SYSTEM_PROMPTS.get(question_type, SYSTEM_PROMPTS['generic-generate'])
        
        for attempt in range(retry_count):
            try:
                response = self.api_client.generate_response(
                    prompt=item['prompt'],
                    system_prompt=system_prompt,
                    max_tokens=2048,
                    temperature=0.1
                )
                
                if response:
                    # 验证并修正响应格式
                    formatted_response = self.api_client.validate_and_fix_response_format(response)
                    if formatted_response:
                        self.success_count += 1
                        return self.convert_to_x_r1_format(item, formatted_response)
                    else:
                        logger.warning(f"响应格式修正失败，重试中... (第{attempt + 1}次)")
                        time.sleep(2)
                else:
                    logger.warning(f"API返回空响应，重试中... (第{attempt + 1}次)")
                    time.sleep(2)  # 短暂延迟后重试
                    
            except Exception as e:
                logger.error(f"生成数据项时发生错误: {e}")
                time.sleep(2)
        
        self.failure_count += 1
        logger.error(f"处理数据项失败，ID: {item.get('id', 'unknown')}")
        return None
    
    def generate_dataset(self, source_file: str, output_file: str, 
                        batch_size: int = 10, delay: float = 1.0) -> bool:
        """
        生成完整数据集
        
        Parameters
        ----------
        source_file : str
            源数据文件路径
        output_file : str
            输出文件路径
        batch_size : int
            批处理大小
        delay : float
            请求间延迟时间（秒）
            
        Returns
        -------
        bool
            是否成功生成数据集
        """
        # 加载源数据
        source_data = self.load_source_data(source_file)
        if not source_data:
            return False
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化计数器
        self.success_count = 0
        self.failure_count = 0
        generated_data = []
        
        logger.info(f"开始生成数据集，共 {len(source_data)} 个项目")
        
        # 使用进度条处理数据
        with tqdm(total=len(source_data), desc="生成数据集") as pbar:
            for i, item in enumerate(source_data):
                result = self.generate_single_item(item)
                
                if result:
                    generated_data.append(result)
                
                # 更新进度条
                pbar.set_postfix({
                    'success': self.success_count,
                    'failed': self.failure_count
                })
                pbar.update(1)
                
                # 控制请求频率
                if i < len(source_data) - 1:  # 最后一个请求不需要延迟
                    time.sleep(delay)
                
                # 定期保存中间结果
                if (i + 1) % batch_size == 0:
                    self._save_intermediate_results(generated_data, output_file)
        
        # 保存最终结果
        success = self._save_final_results(generated_data, output_file)
        
        # 打印统计信息
        self._print_statistics(len(source_data))
        
        return success
    
    def _save_intermediate_results(self, data: List[Dict[str, Any]], output_file: str) -> None:
        """
        保存中间结果
        
        Parameters
        ----------
        data : List[Dict]
            当前生成的数据
        output_file : str
            输出文件路径
        """
        try:
            temp_file = f"{output_file}.temp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"已保存中间结果: {len(data)} 条数据")
        except Exception as e:
            logger.error(f"保存中间结果失败: {e}")
    
    def _save_final_results(self, data: List[Dict[str, Any]], output_file: str) -> bool:
        """
        保存最终结果
        
        Parameters
        ----------
        data : List[Dict]
            生成的数据
        output_file : str
            输出文件路径
            
        Returns
        -------
        bool
            是否保存成功
        """
        try:
            # 保存为JSONL格式（与A-data.jsonl格式保持一致）
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            # 同时保存为JSON格式（便于查看）
            json_file = output_file.replace('.jsonl', '.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据集生成完成，保存至: {output_file}")
            logger.info(f"同时保存JSON格式至: {json_file}")
            
            # 删除临时文件
            temp_file = f"{output_file}.temp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return True
            
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")
            return False
    
    def _print_statistics(self, total_items: int) -> None:
        """
        打印统计信息
        
        Parameters
        ----------
        total_items : int
            总数据项数量
        """
        success_rate = (self.success_count / total_items) * 100 if total_items > 0 else 0
        
        logger.info("=" * 50)
        logger.info("数据集生成统计:")
        logger.info(f"总项目数: {total_items}")
        logger.info(f"成功生成: {self.success_count}")
        logger.info(f"失败数量: {self.failure_count}")
        logger.info(f"成功率: {success_rate:.2f}%")
        logger.info("=" * 50)

def main() -> None:
    """
    主函数 - 解析命令行参数并执行数据集生成
    
    该函数负责初始化API客户端、解析命令行参数并执行完整的数据集生成流程
    """
    parser = argparse.ArgumentParser(description="生成X-R1训练数据集")
    parser.add_argument("--input", "-i", default="A-data/A-data.jsonl", 
                       help="输入数据文件路径")
    parser.add_argument("--output", "-o", default="./LLM-models-datasets/generated_x_r1_dataset03/generated_x_r1_dataset.jsonl",
                       help="输出数据文件路径")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                       help="批处理大小")
    parser.add_argument("--delay", "-d", type=float, default=3.0,  # 从1.0增加到3.0秒
                       help="请求间延迟时间（秒）- 避免API频率限制")
    parser.add_argument("--api-key", "-k", default=SILICONFLOW_API_KEY,
                       help="硅基流动 API密钥")
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 初始化API客户端
    api_client = SiliconFlowAPIClient(args.api_key)
    
    # 初始化数据集生成器
    generator = DatasetGenerator(api_client)
    
    # 生成数据集
    logger.info("开始数据集生成任务...")
    logger.info(f"使用硅基流动API，延迟时间: {args.delay}秒")
    start_time = datetime.now()
    
    success = generator.generate_dataset(
        source_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        delay=args.delay
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    if success:
        logger.info(f"数据集生成成功! 耗时: {duration}")
    else:
        logger.error("数据集生成失败!")

if __name__ == "__main__":
    main() 