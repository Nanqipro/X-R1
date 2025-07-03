# X-R1 数据集生成工具

## 概述

该工具用于从A-data.jsonl文件读取题目，调用DeepSeek API生成高质量答案，并转换为X-R1框架可用的训练数据格式。

## 功能特点

- **多种题目类型支持**: 支持代码生成、数学题、选择题和通用问答
- **智能系统提示词**: 针对不同题目类型配置专门的系统提示词
- **推理过程格式**: 生成的回答严格按照 `<think>...</think><answer>...</answer>` 格式
- **错误恢复机制**: 自动重试失败的API请求，保证数据生成的稳定性
- **进度监控**: 实时显示生成进度和成功率
- **中间结果保存**: 定期保存中间结果，防止意外中断导致数据丢失
- **格式兼容**: 生成的数据完全兼容X-R1框架和Bespoke-Stratos格式

## 环境要求

### Python版本
- Python >= 3.8

### 依赖包
```bash
pip install requests tqdm datasets
```

或使用项目的requirements.txt：
```bash
pip install -r requirements.txt
```

### API密钥
需要有效的DeepSeek API密钥，已预配置在代码中：
- API Key: `sk-60e54dd882314d11b6dd43fe1bf55f11`
- API URL: `https://api.deepseek.com/v1/chat/completions`

## 使用方法

### 基本用法

```bash
# 使用默认参数运行（输出到指定目录）
python data-generate.py

# 指定输入和输出文件
python data-generate.py --input A-data/A-data.jsonl --output /custom/path/my_dataset.jsonl

# 调整请求参数
python data-generate.py --batch-size 5 --delay 2.0
```

### 命令行参数

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `--input` | `-i` | `A-data/A-data.jsonl` | 输入数据文件路径 |
| `--output` | `-o` | `/home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/generated_x_r1_dataset.jsonl` | 输出数据文件路径 |
| `--batch-size` | `-b` | `10` | 批处理大小（每处理多少条数据保存一次中间结果） |
| `--delay` | `-d` | `1.0` | 请求间延迟时间（秒） |
| `--api-key` | `-k` | 预设值 | DeepSeek API密钥 |

### 高级用法示例

```bash
# 大批量处理，增加延迟避免API限制
python data-generate.py \
    --input A-data/A-data.jsonl \
    --output /home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/large_dataset.jsonl \
    --batch-size 20 \
    --delay 1.5

# 处理自定义数据文件
python data-generate.py \
    --input custom_questions.jsonl \
    --output /home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/custom_dataset.jsonl \
    --api-key your_api_key_here
```

## 数据格式

### 输入格式（A-data.jsonl）
```json
{
    "id": "unique_id",
    "type": "code-generate|math|choice|generic-generate", 
    "prompt": "问题内容..."
}
```

### 输出格式（X-R1兼容）
```json
{
    "system": "系统提示词...",
    "conversations": [
        {
            "from": "user",
            "value": "问题内容..."
        },
        {
            "from": "assistant",
            "value": "<think>\n详细的思考过程...\n</think>\n\n<answer>\n最终答案...\n</answer>"
        }
    ]
}
```

## 推理过程格式要求

所有生成的回答都必须严格遵循以下格式：

```
<think>
[详细的思考过程，包括：
- 问题分析和理解
- 解题思路和方法选择
- 详细的推理或计算步骤
- 相关知识点的应用]
</think>

<answer>
[最终答案或解决方案]
</answer>
```

这种格式确保了：
- **思考过程透明**: 清晰展示推理链条
- **答案明确**: 最终结果容易提取
- **格式统一**: 便于后续处理和训练
- **质量保证**: 强制包含思考过程提高回答质量

## 题目类型和系统提示词

### 1. 代码生成 (code-generate)
专门针对Python编程题目，要求：
- 符合Python规范和类型注解
- 提供详细的中文注释
- 处理边界情况和异常
- 确保代码效率和可读性

### 2. 数学题 (math)
针对数学计算和证明题目，要求：
- 详细展示计算步骤
- 说明使用的公式和定理
- 确保计算准确性
- 提供多种解法（如适用）

### 3. 选择题 (choice)
针对逻辑推理和知识选择题，要求：
- 逐一分析各选项
- 说明选择或排除的理由
- 运用相关知识点
- 确保推理逻辑严密

### 4. 通用问答 (generic-generate)
针对一般性问题，要求：
- 回答准确可靠
- 语言清晰简洁
- 提供背景知识
- 确保完整性和实用性

## 监控和日志

### 日志文件
脚本运行时会生成 `dataset_generation.log` 文件，记录：
- API请求状态
- 数据处理进度
- 错误信息和警告
- 最终统计结果

### 实时监控
运行时会显示：
- 进度条和百分比
- 成功处理数量
- 失败处理数量
- 当前处理速度

### 统计信息
完成后会显示：
- 总处理项目数
- 成功生成数量
- 失败数量
- 成功率百分比
- 总耗时

## 错误处理

### 自动重试机制
- 默认重试3次
- 失败后延迟2秒再重试
- 记录详细错误信息

### 常见问题和解决方案

1. **API密钥无效**
   ```
   错误: API请求失败: 401 - Unauthorized
   解决: 检查API密钥是否正确
   ```

2. **网络连接问题**
   ```
   错误: API请求异常: Connection timeout
   解决: 检查网络连接，增加延迟时间
   ```

3. **API限流**
   ```
   错误: API请求失败: 429 - Too Many Requests
   解决: 增加 --delay 参数值
   ```

4. **文件权限问题**
   ```
   错误: 保存最终结果失败: Permission denied
   解决: 检查输出目录的写入权限
   ```

## 输出文件

脚本会生成以下文件：
- `{output_name}.jsonl`: 主要输出文件（JSONL格式）
- `{output_name}.json`: 便于查看的JSON格式文件
- `{output_name}.temp`: 临时文件（处理完成后自动删除）
- `dataset_generation.log`: 详细日志文件

## 性能优化建议

### API请求优化
- 根据API限制调整 `--delay` 参数
- 使用适当的 `--batch-size` 平衡性能和稳定性
- 监控API配额使用情况

### 资源管理
- 确保有足够的磁盘空间存储输出文件
- 监控内存使用，大文件处理时考虑分批处理
- 在稳定的网络环境下运行

### 数据质量
- 定期检查生成的数据质量
- 根据需要调整系统提示词
- 验证输出格式的正确性

## 示例运行输出

```
2024-12-19 10:30:15 - INFO - 成功加载 163 条数据
2024-12-19 10:30:15 - INFO - 开始生成数据集，共 163 个项目
生成数据集: 100%|██████████| 163/163 [02:45<00:00,  1.01s/it, success=158, failed=5]
2024-12-19 10:33:00 - INFO - 数据集生成完成，保存至: generated_x_r1_dataset.jsonl
2024-12-19 10:33:00 - INFO - 同时保存JSON格式至: generated_x_r1_dataset.json
==================================================
数据集生成统计:
总项目数: 163
成功生成: 158
失败数量: 5
成功率: 96.93%
==================================================
2024-12-19 10:33:00 - INFO - 数据集生成成功! 耗时: 0:02:45.123456
```

## 与X-R1框架集成

生成的数据集默认保存在X-R1框架的数据集目录中，可以直接用于训练：

1. **数据集已自动保存在正确位置**
   ```bash
   # 默认输出路径：/home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/generated_x_r1_dataset.jsonl
   # 数据集已经在LLM-models-datasets目录中
   ```

2. **修改训练配置**
   在训练配置文件中设置：
   ```yaml
   dataset_name: ./LLM-models-datasets/generated_x_r1_dataset.jsonl
   ```

3. **启动训练**
   ```bash
   python src/x_r1/grpo.py --config recipes/your_config.yaml
   ```

### 示例训练配置

创建一个新的配置文件 `recipes/custom_dataset_config.yaml`：

```yaml
# Model arguments
model_name_or_path: Qwen/Qwen2.5-1.5B
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Data training arguments
dataset_name: ./LLM-models-datasets/generated_x_r1_dataset.jsonl
dataset_configs:
- train

# GRPO trainer config
bf16: true
use_vllm: true
vllm_gpu_memory_utilization: 0.7
gradient_accumulation_steps: 8
learning_rate: 3.0e-06
max_prompt_length: 256
num_generations: 8
max_completion_length: 1024
num_train_epochs: 3
output_dir: output/X-R1-custom
per_device_train_batch_size: 2
```

## 验证生成的数据集

使用验证脚本检查数据集格式：

```bash
# 验证数据集格式
python test_dataset_format.py /home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/generated_x_r1_dataset.jsonl

# 详细验证（显示第一个样本）
python test_dataset_format.py /home/nanchang/ZJ/gitlocal/X-R1/LLM-models-datasets/generated_x_r1_dataset.jsonl --verbose
```

## 许可证

此工具遵循项目的开源许可证条款。

## 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 联系方式

如有问题，请在项目仓库中创建Issue。 