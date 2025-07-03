# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import datasets
import torch
import transformers
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers.trainer_utils import set_seed, get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


from configs import GRPOConfig
from rewards import (
    accuracy_reward,
    accuracy_reward_continuous,
    format_reward,
    format_reward_continuous,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
)
from utils.callbacks import get_callbacks
from x_grpo_trainer import XGRPOTrainer
from trl import ModelConfig, TrlParser, get_peft_config
from peft import LoraConfig, PeftModel, get_peft_model


logger = logging.getLogger(__name__)

import wandb


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project


@dataclass
class ScriptArguments:
    """
    基础脚本参数类，包含数据集相关配置
    """
    dataset_name: str = field(
        default="FreedomIntelligence/medical-o1-verifiable-problem",
        metadata={"help": "训练数据集名称"}
    )
    dataset_config: str = field(
        default="default",
        metadata={"help": "数据集配置名称"}
    )
    dataset_configs: list[str] = field(
        default_factory=lambda: ["train"],
        metadata={"help": "数据集配置列表"}
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "训练数据集分割"}
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "测试数据集分割"}
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'accuracy_continuous', 'format', 'format_continuous', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    script_num_iterations: int = field(
        default=1,
        metadata={"help": "multi-step new/old policy ratio iteration"},
    )



SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # align the dataset
    if script_args.dataset_name == "FreedomIntelligence/medical-o1-verifiable-problem":
        dataset = dataset.rename_columns({
            "Open-ended Verifiable Question": "problem",
            "Ground-True Answer": "solution"
        })
    elif "mmlu" in script_args.dataset_name.lower():
        # MMLU数据集的字段映射
        dataset = dataset.rename_columns({
            "question": "problem",
            "answer": "solution"
        })
    elif "OpenThoughts-114k" in script_args.dataset_name or "open-thoughts" in script_args.dataset_name.lower():
        # OpenThoughts-114k数据集处理 - 从conversations字段中提取问题和答案
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
        
        dataset = dataset.map(extract_conversation_content)
    elif "Bespoke-Stratos" in script_args.dataset_name or "bespoke-stratos" in script_args.dataset_name.lower():
        # Bespoke-Stratos-17k数据集处理 - 从conversations字段中提取问题和答案
        def extract_bespoke_conversation_content(example):
            """
            从Bespoke-Stratos数据集的conversations格式中提取用户问题和助手答案
            
            Parameters
            ----------
            example : dict
                数据集样本，包含 'conversations' 字段
                
            Returns
            -------
            dict
                包含 'problem' 和 'solution' 字段的字典
            """
            conversations = example.get("conversations", [])
            result = {"problem": "", "solution": ""}
            
            if conversations:
                # 查找用户消息作为问题
                for conv in conversations:
                    if conv.get("from") in ["user", "human"]:
                        result["problem"] = conv.get("value", "")
                        break
                
                # 查找助手消息作为答案
                for conv in conversations:
                    if conv.get("from") in ["assistant", "gpt"]:
                        result["solution"] = conv.get("value", "")
                        break
                
                # 如果没有找到用户消息，使用第一个消息作为问题
                if not result["problem"] and conversations:
                    result["problem"] = conversations[0].get("value", "")
                
                # 如果没有找到助手消息，使用最后一个消息作为答案  
                if not result["solution"] and len(conversations) > 1:
                    result["solution"] = conversations[-1].get("value", "")
            
            return result
        
        dataset = dataset.map(extract_bespoke_conversation_content)
    elif "generated_x_r1_dataset" in script_args.dataset_name:
        # generated_x_r1_dataset数据集处理 - 从conversations字段中提取问题和答案
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
        
        dataset = dataset.map(extract_conversation_content)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "accuracy_continuous": accuracy_reward_continuous,
        "format": format_reward,
        "format_continuous": format_reward_continuous,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    
    # 安全地处理数据集列删除 - 改进的类型检查
    try:
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            # 处理DatasetDict类型
            for split_name in list(dataset.keys()):
                split_dataset = dataset[split_name]
                if hasattr(split_dataset, 'column_names') and split_dataset.column_names is not None:
                    if "messages" in split_dataset.column_names:
                        dataset[split_name] = split_dataset.remove_columns("messages")
        elif isinstance(dataset, (Dataset, IterableDataset)):
            # 处理单个Dataset类型
            if hasattr(dataset, 'column_names') and dataset.column_names is not None:
                if "messages" in dataset.column_names:
                    dataset = dataset.remove_columns("messages")
    except Exception as e:
        logger.warning(f"Unable to remove 'messages' column: {e}")


    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    training_args.gradient_checkpointing = False
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # model = AutoModelForCausalLM.from_pretrained(**model_kwargs, pretrained_model_name_or_path = model_args.model_name_or_path)
    training_args.model_init_kwargs = model_kwargs
    # peft_config=get_peft_config(model_args)
    # print(peft_config)
    # if peft_config not None:
    #     model = get_peft_model(model, peft_config)
    # print(model)


    #############################
    # Initialize the XGRPO trainer
    #############################
    
    # 安全获取数据集分割
    def get_dataset_split(dataset, split_name: str):
        """安全地获取数据集分割"""
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            return dataset[split_name]
        elif isinstance(dataset, (Dataset, IterableDataset)):
            # 对于单个数据集，假设就是训练集
            if split_name == script_args.dataset_train_split:
                return dataset
            else:
                return None
        else:
            return None
    
    train_dataset = get_dataset_split(dataset, script_args.dataset_train_split)
    eval_dataset = get_dataset_split(dataset, script_args.dataset_test_split) if training_args.eval_strategy != "no" else None
    
    trainer = XGRPOTrainer(
        model=model_args.model_name_or_path,
        # model = model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args), # LoRA parameter
        callbacks=get_callbacks(training_args, model_args),
    )

    print(trainer)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    
    # 安全计算训练样本数量
    try:
        if train_dataset is not None and hasattr(train_dataset, '__len__'):
            # 安全地计算数据集长度，避免类型检查错误
            train_samples = train_dataset.__len__()  # type: ignore
            metrics["train_samples"] = train_samples
        else:
            logger.warning("Unable to calculate train_samples length - dataset doesn't support __len__")
    except (TypeError, AttributeError):
        logger.warning("Unable to calculate train_samples length")
        
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["X-R1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        try:
            if hasattr(trainer, 'model') and trainer.model is not None:
                if hasattr(trainer.model, 'config') and trainer.model.config is not None:
                    trainer.model.config.use_cache = True
                    trainer.model.config.save_pretrained(training_args.output_dir)
        except Exception as e:
            logger.warning(f"Unable to save model config: {e}")

    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    # 尝试不同的TrlParser参数格式
    try:
        parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))  # type: ignore
    except Exception:
        # 如果元组格式不行，尝试列表格式
        parser = TrlParser([GRPOScriptArguments, GRPOConfig, ModelConfig])  # type: ignore
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
