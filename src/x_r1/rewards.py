"""Reward functions for GRPO training."""

import re
import math
from typing import Dict, Callable
import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# Initialize SiliconFlow client with environment variable for security
# 从环境变量获取API密钥，确保安全性
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
if not siliconflow_api_key:
    raise ValueError("SILICONFLOW_API_KEY environment variable must be set")

client = OpenAI(
    api_key=siliconflow_api_key,
    base_url="https://api.siliconflow.cn/v1"
)

def normalize_text(text: str | None) -> str:
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Convert to string if not already (handles int, float, etc.)
    if not isinstance(text, str):
        text = str(text)
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text.strip()

def extract_answer(text: str | None) -> str:
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def evaluate_answer_similarity(answer: str, solution: str) -> float:
    """Use SiliconFlow API to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical answer evaluator. Compare the student's answer with the correct solution and output ONLY '1.0' if they match in meaning, or '0.0' if they don't match. No other output is allowed."
                },
                {
                    "role": "user",
                    "content": f"Student answer: {answer}\nCorrect solution: {solution}\nOutput only 1.0 or 0.0:"
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content
        if result:
            result = result.strip()
            return float(result)
        else:
            raise Exception("Empty response from API")
    except Exception as e:
        print(f"Error in SiliconFlow evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if normalize_text(answer) == normalize_text(solution) else 0.0

def accuracy_reward(completions: list[list[Dict[str, str]]], solution: list[str], **kwargs) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # Convert solution to string if it's not already (handles MMLU integer answers)
        sol_str = str(sol) if not isinstance(sol, str) else sol
        
        # First try latex parsing
        gold_parsed = parse(
            sol_str,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # print('latex gold parsed')
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
            # print('\nprompt:', prompt)
            print('-'*100)
            print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        else:
            # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
            answer_content = extract_answer(content)
            normalized_content = normalize_text(answer_content)
            normalized_solution = normalize_text(sol)
            reward = evaluate_answer_similarity(normalized_content, normalized_solution)
            print('-'*100)
            print('\nanswer_parsed:', normalized_content, '\ngold_parsed:', normalized_solution, '\nreward:', reward)
        rewards.append(reward)

    print('\naccuracy rewards:', rewards)

    return rewards


def accuracy_reward_continuous(completions: list[list[Dict[str, str]]], solution: list[str], **kwargs) -> list[float]:
    """改进的奖励函数，提供连续的奖励值而不是简单的0/1."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        # Convert solution to string if it's not already (handles MMLU integer answers)
        sol_str = str(sol) if not isinstance(sol, str) else sol
        
        # First try latex parsing
        gold_parsed = parse(
            sol_str,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            
            # 基础奖励：完全匹配得1.0，否则看部分匹配程度
            base_reward = float(verify(answer_parsed, gold_parsed))
            
            if base_reward == 0.0:
                # 如果不完全匹配，计算相似度奖励
                # 1. 检查是否包含正确答案的部分内容
                gold_str = str(gold_parsed).lower() if gold_parsed else ""
                answer_str = str(answer_parsed).lower() if answer_parsed else ""
                content_lower = content.lower()
                
                # 2. 部分匹配奖励
                if gold_str and gold_str in content_lower:
                    base_reward = 0.3  # 包含正确答案得30%奖励
                elif answer_str and len(answer_str) > 0:
                    # 3. 基于字符串相似度的奖励
                    common_chars = len(set(gold_str) & set(answer_str))
                    total_chars = len(set(gold_str) | set(answer_str))
                    if total_chars > 0:
                        similarity = common_chars / total_chars
                        base_reward = min(0.2, similarity * 0.2)  # 最多20%的相似度奖励
            
            # 4. 长度惩罚：过长的答案会降低奖励
            length_penalty = min(1.0, 500 / max(len(content), 100))  # 超过500字符开始惩罚
            final_reward = base_reward * length_penalty
            
            print('-'*100)
            print(f'\nanswer_parsed: {answer_parsed}')
            print(f'gold_parsed: {gold_parsed}')
            print(f'base_reward: {base_reward:.3f}, length_penalty: {length_penalty:.3f}, final_reward: {final_reward:.3f}')
            
            reward = final_reward
        else:
            # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
            answer_content = extract_answer(content)
            normalized_content = normalize_text(answer_content)
            normalized_solution = normalize_text(sol)
            
            # 使用API评估，但添加连续性
            base_reward = evaluate_answer_similarity(normalized_content, normalized_solution)
            
            # 添加部分匹配奖励
            if base_reward == 0.0 and normalized_solution:
                # 简单的包含关系检查
                solution_words = set(normalized_solution.split())
                answer_words = set(normalized_content.split())
                
                if solution_words and answer_words:
                    # 计算词汇重叠度
                    overlap = len(solution_words & answer_words)
                    total_words = len(solution_words)
                    if total_words > 0:
                        word_overlap_ratio = overlap / total_words
                        base_reward = min(0.3, word_overlap_ratio * 0.3)  # 最多30%的词汇重叠奖励
            
            # 长度惩罚
            length_penalty = min(1.0, 300 / max(len(content), 50))
            reward = base_reward * length_penalty
            
            print('-'*100)
            print(f'\nanswer_parsed: {normalized_content}')
            print(f'gold_parsed: {normalized_solution}')
            print(f'base_reward: {base_reward:.3f}, length_penalty: {length_penalty:.3f}, final_reward: {reward:.3f}')
        
        rewards.append(reward)

    print(f'\naccuracy rewards (continuous): {[f"{r:.3f}" for r in rewards]}')
    return rewards


def accuracy_answer_reward(completion: str, answer: str, **kwargs) -> float:
    """Reward function that checks if the completion is the same as the ground truth."""
    '''
    input is completion string, answer is extracted gold answer.
    '''
    gold_parsed = answer
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        reward = float(verify(answer_parsed, gold_parsed))
        print('-'*100)
        print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        return reward
    else:
        # Handle case when gold_parsed is empty
        print('-'*100)
        print('\nSkipping unparseable gold answer:', answer)
        return 0.0


def format_reward(completions: list[list[Dict[str, str]]], **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    print('-'*100)
    print('\nformat rewards:', rewards)
    return rewards


def format_reward_continuous(completions: list[list[Dict[str, str]]], **kwargs) -> list[float]:
    """连续版本的格式奖励函数，提供更细粒度的奖励."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        reward = 0.0
        
        # 检查完整格式
        full_pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        if re.match(full_pattern, content, re.DOTALL):
            reward = 1.0  # 完整格式得满分
        else:
            # 部分格式奖励
            has_think_start = bool(re.search(r"<think>", content))
            has_think_end = bool(re.search(r"</think>", content))
            has_answer_start = bool(re.search(r"<answer>", content))
            has_answer_end = bool(re.search(r"</answer>", content))
            
            # 根据包含的标签给予部分奖励
            partial_score = 0.0
            if has_think_start:
                partial_score += 0.2
            if has_think_end:
                partial_score += 0.2
            if has_answer_start:
                partial_score += 0.3
            if has_answer_end:
                partial_score += 0.3
                
            # 检查标签是否配对
            think_pairs = len(re.findall(r"<think>.*?</think>", content, re.DOTALL))
            answer_pairs = len(re.findall(r"<answer>.*?</answer>", content, re.DOTALL))
            
            if think_pairs > 0:
                partial_score += 0.1  # 有配对的think标签额外加分
            if answer_pairs > 0:
                partial_score += 0.1  # 有配对的answer标签额外加分
                
            # 检查标签顺序（think应该在answer之前）
            think_pos = content.find("<think>")
            answer_pos = content.find("<answer>")
            if think_pos != -1 and answer_pos != -1 and think_pos < answer_pos:
                partial_score += 0.1  # 正确顺序额外加分
                
            reward = min(0.9, partial_score)  # 部分匹配最多给90%奖励
        
        rewards.append(reward)
    
    print('-'*100)
    print(f'\nformat rewards (continuous): {[f"{r:.3f}" for r in rewards]}')
    return rewards


def reasoning_steps_reward(completions: list[list[Dict[str, str]]], **kwargs) -> list[float]:
    """Reward function that checks for clear step-by-step reasoning.
    
    检查清晰逐步推理的奖励函数
    
    Regex pattern:
        Step \\d+: - matches "Step 1:", "Step 2:", etc.
        ^\\d+\\. - matches numbered lists like "1.", "2.", etc. at start of line
        \\n- - matches bullet points with hyphens
        \\n\\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[list[Dict[str, str]]], solutions: list[str], **kwargs) -> list[float]:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        # Convert solution to string if it's not already (handles MMLU integer answers)
        sol_str = str(sol) if not isinstance(sol, str) else sol
        
        gold_parsed = parse(
            sol_str,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
) -> Callable[[list[list[Dict[str, str]]], list[str]], list[float]]:
    def cosine_scaled_reward(completions: list[list[Dict[str, str]]], solution: list[str], **kwargs) -> list[float]:
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float) -> Callable[[list[list[Dict[str, str]]]], list[float]]:
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int) -> zip:
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions: list[list[Dict[str, str]]], **kwargs) -> list[float]:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
