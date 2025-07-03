import json
from vllm import LLM, SamplingParams
import os

class Infer:
    def __init__(self, model="./LLM-models-datasets/Qwen2.5-3B"):
        if not os.path.exists(model):
            print(f"Error: Model path '{model}' does not exist.")
            exit(1)
        
        self.llm = LLM(
    model=model,
    trust_remote_code=True,
    tokenizer_mode="slow"  # 添加这行
)
        self.sampling_params = {
            "choice": SamplingParams(max_tokens=1024, temperature=0.8, top_p=0.95),
            "code-generate": SamplingParams(n=3, max_tokens=2048, temperature=0.8, top_p=0.95),
            "generic-generate": SamplingParams(max_tokens=128, temperature=0.8, top_p=0.95),
            "math": SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
        }
        
        # 修复数学模板中的大括号转义问题
        self.prompt_templates = {
            "choice": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "code-generate": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n",
            "generic-generate": "You will be asked to read a passage and answer a question. Think step by step, then write a line of the form 'Answer: $ANSWER' at the end of your response.",
            "math": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \$ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n{Question}\n\nRemember to put your answer on its own line after 'Answer:', and indicate your final answer in boxed LaTeX. For example, if the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}}."
        }
        
    def escape_braces(self, text):
        """转义文本中的所有大括号，防止format解析错误"""
        return text.replace("{", "{{").replace("}", "}}")

    def infer(self, data_file="A-data.jsonl"):
        a_datas = []
        try:
            with open(data_file, 'r', encoding="utf-8") as f:
                for line in f:
                    a_data = json.loads(line)
                    a_datas.append(a_data)
        except FileNotFoundError:
            print(f"Error: Data file '{data_file}' not found.")
            return {"error": f"File not found: {data_file}"}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in '{data_file}'. Details: {e}")
            return {"error": f"Invalid JSON in {data_file}: {e}"}

        res = {"result": {"results": []}}

        for i, a_data in enumerate(a_datas):
            type_ = a_data.get("type")
            id_ = a_data.get("id")
            print(f"\n--- Processing entry {i+1} (ID: {id_}, Type: {type_}) ---")

            if type_ not in self.prompt_templates:
                print(f"WARNING: Unknown type '{type_}' for ID '{id_}'. Skipping.")
                continue

            template = self.prompt_templates[type_]
            prompt = ""
            
            try:
                # 对输入内容进行大括号转义
                if type_ == "choice":
                    choices = a_data["choices"]
                    escaped_choices = {k: self.escape_braces(v) for k, v in choices.items()}
                    prompt = template.format(
                        Question=self.escape_braces(a_data["prompt"]),
                        A=escaped_choices.get("A", ""),
                        B=escaped_choices.get("B", ""),
                        C=escaped_choices.get("C", ""),
                        D=escaped_choices.get("D", "")
                    )
                else:
                    # 对所有其他类型的内容进行转义
                    escaped_prompt = self.escape_braces(a_data["prompt"])
                    if type_ == "math":
                        prompt = template.format(Question=escaped_prompt)
                    else:  # code-generate 和 generic-generate
                        prompt = template + escaped_prompt

                print(f"DEBUG: Generated prompt for ID '{id_}':\n{prompt}\n")
                
                generated_text = []
                if not prompt.strip():
                    print(f"WARNING: Empty prompt for ID '{id_}'. Skipping.")
                    generated_text = "ERROR: Empty prompt"
                else:
                    outputs = self.llm.generate(prompt, self.sampling_params[type_])
                    for output in outputs:
                        for o in output.outputs:
                            generated_text.append(o.text)
                    generated_text = generated_text[0] if len(generated_text) == 1 else generated_text

                res["result"]["results"].append({"id": id_, "content": generated_text})

            except KeyError as ke:
                print(f"ERROR: Missing key in data for ID '{id_}': {ke}")
                res["result"]["results"].append({"id": id_, "content": f"ERROR: Missing data - {ke}"})
            except Exception as e:
                print(f"ERROR: Unexpected error for ID '{id_}': {e}")
                res["result"]["results"].append({"id": id_, "content": f"ERROR: {e}"})

        return res

if __name__ == "__main__":
    data_file_name = "./A-data/A-data.jsonl"
    
    # 使用合并后的模型路径
    merged_model_path = "./LLM-models-datasets/Qwen2.5-3B"
    
    # 检查合并后的模型是否存在，如果不存在则提示用户先合并
    if not os.path.exists(merged_model_path):
        print(f"❌ 错误: 合并后的模型路径不存在: {merged_model_path}")
        print("请先运行以下命令合并LoRA模型:")
        print("python merge_lora_model.py")
        print("或者手动指定路径:")
        print("python merge_lora_model.py --base ./LLM-models-datasets/Qwen2.5-3B --lora ./LLM-models-datasets/X-R1-3B-LoRA-Advanced-Fast --output ./LLM-models-datasets/Qwen2.5-3B")
        exit(1)
    
    infer = Infer(model=merged_model_path)
    res = infer.infer(data_file=data_file_name)
    
    output_file_name = "./A-data/res-qwen2.5-3b.json"
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n--- Inference complete. Results saved to '{output_file_name}' ---")