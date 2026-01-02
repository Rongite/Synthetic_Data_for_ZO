#!/usr/bin/env python3
"""
自动生成的合成数据生成脚本

任务: Copa
训练方法: mezo
生成模型: gpt-4o
策略: rest
生成时间: 2025-12-30 18:43:11
"""

from tqdm import tqdm
import os
import json
from openai import OpenAI

# 配置
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api2.aigcbest.top/v1")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    timeout=120
)

# ⭐ Few-shot examples placeholder (will be injected by review_top20.py)
# FEWSHOT_EXAMPLES = [...]

def generate_prompt(premise, choice1, choice2, question, label):
    """生成改写提示词"""

    # ⭐ 构建few-shot文本（如果存在FEWSHOT_EXAMPLES）
    fewshot_text = ""
    if 'FEWSHOT_EXAMPLES' in globals() and len(FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {i}:\n"
            fewshot_text += f"Original premise: {ex['original']}\n"
            fewshot_text += f"Rephrased premise: {ex['rephrased']}\n"
            # 添加其他字段作为上下文
            for key in ex:
                if key not in ['original', 'rephrased']:
                    fewshot_text += f"{key}: {ex[key]}\n"
            fewshot_text += "\n"

    # ⭐ 原始prompt模板
    prompt_template = """\
You are tasked with rephrasing the given premise while preserving its original meaning. Your goal is to create rephrased data optimized for enhancing gradient estimation in training with a memory-efficient zeroth-order optimizer (MeZO).

{{REPHRASE_FEWSHOT}}

### Key Requirements for Premise Rephrasing:
1. **Task-Specific Context**:
   - The **Copa dataset** focuses on **causal reasoning**, where the goal is to determine the **cause or effect** of a given premise.
   - The rephrased **premise** must **not alter the logical relationship** with the correct choice.

2. **Consistency with Correct Answer**:
   - Ensure that the rephrased premise maintains the same correct answer as the original.
   - The correct answer is: Choice {label + 1}: "{choice1 if label == 0 else choice2}"

3. **Optimized for MeZO Training**:
   - **Enhance Gradient Sensitivity**: Create clear semantic boundaries to increase gradient sensitivity.
   - **Focus on Memory Efficiency**: Reduce redundancy, keep sentences concise.
   - **Robustness to Data Sparsity**: Ensure essential information is preserved.
   - **Non-Differentiable Optimization Readiness**: Create clear impacts on performance metrics.

4. **Maintain Neutral Stance**:
   - Do not explicitly indicate which choice is correct.
   - The rephrased premise should require reasoning to determine the answer.

5. **High-Quality Data Generation**:
   - Produce natural, fluent, and coherent text.
   - Use paraphrasing, synonyms, or restructuring to diversify data.

### Your Task:
Rephrase the following premise while ensuring it remains consistent with the correct answer.

**Original premise**: "{premise}"
**Choice 1**: "{choice1}"
**Choice 2**: "{choice2}"
**Question**: "{question}"
**Correct answer**: "{choice1 if label == 0 else choice2}"

**Directly output only one rephrased premise** without any other characters or explanatory statements:

"""

    # ⭐ 替换{{REPHRASE_FEWSHOT}}占位符
    prompt = prompt_template.replace("{{REPHRASE_FEWSHOT}}", fewshot_text)

    # ⭐ 替换字段值（使用.format()）
    return prompt.format(premise=premise, choice1=choice1, choice2=choice2, question=question, label=label)

# 加载原始数据
data = []
input_file = "/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa/copa_train.jsonl"
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(f"加载了 {len(data)} 条原始数据")

# 准备输出
# 🆕 创建数据集子目录
dataset_dir = os.path.join("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o", "Copa")
os.makedirs(dataset_dir, exist_ok=True)

output_file = os.path.join(dataset_dir, "copa_train_rest.jsonl")
out_file = open(output_file, "w", encoding='utf-8')

print(f"输出文件: {output_file}")

# 处理数据
progress = 0
for i in tqdm(range(len(data))):
    progress += 1
    if progress <= 20:
        continue


    # 构造提示词
    prompt_args = {"premise": data[i]["premise"], "choice1": data[i]["choice1"], "choice2": data[i]["choice2"], "question": data[i]["question"], "label": data[i]["label"]}
    prompt = generate_prompt(**prompt_args)

    # 调用 API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )

        # 提取结果
        rephrased_text = response.choices[0].message.content.strip()

        # 构造输出
        result = data[i].copy()
        result["premise"] = rephrased_text

        # 写入文件
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_file.flush()

    except Exception as e:
        print(f"\n处理第 {i} 条数据时出错: {e}")
        # 出错时使用原始数据
        out_file.write(json.dumps(data[i], ensure_ascii=False) + "\n")
        out_file.flush()

out_file.close()
print(f"\n完成! 输出: {output_file}")
