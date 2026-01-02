#!/usr/bin/env python3
"""
自动生成的合成数据验证脚本（拒绝采样）

任务: Copa
训练方法: mezo
验证模型: gpt-4o
Field to rephrase: premise
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

# ⭐ 尝试从validation_checkpoints加载自动生成的few-shot
# （由annotate_samples.py生成）
VALIDATION_FEWSHOT_EXAMPLES = []
try:
    import sys
    from pathlib import Path
    checkpoint_file = Path(__file__).parent.parent / "validation_checkpoints" / "validation_fewshot.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            fewshot_data = json.load(f)
            VALIDATION_FEWSHOT_EXAMPLES = fewshot_data.get('examples', [])
        print(f"✓ 加载了 {len(VALIDATION_FEWSHOT_EXAMPLES)} 个自动生成的validation few-shot examples")
except Exception as e:
    print(f"⚠️  未找到自动生成的few-shot，将使用配置文件中的few-shot: {e}")

def generate_validation_prompt(original_premise, original_choice1, original_choice2, original_question, original_label, rephrased_premise):
    """生成验证提示词"""

    # ⭐ 构建few-shot文本
    fewshot_text = ""

    # 优先使用自动生成的few-shot（来自样本21-40）
    if len(VALIDATION_FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(VALIDATION_FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {i}:\n"
            fewshot_text += f"Original premise: {ex.get('original_premise', 'N/A')}\n"
            fewshot_text += f"Rephrased premise: {ex.get('rephrased_premise', 'N/A')}\n"
            # 添加其他字段
            for key in ex:
                if not key.startswith('original_') and not key.startswith('rephrased_') and key != 'evaluation':
                    fewshot_text += f"{key}: {ex[key]}\n"
            fewshot_text += f"Evaluation: {ex.get('evaluation', 'same')}\n\n"
    else:
        # 备用：使用配置文件中手动提供的few-shot
        manual_examples = [{'original_premise': 'My body cast a shadow over the grass.', 'rephrased_premise': 'A shadow from my body fell across the grass.', 'choice1': 'The sun was rising.', 'choice2': 'The grass was cut.', 'question': 'cause', 'label': 0, 'evaluation': 'same'}, {'original_premise': "The woman tolerated her friend's difficult behavior.", 'rephrased_premise': "The woman accepted her friend's challenging conduct.", 'choice1': 'The woman knew her friend was going through a hard time.', 'choice2': 'The woman felt that her friend took advantage of her kindness.', 'question': 'cause', 'label': 0, 'evaluation': 'same'}, {'original_premise': 'The women met for coffee.', 'rephrased_premise': 'The two women decided to gather at a café.', 'choice1': 'The cafe reopened in a new location.', 'choice2': 'They wanted to catch up with each other.', 'question': 'cause', 'label': 1, 'evaluation': 'same'}]
        for i, ex in enumerate(manual_examples, 1):
            if isinstance(ex, dict):
                fewshot_text += f"Example {i}:\n"
                for k, v in ex.items():
                    fewshot_text += f"{k}: {v}\n"
                fewshot_text += "\n"

    # ⭐ 原始prompt模板
    prompt_template = """\
Task: Verify if the rephrased premise maintains consistency with the correct answer choice.

{{VALIDATION_FEWSHOT}}

Original premise: "{original_premise}"
Rephrased premise: "{rephrased_premise}"
Choice 1: "{original_choice1}"
Choice 2: "{original_choice2}"
Question: "{original_question}"
Correct answer: "{original_choice1 if original_label == 0 else original_choice2}"

Output [same/not the same]:

"""

    # ⭐ 替换{{VALIDATION_FEWSHOT}}占位符
    prompt = prompt_template.replace("{{VALIDATION_FEWSHOT}}", fewshot_text)

    # ⭐ 构建字段字典用于format
    format_dict = {}
    for field in ['premise', 'choice1', 'choice2', 'question', 'label']:
        format_dict[f'original_{field}'] = locals().get(f'original_{field}', '')
    format_dict['rephrased_premise'] = locals().get('rephrased_premise', '')

    # ⭐ 替换字段值
    return prompt.format(**format_dict)
"""

# 加载原始数据
original_data = []
with open("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa/copa_train.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        original_data.append(json.loads(line.strip()))

# 加载合成数据
# 🆕 从数据集子目录读取
dataset_dir = os.path.join("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o", "Copa")
synthetic_data = []
synthetic_file = os.path.join(dataset_dir, "copa_train.jsonl")
with open(synthetic_file, 'r', encoding='utf-8') as f:
    for line in f:
        synthetic_data.append(json.loads(line.strip()))

print(f"原始数据: {len(original_data)} 条")
print(f"合成数据: {len(synthetic_data)} 条")

if len(original_data) != len(synthetic_data):
    print("⚠ 警告: 数据量不匹配!")

# 准备输出（临时文件）
temp_output_file = os.path.join(dataset_dir, "copa_train_validated.jsonl")
out_file = open(temp_output_file, "w", encoding='utf-8')

correct_count = 0
total_count = 0

# 验证每条数据
for i in tqdm(range(min(len(original_data), len(synthetic_data)))):
    original = original_data[i]
    synthetic = synthetic_data[i]

    # 🔴 排除样本21-40（索引20-39）
    # 这些样本用作judger的few-shot examples，不应被judger验证（避免数据泄露）
    if 20 <= i < 40:
        # 直接使用合成数据，不经过judger验证
        out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\n")
        correct_count += 1
        total_count += 1
        out_file.flush()
        continue

    # 构造验证提示词
    prompt_args = {}
    for field in ['premise', 'choice1', 'choice2', 'question', 'label']:
        prompt_args[f'original_{field}'] = original[field]
    prompt_args['rephrased_premise'] = synthetic['premise']

    prompt = generate_validation_prompt(**prompt_args)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful judge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        result = response.choices[0].message.content.strip().lower()

        # 判断是否通过验证
        if 'not the same' in result or 'not same' in result:
            # 验证失败，使用原始数据
            out_file.write(json.dumps(original, ensure_ascii=False) + "\n")
        else:
            # 验证成功，使用合成数据
            out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\n")
            correct_count += 1

        total_count += 1
        out_file.flush()

    except Exception as e:
        print(f"\n验证第 {i} 条数据时出错: {e}")
        # 出错时使用原始数据
        out_file.write(json.dumps(original, ensure_ascii=False) + "\n")
        total_count += 1
        out_file.flush()

out_file.close()

accuracy = correct_count / total_count if total_count > 0 else 0
print(f"\n验证完成!")
print(f"通过率: {correct_count}/{total_count} = {accuracy:.2%}")
print(f"临时输出文件: {temp_output_file}")

# 🆕 最终化数据集：重命名validated文件 + 复制validation/test
print("\n最终化数据集...")
import shutil

# 1. 将validated文件重命名为正式的train文件
final_train_file = os.path.join(dataset_dir, "copa_train.jsonl")
if os.path.exists(final_train_file):
    os.remove(final_train_file)  # 删除原始的未验证文件
shutil.move(temp_output_file, final_train_file)
print(f"✓ 训练集: {final_train_file}")

# 2. 复制validation和test文件from原始数据集
original_dir = "/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa"
files_config = {'train': 'copa_train.jsonl', 'validation': 'copa_validation.jsonl', 'test': 'copa_test.jsonl'}

# 复制validation文件
if 'validation' in files_config:
    val_file = files_config['validation']
    src_val = os.path.join(original_dir, val_file)
    dst_val = os.path.join(dataset_dir, val_file)
    if os.path.exists(src_val):
        shutil.copy2(src_val, dst_val)
        print(f"✓ 验证集: {dst_val}")
    else:
        print(f"⚠  警告: 验证集文件不存在: {src_val}")

# 复制test文件（如果有）
if 'test' in files_config:
    test_file = files_config['test']
    src_test = os.path.join(original_dir, test_file)
    dst_test = os.path.join(dataset_dir, test_file)
    if os.path.exists(src_test):
        shutil.copy2(src_test, dst_test)
        print(f"✓ 测试集: {dst_test}")

print(f"\n✅ 数据集已完成！可用于MeZO训练：")
print(f"   python PromptZO/MeZO/large_models/run.py --task {dataset_dir}")
