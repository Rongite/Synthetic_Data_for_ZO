#!/usr/bin/env python3
"""
Auto-generated synthetic data generation script

Task: Copa
Training Method: mezo
Generation Model: gpt-4o
Strategy: top20
Generation Time: 2025-12-30 18:43:11
"""

from tqdm import tqdm
import os
import json
from openai import OpenAI

# Configuration
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
    """Generate rephrasing prompt"""

    # ⭐ Build few-shot text（If existsFEWSHOT_EXAMPLES）
    fewshot_text = ""
    if 'FEWSHOT_EXAMPLES' in globals() and len(FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {i}:\n"
            fewshot_text += f"Original premise: {ex['original']}\n"
            fewshot_text += f"Rephrased premise: {ex['rephrased']}\n"
            # add other fieldsas as  up  down text
            for key in ex:
                if key not in ['original', 'rephrased']:
                    fewshot_text += f"{key}: {ex[key]}\n"
            fewshot_text += "\n"

    # ⭐ Original prompt template
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

    # ⭐ Replace {{REPHRASE_FEWSHOT}} placeholder
    prompt = prompt_template.replace("{{REPHRASE_FEWSHOT}}", fewshot_text)

    # ⭐ Replace field values（use.format()）
    return prompt.format(premise=premise, choice1=choice1, choice2=choice2, question=question, label=label)

# Loaded original data
data = []
input_file = "/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa/copa_train.jsonl"
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(f"Loaded {len(data)}  samplesoriginalData")

# Prepare output
# 🆕 CreateDatasetSubDirectory
dataset_dir = os.path.join("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o", "Copa")
os.makedirs(dataset_dir, exist_ok=True)

output_file = os.path.join(dataset_dir, "copa_train_top20.jsonl")
out_file = open(output_file, "w", encoding='utf-8')

print(f"outputFile: {output_file}")

# Process data
progress = 0
for i in tqdm(range(len(data))):
    progress += 1
    if progress > 20:
        break


    # ConstructPrompt
    prompt_args = {"premise": data[i]["premise"], "choice1": data[i]["choice1"], "choice2": data[i]["choice2"], "question": data[i]["question"], "label": data[i]["label"]}
    prompt = generate_prompt(**prompt_args)

    # Call API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )

        # Extract result
        rephrased_text = response.choices[0].message.content.strip()

        # Constructoutput
        result = data[i].copy()
        result["premise"] = rephrased_text

        # writeFile
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_file.flush()

    except Exception as e:
        print(f"\nProcessline {i}  samplesDataError when: {e}")
        # Use original data on error
        out_file.write(json.dumps(data[i], ensure_ascii=False) + "\n")
        out_file.flush()

out_file.close()
print(f"\nComplete! output: {output_file}")
