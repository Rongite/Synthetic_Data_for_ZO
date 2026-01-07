from tqdm import tqdm
import os
import ast
import json
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import time
from openai import OpenAI

# enter the environment
'''

cd /home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo
module use /soft/modulefiles
module load conda
conda activate llava
module load cudatoolkit-standalone/11.8.0
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.8.0

'''

client = OpenAI(
    # api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    base_url = "https://api2.aigcbest.top/v1"
)

def generate_prompt(premise, hypothesis, label):
    return f"""
            You are tasked with rephrasing the given **premise** while preserving its original meaning and ensuring consistency with its associated **hypothesis**. Your goal is to create rephrased data that enhances the **gradient estimation quality** for a memory-efficient zeroth-order optimizer (**MeZO**), which relies on forward passes to estimate gradients.

            ### Key Requirements for Premise Rephrasing:
            1. **Task-Specific Context**:
              - The **RTE dataset** focuses on **textual entailment**, where the goal is to determine whether the **hypothesis** is **entailed or contradicted** with respect to the **premise**.
              - The rephrased **premise** must **not alter the logical relationship** between the **premise** and **hypothesis**.
              - The labels are defined as follows:
                - **0 (Entailment)**: The **premise** supports the **hypothesis**.
                - **1 (Contradiction)**: The **premise** contradicts the **hypothesis**.

            2. **Consistency with the Original Label**:
              - Ensure that the logical relationship between the **premise** and **hypothesis** remains unchanged.
              - If the label is **0 (Entailment)**, the rephrased **premise** must still entail the **hypothesis**.
              - If the label is **1 (Contradiction)**, the rephrased **premise** must still contradict the **hypothesis**.

            3. **Optimized for MeZO Training**:
              - **Enhance Gradient Sensitivity**: Enhance the model's **gradient sensitivity** to subtle variations in input by ensuring clear **semantic boundaries** and well-defined relationships in the rephrased data. Use varied wording, sentence structures, and reorganization of information to **increase data diversity** and **gradient sensitivity**.
              - **Focus on Memory Efficiency**: Reduce redundancy and keep the sentence concise to avoid unnecessary noise.
              - **Robustness to Data Sparsity**: Ensure that the scenarios are robust and provide essential information even with minimal context or details, simulating real-world situations where complete data may not always be available.
              - **Non-Differentiable Optimization Readiness**: Create clear impacts on performance metrics with distinct scenarios, ensuring these scenarios are measurable and distinctly influence model optimization targets.

            4. **Maintain Neutral Stance**:
              - Maintain the **neutral stance** of the original **premise**, carefully avoiding any explicit cues about whether the **hypothesis** is **entailed or contradicted**.
              - Ensure that the rephrased **premise** does not explicitly indicate its relationship to the **hypothesis**. The inference must still be a reasoning task based on the **hypothesis**.
              - The rephrased **premise** should only set the stage for inference without resolving or hinting at the correct relationship.
              - Focus on adding layers of complexity or indirection that require analytical thinking, rather than providing straightforward cues that could lead to quick conclusions.

            5. **High-Quality Data Generation**:
              - Produce a **natural, fluent, and coherent** rephrased **premise**.
              - Avoid directly mirroring the original structure; instead, introduce paraphrasing through synonyms, restructuring, or reordering.

            ### Your Task:
            Rephrase the following **premise** while ensuring that it remains logically consistent with the given **hypothesis** and its **label**. Ensure that the rephrased **premise** enhances **gradient estimation** for **MeZO** by being logically clear, semantically stable, and appropriately diverse.

            **Original premise**: "{premise}"  
            **Hypothesis**: "{hypothesis}"  
            **Label**: "{label}" (0 = **entailment**, 1 = **contradiction**)  

            **Directly output only one rephrased premise** without any other characters or explanatory statements like "The rephrased sentence is:":
            """


data = []
with open("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/RTE/rte_train.jsonl", 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/RTE/rte_train_top20.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    # if progress <= 815: # delete
    #     continue # delete
    if progress > 20: # delete
        break # delete

    prompt = generate_prompt(data[i]["premise"], data[i]["hypothesis"], data[i]["label"])
    response = client.chat.completions.create( # change
        model="claude-3-7-sonnet-latest",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # print(output)
    sentence = response.choices[0].message.content

    result = data[i]
    # result["sentence"] = temp
    result["premise"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
