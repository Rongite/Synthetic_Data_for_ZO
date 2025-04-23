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

def generate_prompt(question, passage, answer):
    return f"""
            You are tasked with rephrasing the given **passage** while preserving its original meaning and ensuring consistency with its associated **question** and **boolean answer**. Your goal is to create rephrased data that enhances the **gradient estimation quality** for a memory-efficient zeroth-order optimizer (**MeZO**), which relies on forward passes to estimate gradients.

            ### Key Requirements for Passage Rephrasing:

            1. **Task-Specific Context (BoolQ Dataset)**:
              - The **BoolQ dataset** consists of yes/no questions and short passages. The task is to determine whether the **passage** supports a **"true"** or **"false"** answer to the **question**.
              - The rephrased **passage** must **not change the answerability or truth value** of the original passage with respect to the question.
              - If the original **answer** is **true**, the rephrased passage must continue to support a "yes" response.
              - If the original **answer** is **false**, the rephrased passage must not provide justification to answer "yes".

            2. **Consistency with the Original Answer**:
              - Ensure that the rephrased **passage** leads to the **same boolean answer** as the original.
              - Do not insert or omit information that would alter the logical basis of the answer.

            3. **Optimized for MeZO Training**:
              - **Enhance Gradient Sensitivity**: Design the rephrased passage to respond distinctly to minor variations in input phrasing or token order. Use varied vocabulary and structure to strengthen gradient signal.
              - **Focus on Memory Efficiency**: Avoid verbose phrasing or redundant elaboration. Keep the passage informative yet concise to reduce noise.
              - **Robustness to Data Sparsity**: Ensure the passage remains informative even if truncated or partially occluded, simulating real-world input loss.
              - **Non-Differentiable Optimization Readiness**: Use language that leads to clear decision boundaries (yes/no), enabling optimization of discrete metrics such as accuracy.

            4. **Maintain Informational Neutrality**:
              - Do not insert direct yes/no claims. Let the **question**-to-**answer** reasoning depend entirely on the passage content.
              - Avoid rhetorical or leading language that may bias the answer beyond what the original passage supports.

            5. **High-Quality Data Generation**:
              - Produce a **natural, fluent, and coherent** rephrased passage.
              - Avoid direct copying of the original structure; instead, use paraphrasing, clause reordering, or syntax transformation.

            ### Your Task:
            Rephrase the following **passage** while ensuring that it leads to the same answer for the given **question**. The rephrased passage should maintain logical consistency and enhance MeZO’s gradient estimation effectiveness.

            **question**: "{question}"  
            **answer**: "{'true' if answer else 'false'}"  
            **Original passage**: "{passage}"  

            **Directly output only one rephrased passage** without any other characters or explanatory statements like "The rephrased sentence is:":
            """


data = []
with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ/boolq_train.jsonl', 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/BOOLQ/boolq_train_top20.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    # if progress <= 815: # delete
    #     continue # delete
    if progress > 20: # delete
        break # delete

    # def generate_prompt(question, passage, answer):
    prompt = generate_prompt(data[i]["question"], data[i]["passage"], data[i]["answer"])
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # print(output)
    sentence = response.choices[0].message.content

    result = data[i]
    # result["sentence"] = temp
    result["passage"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
