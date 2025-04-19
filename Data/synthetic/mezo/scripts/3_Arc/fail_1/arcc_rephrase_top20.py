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

def generate_prompt(question, choices, answerKey):
    return f"""
            ARC-Challenge Rephrasing Prompt for MeZO Training

            You are tasked with rephrasing the given science question while preserving its original meaning and ensuring consistency with its answer choices. Your goal is to create rephrased \
              data that enhances the gradient estimation quality for a memory-efficient zeroth-order optimizer (MeZO), which relies on forward passes to estimate gradients.

            ### Key Requirements for Question Rephrasing:
            1. **Task-Specific Context**:
              - The **ARC-Challenge dataset** consists of multiple-choice science questions that require reasoning and factual knowledge to answer correctly.
              - The rephrased **question** must **not alter the meaning** or the **correct answer**.
              - The structure of the multiple-choice format must be preserved, ensuring that the **rephrased question** still aligns with the **same answer choices**.

            2. **Consistency with the Correct Answer**:
              - Ensure that the **correct answer** remains the **same** after rephrasing..
              - The **other incorrect choices** must still be plausible but incorrect, maintaining the challenge of the original question.
              - Avoid introducing hints or changes that make the question **easier or harder** than the original.

            3. **Optimized for MeZO Training**:
              - **Enhance Gradient Sensitivity**: Enhance the model's **gradient sensitivity** to subtle variations in input by ensuring clear **semantic boundaries** and well-defined \
                relationships in the rephrased data. Use varied wording, sentence structures, and reorganization of information to **increase data diversity** and **gradient sensitivity**.
              - **Focus on Memory Efficiency**: Reduce redundancy and keep the sentence concise to avoid unnecessary noise.
              - **Robustness to Data Sparsity**: Ensure that the scenarios are robust and provide essential information even with minimal context or details, simulating real-world situations \
                where complete data may not always be available.
              - **Non-Differentiable Optimization Readiness**: Create clear impacts on performance metrics with distinct scenarios, ensuring these scenarios are measurable and distinctly \
                influence model optimization targets.

            4. **Maintain Question Format & Logical Integrity**:
              - Ensure that the **question format** remains a **multiple-choice science question**.
              - Do not modify the number of **choices** or change their **labels (A, B, C, D)**.
              - Do not introduce explicit hints or clues that make the answer more obvious or obscure.
              - The **difficulty level should remain unchanged**—preserve the reasoning complexity required to determine the correct answer.

            5. **High-Quality Data Generation**:
              - The **rephrased question** should be **fluent, natural, and scientifically accurate**.
              - Avoid **word-for-word mirroring**; instead, introduce **synonyms, restructuring, and varied phrasing**.
              - Ensure that the **scientific reasoning required to solve the question remains unchanged**.

            ### Your Task:
            Rephrase the following **science question** while ensuring that it remains logically consistent with the given **answer choices** and **correct answer**. Ensure that the rephrased \
              question enhances **gradient estimation** for **MeZO** by being logically clear, semantically stable, and appropriately diverse.

            **Original question**: 
            "{question}"  

            **Answer choices**:  
            - **A:** {choices["text"][0]}  
            - **B:** {choices["text"][1]}  
            - **C:** {choices["text"][2]}  
            - **D:** {choices["text"][3]}  
            
            **Correct answer**: "{answerKey}"  

            **Directly output only one rephrased question** without any other characters or explanatory statements like “The rephrased question is:":
            """


data = []
with open("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze/ARC-Challenge_train.jsonl", 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/ArcC/ARC-Challenge_train_top20.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True) # no need to create the output path mannually
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    # if progress <= 815: # delete
    #     continue # delete
    if progress > 20: # delete
        break # delete

    prompt = generate_prompt(data[i]["question"], data[i]["choices"], data[i]["answerKey"])
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
    result["question"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
