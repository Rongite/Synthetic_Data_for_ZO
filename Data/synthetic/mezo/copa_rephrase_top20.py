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
    api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    # api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    base_url = "https://api2.aigcbest.top/v1"
)

def generate_prompt(premise, choice1, choice2, question, label):
    # Dynamically insert the correct answer based on the label
    correct_choice = choice1 if label == 0 else choice2
    return f"""
            You are tasked with rephrasing the given premise while preserving its original meaning and ensuring consistency with its associated question and \
              choices. Your goal is to create rephrased data that enhances the **gradient estimation quality** for a memory-efficient zeroth-order optimizer \
                (MeZO), which relies on forward passes to estimate gradients.

            ### Key Requirements for Premise Rephrasing:
            1. **Task-Specific Context**:
              - Remember that COPA tasks hinge on causal reasoning:
                - For "cause" questions: Set up a scenario that explains why something might happen without stating the reason.
                - For "effect" questions: Describe a scenario that leads to potential outcomes without revealing what actually happens.

            2. **Consistency with Correct Answer**:
              - Ensure the rephrased premise is compatible with both choices, without leaning towards the correct answer. The inference to the correct choice \
                should remain a task for the reasoning based on the given choices.
              - Ensure that the correct answer remains unchanged. If the label is "0," Choice 1 must still be correct. If the label is "1," Choice 2 must still be \
                correct.
              - Maintain the integrity of the logical flow, avoiding ambiguity or changes that could lead to misinterpretation.
              
            3. **Optimized for MeZO Training**:
              - Enhance Gradient Sensitivity: Enhance the model's **gradient sensitivity** to subtle variations in input by ensuring clear **semantic \
                  boundaries** and well-defined causal relationships in the rephrased data. Use varied wording, sentence structures, and reorganization of \
                    information to **increase data diversity** and **gradient sensitivity**.
              - Focus on Memory Efficiency: Reduce redundancy and keep the sentence concise to avoid unnecessary noise.
              - Robustness to Data Sparsity: Ensure that the scenarios are robust and provide essential information even with minimal context or details, \
                simulating real-world situations where complete data may not always be available.
              - Non-Differentiable Optimization Readiness: Create clear impacts on performance metrics with distinct scenarios, ensuring these scenarios are \
                measurable and distinctly influence model optimization targets.
                
            4. **Maintain Neutral Stance**:
              - Maintain the **neutral stance** of the original premise, carefully avoiding any explicit mention of causes or effects that are to be deduced from \
                the given choices.
              - Ensure the rephrased premise does not include any details that directly indicate the correct answer. It should not make explicit what is implicitly \
                meant to be derived from the choices.
              - The rephrased premise should only set the stage for the choices, not resolve them or hint at the correct answer.
              - Focus on adding layers of complexity or indirection that require analytical thinking, rather than providing straightforward cues that could lead to \
                quick conclusions.

            5. **High-Quality Data Generation**:
              - Produce a **natural, fluent, and coherent** rephrased premise.
              - Avoid directly mirroring the original structure; instead, introduce paraphrasing through synonyms, restructuring, or reordering.

            ### Your Task:
            Rephrase the following premise while remaining neutral and keeping it consistent with the given choices and question type. Ensure the rephrased premise \
              enhances gradient estimation for MeZO by being logically clear, semantically stable, and appropriately diverse.

            Original premise: "{premise}"
            Choice 1: "{choice1}"
            Choice 2: "{choice2}"
            Question type: "{question}" (cause or effect)
            Correct answer: "{correct_choice}"

            Directly output only one rephrased premise without any other characters and other explanatory statements like "The rephrased sentence is:":
            """

'''
original_data = {"premise":"The friends' debate dragged on interminably.","choice1":"The friends saw eye to eye.","choice2":"The friends were splitting hairs.","question":"cause","idx":29,"label":1}
sentence = original_data["premise"]

prompt = generate_prompt(original_data["premise"], original_data["choice1"], original_data["choice2"], original_data["question"], original_data["label"])

start_time = time.time()
response = client.chat.completions.create( # change
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.0 
)
end_time = time.time()
print(response.choices[0].message.content)
print(end_time - start_time, 's')

'''

data = []
with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa/copa_train.jsonl', 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/Copa/copa_train_top20.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    # if progress <= 815: # delete
    #     continue # delete
    if progress > 20: # delete
        break # delete

    prompt = generate_prompt(data[i]["premise"], data[i]["choice1"], data[i]["choice2"], data[i]["question"], data[i]["label"])
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1.0 
    )
    
    # print(output)
    sentence = response.choices[0].message.content

    result = data[i]
    # result["sentence"] = temp
    result["premise"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
