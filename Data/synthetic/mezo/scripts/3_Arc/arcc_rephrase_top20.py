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
            You are tasked with rephrasing the given science question from the ARC-Challenge dataset into a semantically equivalent form, while preserving the \
              original meaning, answer choices, and the correct answer. Your goal is to produce rephrased data optimized specifically for enhancing gradient \
                estimation in training with a memory-efficient zeroth-order optimizer (MeZO).

            ### Key Requirements for Question Rephrasing:
            1. **Semantic Stability**:
              - Preserve the exact meaning and scientific accuracy of the original question.
              - Maintain strict logical and semantic consistency between the rephrased question and the provided answer choices.

            2. **Gradient Sensitivity**:
              - Phrase the question clearly to emphasize subtle semantic differences between each answer choice.
              - Adjust sentence structure or wording to sharpen the distinctions between answer choices, making the gradients 
                estimated by MeZO more sensitive and clearly differentiable.

            3. **Difficulty Level & Precision**:
              - Maintain the original reasoning complexity and difficulty level; avoid simplifications or excessive elaboration 
                that would alter complexity.
              - Clearly distinguish correct from incorrect choices through nuanced phrasing without explicitly indicating the correct answer.

            4. **Answer Choice Consistency**:
              - Do not alter the original answer choices.
              - The correct answer must remain identical.

            5. **High Data Quality**:
              - Ensure fluency, clarity, and natural scientific language.
              - Utilize paraphrasing, synonyms, varied sentence structures, or subtle reordering to diversify data representation 
                and enhance model sensitivity to small textual differences.

            ### Your Task:
            Given the original ARC-Challenge question below, produce a single rephrased question strictly adhering to the above requirements. 
            Ensure logical clarity and semantic stability, optimized specifically to aid gradient estimation for MeZO optimization.

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
    result["question"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
