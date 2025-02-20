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

cd /grand/sbi-fair/jikaiLoong/Python_code/llava/synthetic_data/lora
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

'''
original_data = {"premise":"The friends' debate dragged on interminably.","choice1":"The friends saw eye to eye.","choice2":"The friends were splitting hairs.","question":"cause","idx":29,"label":1}
sentence = original_data["premise"]

prompt = f"""
You are tasked with rephrasing the following premise while preserving its original meaning and sentiment. The rephrased data will be used to fine-tune a model using a \
  memory-efficient zeroth-order optimizer called MeZO. This optimizer significantly reduces memory usage compared to traditional backpropagation methods. Below is the \
    abstract from the MeZO research paper for context:

"Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of \
  memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be catastrophically slow for optimizing \
    large models. In this work, we propose a memory-efficient zeroth-order optimizer (MeZO), adapting the classical ZO-SGD method to operate in-place, thereby \
      fine-tuning LMs with the same memory footprint as inference. For example, with a single A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas \
        fine-tuning with backpropagation can train only a 2.7B LM with the same budget. Our results demonstrate that MeZO achieves comparable performance to \
          fine-tuning with backpropagation across multiple tasks while drastically reducing memory requirements."

Please rephrase the following premise while ensuring its meaning and sentiment remain unchanged. Aim for diverse language usage to improve data quality for \
  fine-tuning:

Original: "{sentence}"
Output only the best rephrased sentence without any other characters:
"""

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

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/Copa/copa_train.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
# progress = 0 # delete
for i in tqdm(range(len(data))):
    # progress += 1 # delete
    # if progress <= 815: # delete
    #     continue # delete
    sentence = data[i]["premise"]
    # temp = []

    prompt = f"""
    You are tasked with rephrasing the following premise while preserving its original meaning and sentiment. The rephrased data will be used to fine-tune a model using a \
      memory-efficient zeroth-order optimizer called MeZO. This optimizer significantly reduces memory usage compared to traditional backpropagation methods. Below is the \
        abstract from the MeZO research paper for context:

    "Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of \
      memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be catastrophically slow for optimizing \
        large models. In this work, we propose a memory-efficient zeroth-order optimizer (MeZO), adapting the classical ZO-SGD method to operate in-place, thereby \
          fine-tuning LMs with the same memory footprint as inference. For example, with a single A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas \
            fine-tuning with backpropagation can train only a 2.7B LM with the same budget. Our results demonstrate that MeZO achieves comparable performance to \
              fine-tuning with backpropagation across multiple tasks while drastically reducing memory requirements."

    Please rephrase the following premise while ensuring its meaning and sentiment remain unchanged. Aim for diverse language usage to improve data quality for \
      fine-tuning:

    Original: "{sentence}"
    Directly output only one rephrased sentence without any other characters and other explanatory statements like "The rephrased sentence is:":
    """
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0 
    )
    
    # print(output)
    sentence = response.choices[0].message.content

    result = data[i]
    # result["sentence"] = temp
    result["premise"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
