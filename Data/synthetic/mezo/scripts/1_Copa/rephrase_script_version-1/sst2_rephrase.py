from tqdm import tqdm
import os
import ast
import json
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import time

# enter the environment
'''

cd /grand/sbi-fair/jikaiLoong/Python_code/llava/synthetic_data/lora
module use /soft/modulefiles
module load conda
conda activate llava
module load cudatoolkit-standalone/11.8.0
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.8.0

'''

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# original_data = {"sentence":"hide new secretions from the parental units ","label":0,"idx":0}
# sentence = original_data["sentence"]

# prompt = f"""
# You are tasked with rephrasing the provided sentence without changing its original sentiment and meaning. The rephrased data will be used to fine-tune a \
#   model using a memory-efficient zeroth-order optimizer called MeZO, which reduces memory usage significantly compared to traditional backpropagation methods. \
#     The goal is to generate data that helps the model achieve similar or better accuracy when fine-tuned with limited memory, compared to using the original \
#       dataset.

# Below is the abstract from the MeZO research paper for context:

# "Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large \
#   amount of memory. Zeroth-order (ZO) methods can estimate gradients using only two forward passes but are traditionally slow for large models. In this \
#     work, we propose a memory-efficient zeroth-order optimizer (MeZO), adapting the ZO-SGD method to operate in-place with the same memory footprint as \
#       inference. Our results show MeZO achieves comparable performance to backpropagation-based fine-tuning with up to 12× memory reduction. This optimizer \
#         allows training a 30B model on a single A100 GPU while backpropagation can only handle a 2.7B model."

# Please rephrase the sentence below while keeping its sentiment and meaning intact. Aim for diverse language usage to enhance the data quality for \
#   fine-tuning with MeZO:

# Original: "{sentence}"
# Output only the best rephrased sentence without any other characters:
# """

# start_time = time.time()
# messages = [
#     {"role": "user", "content": prompt}
# ]
 
# input_text = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
# )
# inputs = processor(text=input_text, images=None, return_tensors="pt").to(model.device)
# output = model.generate(**inputs, max_new_tokens=128000, temperature=0.1)
# end_time = time.time()
# print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))
# print(end_time - start_time, 's')

data = []
with open('/home/jlong1/Downloads/Data/zo/0_original_data/SST2/sst2_train.jsonl', 'r', encoding='utf-8') as file: # input file
    for line in file:
            # 解析每一行的 JSON 数据，并将其附加到 data 列表中
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Data/zo/1_original_zo/synthetic/synthetic.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")

error_file = os.path.expanduser("/home/jlong1/Downloads/Data/zo/1_original_zo/synthetic/err_data/err_data.jsonl") # error file
os.makedirs(os.path.dirname(error_file), exist_ok=True)
err_file = open(error_file, "w")

# progress = 0 # delete
for i in tqdm(range(len(data))):
    # progress += 1 # delete
    # if progress <= 815: # delete
    #     continue # delete
    sentence = data[i]["sentence"]
    # temp = []

    prompt = f"""
    You are tasked with rephrasing the provided sentence without changing its original sentiment and meaning. The rephrased data will be used to fine-tune a \
      model using a memory-efficient zeroth-order optimizer called MeZO, which reduces memory usage significantly compared to traditional backpropagation methods. \
        The goal is to generate data that helps the model achieve similar or better accuracy when fine-tuned with limited memory, compared to using the original \
          dataset.

    Below is the abstract from the MeZO research paper for context:

    "Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large \
      amount of memory. Zeroth-order (ZO) methods can estimate gradients using only two forward passes but are traditionally slow for large models. In this \
        work, we propose a memory-efficient zeroth-order optimizer (MeZO), adapting the ZO-SGD method to operate in-place with the same memory footprint as \
          inference. Our results show MeZO achieves comparable performance to backpropagation-based fine-tuning with up to 12× memory reduction. This optimizer \
            allows training a 30B model on a single A100 GPU while backpropagation can only handle a 2.7B model."

    Please rephrase the sentence below while keeping its sentiment and meaning intact. Aim for diverse language usage to enhance the data quality for \
      fine-tuning with MeZO:

    Original: "{sentence}"
    Directly output a rephrased sentence without any other characters and other explanatory statements like "The rephrased sentence is:":
    """

    messages = [
        {"role": "user", "content": prompt}
    ]
    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    inputs = processor(text=input_text, images=None, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=4096)
    output = processor.decode(output[0][inputs["input_ids"].shape[-1]:])
    # print(output)
    sentence = output.rsplit("<|eot_id|>", 1)[0]
    # try:
    #     sentence = json.loads(output.rsplit("<|eot_id|>", 1)[0])
    # except:
    #     err_file.write(json.dumps(data[i]) + "\n")
    #     err_file.flush()
    #     continue

    result = data[i]
    # result["sentence"] = temp
    result["sentence"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    err_file.flush()
    i += 1
    