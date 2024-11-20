from tqdm import tqdm
import os
import json
from vllm import LLM, SamplingParams


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_NET"] = "Socket"

# Initialize the vLLM model
model_id = "meta-llama/Llama-3.2-3B"
llm = LLM(model=model_id, tensor_parallel_size=4, device="cuda", dtype="bfloat16", max_model_len=2048)

# Load input data
data = []
input_file = '/grand/sbi-fair/jikaiLoong/Python_code/vllm/data/original/sst2_train.jsonl'  # Input file path
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Output file paths
output_file = os.path.expanduser("/grand/sbi-fair/jikaiLoong/Python_code/vllm/data/synthetic/sst2_train.jsonl")

# Ensure output directories exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Open files for writing
out_file = open(output_file, "w")

# Process each sentence
for i in tqdm(range(len(data))):
    sentence = data[i]["sentence"]

    # Prompt for vLLM
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

    # Define sampling parameters
    sampling_params = SamplingParams(max_tokens=512, temperature=0.1, top_p=0.5)

    # Generate output using vLLM
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    
    # Parse and update the result
    result = data[i]
    result["sentence"] = generated_text

    # Write the result to the output file
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()

# Close files
out_file.close()