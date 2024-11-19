### evaluate using open ai api
import json
from openai import OpenAI
from tqdm import tqdm
import os

client = OpenAI(
    api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    # api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    base_url = "https://api2.aigcbest.top/v1"
)

eval_list = []

# Example hypothesis and reference
example = [ 
    { 
        "original": "hide new secretions from the parental units ", # modify
        "rephrased": "Conceal sensitive information from parental figures." # modify 
    }, 
    { 
        "original": "contains no wit , only labored gags ", # modify 
        "rephrased": "Lacks cleverness, relying instead on forced humor." # modify 
    }, 
] 

# Customized few-shot examples
example_prompt_0 = f"Given the original sentence '{example[0]['original']}' and the rephrased sentence '{example[0]['rephrased']}', jude if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."
example_prompt_1 = f"Given the original sentence '{example[1]['original']}' and the rephrased sentence '{example[1]['rephrased']}', jude if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."

eval_list = []
count = 0
with open('/Users/jikailong/Desktop/2_virtual_server/synthetic_data/zo/original/sst2/sst2_train.jsonl', 'r', encoding="utf-8") as f1: # modify
    for line in f1:
        data = json.loads(line)
        temp = {}
        temp["original"] = data["sentence"]
        temp["label"] = data["label"]
        temp["idx"] = data["idx"]
        eval_list.append(temp)


with open('/Users/jikailong/Desktop/2_virtual_server/synthetic_data/zo/synthetic/sst2/sst2_train.jsonl', 'r', encoding="utf-8") as f2: # modify
    for line in f2:
        data = json.loads(line)
        eval_list[count]["rephrased"] = data["sentence"]
        count += 1

output_file = os.path.expanduser("/Users/jikailong/Desktop/2_virtual_server/synthetic_data/zo/rejection_sampling/sst2/sst2_train.jsonl") # output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")

for i in tqdm(range(len(eval_list))):
    output_data = {}
    if i < 2:
        if i== 0:
            eval_list[i]['eval_result'] = "same" # modify
            output_data["sentence"] = eval_list[i]["rephrased"]
            output_data["label"] = eval_list[i]["label"]
            output_data["idx"] = eval_list[i]["idx"]
            out_file.write(json.dumps(output_data) + "\n")
            out_file.flush()
        else:
            eval_list[i]['eval_result'] = "same" # modify
            output_data["sentence"] = eval_list[i]["rephrased"]
            output_data["label"] = eval_list[i]["label"]
            output_data["idx"] = eval_list[i]["idx"]
            out_file.write(json.dumps(output_data) + "\n")
            out_file.flush()
        i += 1
        continue
    prompt = f"Given the original sentence '{eval_list[i]['original']}' and the rephrased sentence '{eval_list[i]['rephrased']}', jude if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful judge."},
            {"role": "user", "content": example_prompt_0},
            {"role": "assistant", "content": "same"}, # modify
            {"role": "user", "content": example_prompt_1},
            {"role": "assistant", "content": "same"}, # modify
            {"role": "user", "content": prompt}
        ],
        temperature=0.0 
    )
    if response.choices[0].message.content == 'not the same': # change
        print(eval_list[i]["rephrased"])
        eval_list[i]['eval_result'] = response.choices[0].message.content # change
        output_data["sentence"] = eval_list[i]["original"]
        output_data["label"] = eval_list[i]["label"]
        output_data["idx"] = eval_list[i]["idx"]
        out_file.write(json.dumps(output_data) + "\n")
        out_file.flush()
        continue
    
    # print(response)
    eval_list[i]['eval_result'] = response.choices[0].message.content # change
    output_data["sentence"] = eval_list[i]["rephrased"]
    output_data["label"] = eval_list[i]["label"]
    output_data["idx"] = eval_list[i]["idx"]
    out_file.write(json.dumps(output_data) + "\n")
    out_file.flush()
