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
        "id": "1",
        "original": "My body cast a shadow over the grass.", # modify
        "rephrased": "A shadow from my body was cast across the grass.", # modify 
        "eval": "same"
    }, 
    { 
        "id": "2",
        "original": "The woman tolerated her friend's difficult behavior.", # modify
        "rephrased": "The woman put up with her friend's challenging behavior.", # modify 
        "eval": "same"
    },
    { 
        "id": "3",
        "original": "The women met for coffee.", # modify
        "rephrased": "The women gathered for a coffee.", # modify 
        "eval": "not the same"
    },
    { 
        "id": "4",
        "original": "The runner wore shorts.", # modify
        "rephrased": "The athlete had on a pair of shorts.", # modify 
        "eval": "same"
    },
    { 
        "id": "5",
        "original": "The guests of the party hid behind the couch.", # modify
        "rephrased": "The party attendees concealed themselves behind the couch.", # modify 
        "eval": "same"
    },
    { 
        "id": "6",
        "original": "The politician lost the election.", # modify
        "rephrased": "The election did not turn out in the politician's favor.", # modify 
        "eval": "same"
    },
    { 
        "id": "7",
        "original": "The stain came out of the shirt.", # modify
        "rephrased": "The shirt had the stain successfully removed.", # modify 
        "eval": "same"
    },
    { 
        "id": "8",
        "original": "The man got a discount on his groceries.", # modify
        "rephrased": "A man received a price reduction on his grocery bill.", # modify 
        "eval": "same"
    },
    { 
        "id": "9",
        "original": "The physician misdiagnosed the patient.", # modify
        "rephrased": "The patient was given an incorrect diagnosis by the doctor.", # modify 
        "eval": "same"
    },
    { 
        "id": "10",
        "original": "The customer filed a complaint with the store manager.", # modify
        "rephrased": "The customer lodged a grievance with the store manager.", # modify 
        "eval": "same"
    },
    { 
        "id": "11",
        "original": "The woman repaired her faucet.", # modify
        "rephrased": "The woman fixed her tap.", # modify 
        "eval": "same"
    },
    { 
        "id": "12",
        "original": "The elderly woman suffered a stroke.", # modify
        "rephrased": "An aged woman experienced a stroke.", # modify 
        "eval": "same"
    },
    { 
        "id": "13",
        "original": "The pond froze over for the winter.", # modify
        "rephrased": "The pond became ice-covered during the winter.", # modify 
        "eval": "same"
    },
    { 
        "id": "14",
        "original": "The offender violated parole.", # modify
        "rephrased": "The parolee committed an infraction of their parole conditions.", # modify 
        "eval": "same"
    },
    { 
        "id": "15",
        "original": "I poured water on my sleeping friend.", # modify
        "rephrased": "I drenched my friend with water while they were asleep.", # modify 
        "eval": "not the same"
    },
    { 
        "id": "16",
        "original": "The girl gasped.", # modify
        "rephrased": "The girl let out a sharp intake of breath.", # modify 
        "eval": "same"
    },
    { 
        "id": "17",
        "original": "The shirt shrunk.", # modify
        "rephrased": "The shirt became smaller.", # modify 
        "eval": "not the same"
    },
    { 
        "id": "18",
        "original": "It got dark outside.", # modify
        "rephrased": "Night fell.", # modify 
        "eval": "same"
    },
    { 
        "id": "19",
        "original": "I hung up the phone.", # modify
        "rephrased": "I ended the phone call.", # modify 
        "eval": "same"
    },
    { 
        "id": "20",
        "original": "The woman's ring slipped off in the shower.", # modify
        "rephrased": "While taking a shower, the woman's ring fell off.", # modify 
        "eval": "same"
    }
]  

def few_shot(i, example):
    return example[i]

def prompt(sample_example):
    return f"Given the original sentence '{sample_example['original']}' and the rephrased sentence '{sample_example['rephrased']}', judge if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."

def messages(example):
    messages = [{"role": "system", "content": "You are a helpful judge."}]
    for i in range(len(example)):
        sample_example = few_shot(i, example)
        the_prompt = prompt(sample_example)
        messages.append({"role": "user", "content": the_prompt})
        messages.append({"role": "assistant", "content": sample_example["eval"]})
    return messages

messages = messages(example)

eval_list = []
count = 0
with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa/copa_train.jsonl', 'r', encoding="utf-8") as f1: # modify
    for line in f1:
        data = json.loads(line)
        temp = {}
        temp["original"] = data["premise"]
        temp["choice1"] = data["choice1"]
        temp["choice2"] = data["choice2"]
        temp["question"] = data["question"]
        temp["idx"] = data["idx"]
        temp["label"] = data["label"]
        eval_list.append(temp)


with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/Copa/copa_train.jsonl', 'r', encoding="utf-8") as f2: # modify
    for line in f2:
        data = json.loads(line)
        eval_list[count]["rephrased"] = data["premise"]
        count += 1

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/Copa/copa_train.jsonl") # output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")

correct_answer = 0
total_answer = 0

for i in tqdm(range(len(eval_list))):
    output_data = {}
    if i < 20:
        eval_list[i]['eval_result'] = example[i]["eval"]
        if example[i]["eval"] == "same":
            output_data["premise"] = eval_list[i]["rephrased"]
            output_data["choice1"] = eval_list[i]["choice1"]
            output_data["choice2"] = eval_list[i]["choice2"]
            output_data["question"] = eval_list[i]["question"]
            output_data["idx"] = eval_list[i]["idx"]
            output_data["label"] = eval_list[i]["label"]
            correct_answer += 1
            total_answer += 1
            out_file.write(json.dumps(output_data) + "\n")
            out_file.flush()
        else:
            print(example[i])
            output_data["premise"] = eval_list[i]["original"]
            output_data["choice1"] = eval_list[i]["choice1"]
            output_data["choice2"] = eval_list[i]["choice2"]
            output_data["question"] = eval_list[i]["question"]
            output_data["idx"] = eval_list[i]["idx"]
            output_data["label"] = eval_list[i]["label"]
            total_answer += 1
            out_file.write(json.dumps(output_data) + "\n")
            out_file.flush()
        continue
    
    prompt = f"Given the original sentence '{eval_list[i]['original']}' and the rephrased sentence '{eval_list[i]['rephrased']}', judge if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=messages,
        temperature=0.0 
    )
    messages.pop()
    if response.choices[0].message.content == 'not the same': # change
        print(eval_list[i]["rephrased"])
        eval_list[i]['eval_result'] = response.choices[0].message.content # change
        output_data["premise"] = eval_list[i]["original"]
        output_data["choice1"] = eval_list[i]["choice1"]
        output_data["choice2"] = eval_list[i]["choice2"]
        output_data["question"] = eval_list[i]["question"]
        output_data["idx"] = eval_list[i]["idx"]
        output_data["label"] = eval_list[i]["label"]
        total_answer += 1
        out_file.write(json.dumps(output_data) + "\n")
        out_file.flush()
        continue
    
    # print(response)
    eval_list[i]['eval_result'] = response.choices[0].message.content # change
    output_data["premise"] = eval_list[i]["rephrased"]
    output_data["choice1"] = eval_list[i]["choice1"]
    output_data["choice2"] = eval_list[i]["choice2"]
    output_data["question"] = eval_list[i]["question"]
    output_data["idx"] = eval_list[i]["idx"]
    output_data["label"] = eval_list[i]["label"]
    correct_answer += 1
    total_answer += 1
    out_file.write(json.dumps(output_data) + "\n")
    out_file.flush()

acc = correct_answer / total_answer
print("The prompt accuracy is ", acc)
