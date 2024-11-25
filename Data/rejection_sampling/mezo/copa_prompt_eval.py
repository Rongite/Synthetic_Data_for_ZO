import json
from openai import OpenAI
from tqdm import tqdm
import os

client = OpenAI(
    api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    # api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    base_url = "https://api2.aigcbest.top/v1"
)

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

test_set = [ 
    { 
        "id": "21",
        "original": "The girl received a trophy.", # modify
        "rephrased": "A trophy was awarded to the girl.", # modify 
        "eval": "same"
    }, 
    { 
        "id": "22",
        "original": "The woman's date wanted to look like a gentleman.", # modify
        "rephrased": "The woman's companion aimed to present himself as a gentleman.", # modify 
        "eval": "same"
    },
    { 
        "id": "23",
        "original": "The farmland needed irrigation.", # modify
        "rephrased": "The farmland required irrigation.", # modify 
        "eval": "same"
    },
    { 
        "id": "24",
        "original": "The host cancelled the party.", # modify
        "rephrased": "The party was called off by the host.", # modify 
        "eval": "same"
    },
    { 
        "id": "25",
        "original": "The woman gave the man her phone number.", # modify
        "rephrased": "The woman handed her phone number to the man.", # modify 
        "eval": "same"
    },
    { 
        "id": "26",
        "original": "The skydiver glided safely to the ground.", # modify
        "rephrased": "The skydiver descended to the ground safely.", # modify 
        "eval": "same"
    },
    { 
        "id": "27",
        "original": "The toddler became cranky.", # modify
        "rephrased": "The toddler turned irritable.", # modify 
        "eval": "same"
    },
    { 
        "id": "28",
        "original": "The child became immune to the disease.", # modify
        "rephrased": "The child developed immunity to the illness.", # modify 
        "eval": "same"
    },
    { 
        "id": "29",
        "original": "The grape juice fermented.", # modify
        "rephrased": "The grape juice underwent fermentation.", # modify 
        "eval": "same"
    },
    { 
        "id": "30",
        "original": "The friends' debate dragged on interminably.", # modify
        "rephrased": "The friends' discussion seemed to go on endlessly.", # modify 
        "eval": "not the same"
    },
    { 
        "id": "31",
        "original": "The woman hummed to herself.", # modify
        "rephrased": "She hummed quietly to herself.", # modify 
        "eval": "not the same"
    },
    { 
        "id": "32",
        "original": "The man hated his new haircut.", # modify
        "rephrased": "He despised his recent haircut.", # modify 
        "eval": "same"
    },
    { 
        "id": "33",
        "original": "The police aimed their weapons at the fugitive.", # modify
        "rephrased": "Law enforcement pointed their firearms at the escapee.", # modify 
        "eval": "same"
    },
    { 
        "id": "34",
        "original": "The patient was dehydrated.", # modify
        "rephrased": "The patient experienced dehydration.", # modify 
        "eval": "not the same"
    },
    { 
        "id": "35",
        "original": "The girl found the missing puzzle piece.", # modify
        "rephrased": "The girl discovered the lost puzzle piece.", # modify 
        "eval": "same"
    },
    { 
        "id": "36",
        "original": "The man urgently leaped out of bed.", # modify
        "rephrased": "The man sprang from the bed with urgency.", # modify 
        "eval": "same"
    },
    { 
        "id": "37",
        "original": "The papers were disorganized.", # modify
        "rephrased": "The papers were in a state of disorder.", # modify 
        "eval": "same"
    },
    { 
        "id": "38",
        "original": "The woman won the lottery.", # modify
        "rephrased": "A woman struck it rich in the lottery.", # modify 
        "eval": "same"
    },
    { 
        "id": "39",
        "original": "The seamstress pushed the threaded needle into the fabric.", # modify
        "rephrased": "The seamstress inserted the needle with thread into the cloth.", # modify 
        "eval": "same"
    },
    { 
        "id": "40",
        "original": "The woman hired a lawyer.", # modify
        "rephrased": "A lawyer was employed by the woman.", # modify 
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
correct_answer = 0
total_answer = 0

for i in tqdm(range(len(test_set))):
    output_data = {}
    prompt = f"Given the original sentence '{test_set[i]['original']}' and the rephrased sentence '{test_set[i]['rephrased']}', judge if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=messages,
        temperature=0.0 
    )
    messages.pop()
    if response.choices[0].message.content == test_set[i]["eval"]: # change
        correct_answer += 1
        total_answer += 1
        continue
    
    # print(response)
    print(test_set[i])
    print(f"gpt eval result: '{response.choices[0].message.content}'")
    total_answer += 1

acc = correct_answer / total_answer
print("The prompt accuracy is ", acc)
