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

def generate_prompt(original_premise, choice1, choice2, question, correct_choice, rephrased_premise):
    return f"""
            Task Description:

            You are tasked with verifying whether a rephrased premise maintains consistency with its associated question and answer choices. Specifically, \
                determine if the correct answer inferred from the rephrased premise remains the same as that inferred from the original premise. Below are \
                    correctly rephrased examples to guide you:

            ### Few-shot Examples:
            
            Original premise: "The girl received a trophy."
            Choice 1: "She won a spelling bee."
            Choice 2: "She made a new friend."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "A trophy was awarded to the girl."

            Original premise: "The woman's date wanted to look like a gentleman."
            Choice 1: "He opened the door for her."
            Choice 2: "He asked her if she liked sushi."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The woman's companion aimed to present himself as courteous."

            Original premise: "The farmland needed irrigation."
            Choice 1: "A canal was constructed."
            Choice 2: "A flood occurred."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The agricultural land required additional water sources."

            Original premise: "The host cancelled the party."
            Choice 1: "She was certain she had the flu."
            Choice 2: "She worried she would catch the flu."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The host called off the gathering."

            Original premise: "The woman gave the man her phone number."
            Choice 1: "She was attracted to him."
            Choice 2: "She was repulsed by him."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The woman shared her phone number with the man."

            Original premise: "The skydiver glided safely to the ground."
            Choice 1: "She opened her parachute."
            Choice 2: "She jumped out of the plane."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The skydiver landed smoothly on the ground."

            Original premise: "The toddler became cranky."
            Choice 1: "Her mother put her down for a nap."
            Choice 2: "Her mother fixed her hair into pigtails."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The toddler grew irritable."

            Original premise: "The child became immune to the disease."
            Choice 1: "He avoided exposure to the disease."
            Choice 2: "He received the vaccine for the disease."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The child developed immunity to the disease."

            Original premise: "The grape juice fermented."
            Choice 1: "The juice turned to wine."
            Choice 2: "The juice evaporated."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The grape juice underwent a fermentation process."

            Original premise: "The friends' debate dragged on interminably."
            Choice 1: "The friends saw eye to eye."
            Choice 2: "The friends were splitting hairs."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The debate among the friends extended without end."

            Original premise: "The woman hummed to herself."
            Choice 1: "She was nervous."
            Choice 2: "She was in a good mood."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "A melody quietly emerged from the woman's lips."

            Original premise: "The man hated his new haircut."
            Choice 1: "He wore a hat."
            Choice 2: "He grew a beard."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "His dissatisfaction with the haircut was apparent."

            Original premise: "The police aimed their weapons at the fugitive."
            Choice 1: "The fugitive fell to the ground."
            Choice 2: "The fugitive dropped his gun."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The officers pointed their guns at the fleeing suspect."

            Original premise: "The patient was dehydrated."
            Choice 1: "The nurse tested his reflexes."
            Choice 2: "The nurse gave him an IV."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The patient experienced a lack of hydration."

            Original premise: "The girl found the missing puzzle piece."
            Choice 1: "She completed the puzzle."
            Choice 2: "She took apart the puzzle."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The girl located the lost puzzle piece."

            Original premise: "The man urgently leaped out of bed."
            Choice 1: "He wanted to shut off the alarm clock."
            Choice 2: "He wanted to iron his pants before work."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The man swiftly got out of bed."

            Original premise: "The papers were disorganized."
            Choice 1: "I made photocopies of them."
            Choice 2: "I put them into alphabetical order."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The papers lacked organization."

            Original premise: "The woman won the lottery."
            Choice 1: "She bought a yacht."
            Choice 2: "She joined a church."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The woman became a lottery winner."

            Original premise: "The seamstress pushed the threaded needle into the fabric."
            Choice 1: "The thread wrapped around the needle."
            Choice 2: "The thread went through the fabric."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The seamstress inserted the needle with thread into the fabric."

            Original premise: "The woman hired a lawyer."
            Choice 1: "She decided to sue her employer."
            Choice 2: "She decided to run for office."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "A lawyer was employed by the woman."

            ---

            Your Task:

            Evaluate the following data and determine whether the rephrased premise maintains consistency with the original premise regarding the correct answer \
                choice.

            Original premise: "{original_premise}"
            Choice 1: "{choice1}"
            Choice 2: "{choice2}"
            Question type: "{question}" (cause or effect)
            Correct answer: "{correct_choice}"
            Rephrased premise: "{rephrased_premise}"

            Example Input:
            Original premise: "The girl received a trophy."
            Choice 1: "She won a spelling bee."
            Choice 2: "She made a new friend."
            Question type: "cause"
            Correct answer: "Choice 1"
            Rephrased premise: "A trophy was awarded to the girl."

            Expected Output: same

            Directly output [same/not the same] without any explanation:

            """

def correct_choice(choice1, choice2, label):
    correct_choice = choice1 if label == 0 else choice2
    return correct_choice

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
    if 20 <= i < 40:
        eval_list[i]['eval_result'] = "same"
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
        continue

    the_correct_choice = correct_choice(eval_list[i]["choice1"], eval_list[i]["choice2"], eval_list[i]["label"])
    prompt = generate_prompt(eval_list[i]["original"], eval_list[i]["choice1"], eval_list[i]["choice2"], eval_list[i]["question"], \
                    the_correct_choice, eval_list[i]["rephrased"])
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0 
    )
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
