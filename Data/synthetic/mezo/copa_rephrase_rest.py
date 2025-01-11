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

# The 6th shot is wrong
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

            ---

            ### Few-shot Examples:
            Original premise: "My body cast a shadow over the grass."
            Choice 1: "The sun was rising."
            Choice 2: "The grass was cut."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "A shadow appeared on the grass beside me."

            Original premise: "The woman tolerated her friend's difficult behavior."
            Choice 1: "The woman knew her friend was going through a hard time."
            Choice 2: "The woman felt that her friend took advantage of her kindness."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The woman was patient with her friend's challenging attitude."

            Original premise: "The women met for coffee."
            Choice 1: "The cafe reopened in a new location."
            Choice 2: "They wanted to catch up with each other."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The women gathered at a cafe."

            Original premise: "The runner wore shorts."
            Choice 1: "The forecast predicted high temperatures."
            Choice 2: "She planned to run along the beach."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "Shorts were the runner's attire."

            Original premise: "The guests of the party hid behind the couch."
            Choice 1: "It was a surprise party."
            Choice 2: "It was a birthday party."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "During the gathering, the attendees positioned themselves out of sight behind the sofa."

            Original premise: "The politician lost the election."
            Choice 1: "He ran negative campaign ads."
            Choice 2: "No one voted for him."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The election was unsuccessful for the politician."

            Original premise: "The stain came out of the shirt."
            Choice 1: "I patched the shirt."
            Choice 2: "I bleached the shirt."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The shirt was no longer marked by the stain."

            Original premise: "The man got a discount on his groceries."
            Choice 1: "He greeted the cashier."
            Choice 2: "He used a coupon."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The man managed to pay less for his grocery bill."

            Original premise: "The physician misdiagnosed the patient."
            Choice 1: "The patient filed a malpractice lawsuit against the physician."
            Choice 2: "The patient disclosed confidential information to the physician."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The doctor made an error in the patient's diagnosis."

            Original premise: "The customer filed a complaint with the store manager."
            Choice 1: "The sales associate undercharged the customer."
            Choice 2: "The sales associate acted rude to the customer."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The customer approached the store manager with a grievance."

            Original premise: "The woman repaired her faucet."
            Choice 1: "The faucet was leaky."
            Choice 2: "The faucet was turned off."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "A woman attended to fixing her faucet."

            Original premise: "The elderly woman suffered a stroke."
            Choice 1: "The woman's daughter came over to clean her house."
            Choice 2: "The woman's daughter moved in to take care of her."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "An elderly woman experienced a medical event affecting her brain."

            Original premise: "The pond froze over for the winter."
            Choice 1: "People skated on the pond."
            Choice 2: "People brought boats to the pond."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "Throughout the winter, the surface of the pond turned solid with ice."

            Original premise: "The offender violated parole."
            Choice 1: "She was sent back to jail."
            Choice 2: "She stole money from a church."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "Breaking the conditions set by parole occurred."

            Original premise: "I poured water on my sleeping friend."
            Choice 1: "My friend awoke."
            Choice 2: "My friend snored."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "A stream of water landed on my slumbering friend."

            Original premise: "The girl gasped."
            Choice 1: "Her friend stuck an ice cube down her back."
            Choice 2: "Her friend gave her a pat on the back."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "A sudden gasp escaped from the girl."

            Original premise: "The shirt shrunk."
            Choice 1: "I poured bleach on it."
            Choice 2: "I put it in the dryer."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The shirt became smaller."

            Original premise: "It got dark outside."
            Choice 1: "Snowflakes began to fall from the sky."
            Choice 2: "The moon became visible in the sky."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "As night approached, the sky lost its light."

            Original premise: "I hung up the phone."
            Choice 1: "The caller said goodbye to me."
            Choice 2: "The caller identified himself to me."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "I ended the call."

            Original premise: "The woman's ring slipped off in the shower."
            Choice 1: "The woman polished the ring."
            Choice 2: "The ring went down the drain."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "During her shower, the woman's ring unexpectedly came loose."

            ---

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

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/Copa/copa_train_rest.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    if progress <= 20: # delete
        continue # delete
    # if progress > 20: # delete
    #     break # delete

    prompt = generate_prompt(data[i]["premise"], data[i]["choice1"], data[i]["choice2"], data[i]["question"], data[i]["label"])
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    
    # print(output)
    sentence = response.choices[0].message.content

    result = data[i]
    # result["sentence"] = temp
    result["premise"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
