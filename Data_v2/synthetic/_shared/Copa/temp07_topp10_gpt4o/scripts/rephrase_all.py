#!/usr/bin/env python3
"""
Auto-generated synthetic data generation script

Task: Copa
Training Method: mezo
Generation Model: gpt-4o
Strategy: all
Generation Time: 2025-12-30 19:06:38
"""

from tqdm import tqdm
import os
import json
from openai import OpenAI

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api2.aigcbest.top/v1")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    timeout=120
)

# ⭐ Few-shot examples placeholder (will be injected by review_top20.py)
# FEWSHOT_EXAMPLES = [...]

def generate_prompt(premise, choice1, choice2, question, label):
    """Generate rephrasing prompt"""

    # ⭐ Build few-shot text（If existsFEWSHOT_EXAMPLES）
    fewshot_text = ""
    if 'FEWSHOT_EXAMPLES' in globals() and len(FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {i}:\n"
            fewshot_text += f"Original premise: {ex['original']}\n"
            fewshot_text += f"Rephrased premise: {ex['rephrased']}\n"
            # add other fieldsas as  up  down text
            for key in ex:
                if key not in ['original', 'rephrased']:
                    fewshot_text += f"{key}: {ex[key]}\n"
            fewshot_text += "\n"

    # ⭐ Original prompt template
    prompt_template = """\
You are tasked with rephrasing the given premise while preserving its original meaning. Your goal is to create rephrased data optimized for enhancing gradient estimation in training with a memory-efficient zeroth-order optimizer (MeZO).

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

### Key Requirements for Premise Rephrasing:
1. **Task-Specific Context**:
   - The **Copa dataset** focuses on **causal reasoning**, where the goal is to determine the **cause or effect** of a given premise.
   - The rephrased **premise** must **not alter the logical relationship** with the correct choice.

2. **Consistency with Correct Answer**:
   - Ensure that the rephrased premise maintains the same correct answer as the original.
   - The correct answer is: Choice {label + 1}: "{choice1 if label == 0 else choice2}"

3. **Optimized for MeZO Training**:
   - **Enhance Gradient Sensitivity**: Create clear semantic boundaries to increase gradient sensitivity.
   - **Focus on Memory Efficiency**: Reduce redundancy, keep sentences concise.
   - **Robustness to Data Sparsity**: Ensure essential information is preserved.
   - **Non-Differentiable Optimization Readiness**: Create clear impacts on performance metrics.

4. **Maintain Neutral Stance**:
   - Do not explicitly indicate which choice is correct.
   - The rephrased premise should require reasoning to determine the answer.

5. **High-Quality Data Generation**:
   - Produce natural, fluent, and coherent text.
   - Use paraphrasing, synonyms, or restructuring to diversify data.

### Your Task:
Rephrase the following premise while ensuring it remains consistent with the correct answer.

**Original premise**: "{premise}"
**Choice 1**: "{choice1}"
**Choice 2**: "{choice2}"
**Question**: "{question}"
**Correct answer**: "{choice1 if label == 0 else choice2}"

**Directly output only one rephrased premise** without any other characters or explanatory statements:

"""

    # ⭐ Replace {{REPHRASE_FEWSHOT}} placeholder
    prompt = prompt_template.replace("{{REPHRASE_FEWSHOT}}", fewshot_text)

    # ⭐ Replace field values（use.format()）
    return prompt.format(premise=premise, choice1=choice1, choice2=choice2, question=question, label=label)

# Loaded original data
data = []
input_file = "/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/Copa/copa_train.jsonl"
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(f"Loaded {len(data)}  samplesoriginalData")

# Prepare output
# 🆕 CreateDatasetSubDirectory
dataset_dir = os.path.join("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o", "Copa")
os.makedirs(dataset_dir, exist_ok=True)

output_file = os.path.join(dataset_dir, "copa_train.jsonl")
out_file = open(output_file, "w", encoding='utf-8')

print(f"outputFile: {output_file}")

# Process data
progress = 0
for i in tqdm(range(len(data))):
    progress += 1

    # ConstructPrompt
    prompt_args = {"premise": data[i]["premise"], "choice1": data[i]["choice1"], "choice2": data[i]["choice2"], "question": data[i]["question"], "label": data[i]["label"]}
    prompt = generate_prompt(**prompt_args)

    # Call API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # Extract result
        rephrased_text = response.choices[0].message.content.strip()

        # Constructoutput
        result = data[i].copy()
        result["premise"] = rephrased_text

        # writeFile
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_file.flush()

    except Exception as e:
        print(f"\nProcessline {i}  samplesDataError when: {e}")
        # Use original data on error
        out_file.write(json.dumps(data[i], ensure_ascii=False) + "\n")
        out_file.flush()

out_file.close()
print(f"\nComplete! output: {output_file}")
