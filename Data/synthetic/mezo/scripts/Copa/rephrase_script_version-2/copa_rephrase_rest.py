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

# The 6th shot is wrong
def generate_prompt(premise, choice1, choice2, question, label):
    # Dynamically insert the correct answer based on the label
    correct_choice = choice1 if label == 0 else choice2
    return f"""
            You are tasked with rephrasing the following premise while preserving its original meaning and ensuring that it remains consistent with its associated \
              question and choices. The rephrased premise must maintain the logic of the original data so that the correct answer remains unchanged. The rephrased \
                data will be used to fine-tune a memory-efficient zeroth-order optimizer (MeZO) model. Below are the ABSTRACT and INTRODUCTION sections from the \
                  MeZO research paper to provide additional context:

            ABSTRACT:
            "Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively \
              large amount of memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be \
                catastrophically slow for optimizing large models. In this work, we propose a memory-efficient zeroth-order optimizer (MeZO), adapting the \
                  classical ZO-SGD method to operate in-place, thereby fine-tuning LMs with the same memory footprint as inference. For example, with a single \
                    A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas fine-tuning with backpropagation can train only a 2.7B LM with the same \
                      budget. Our results demonstrate that (1) MeZO significantly outperforms in-context learning and linear probing; (2) MeZO achieves comparable \
                        performance to fine-tuning with backpropagation across multiple tasks, with up to 12× memory reduction and up to 2× GPU-hour reduction in \
                          our implementation; (3) MeZO is compatible with both full-parameter and parameter-efficient tuning techniques such as LoRA and prefix \
                            tuning; (4) MeZO can effectively optimize non-differentiable objectives (e.g., maximizing accuracy or F1). We support our empirical \
                              findings with theoretical insights, highlighting how adequate pre-training and task prompts enable MeZO to fine-tune huge models, \
                                despite classical ZO analyses suggesting otherwise."

            INTRODUCTION:
            "Fine-tuning pre-trained language models (LMs) has been the dominant methodology for solving many language tasks, adapting to specialized domains ,\
                or incorporating human instructions and preferences. However, as LMs are scaled up, computing gradients for backpropagation requires \
                  a prohibitive amount of memory – in our test, up to 12× the memory required for inference – because it needs to cache activations during the \
                    forward pass, gradients during the backward pass, and, in the case of Adam, also store gradient history (see Section 3.4 for a \
                      detailed analysis).

            As a result, while it is possible to run inference with a 30-billion (30B) parameter LM on a single Nvidia A100 GPU (with 80GB memory), backpropagation \
              with Adam is feasible only for a 2.7B LM. Parameter-efficient fine-tuning methods (PEFT) update just a fraction of the network parameters \
                but still need to cache many activations, because the tuned parameters are scattered throughout the model. In our tests, fine-tuning an OPT-13B \
                  model with full parameter tuning or PEFT requires 12× and 6× more memory than inference respectively."

            FIGURE 1:
            This figure presents the performance of the OPT-13B model across multiple tasks (SST-2, RTE, CB, BoolQ, WSC, WiC, MultiRC, Copa, ReCoRD, SQuAD, and \
              DROP) \
              under four configurations:
            1. Zero-shot: Where no task-specific training is performed, and the model is tested directly.
            2. In-context learning (ICL): Where labeled examples are provided in the input prompt.
            3. MeZO fine-tuning: Using the MeZO optimizer with LoRA or prefix-tuning.
            4. Full fine-tuning (FT): Using Adam optimizer with full parameter tuning, which consumes 12× memory.

            Key observations from Figure 1:
            - MeZO achieves superior results over zero-shot and in-context learning.
            - MeZO’s performance is comparable to full fine-tuning, with less than 1% accuracy or F1 difference on 7 out of 11 tasks.
            - MeZO requires only 1/12th the memory compared to full fine-tuning.

            "Backpropagation also cannot optimize non-differentiable criteria, which have gained popularity in fine-tuning LMs according to discrete metrics or set \
              safety standards. Typically, these adaptations involve expensive reinforcement learning from human feedback (RLHF).

            A classical zeroth-order optimization method, ZO-SGD, uses only differences of loss values to estimate the gradients. Thus, in principle, the method \
              can update neural networks with just forward passes, though naive implementation still doubles the memory overhead and scales poorly with model size. \
                Existing ZO methods have been applied in deep learning settings to find adversarial examples or tune input embeddings [91, 90] but not to directly \
                  optimize large-scale models (see Liu et al. for a survey)."

            MeZO’s Contributions:
            1. In MeZO, we adapt the ZO-SGD algorithm and a number of variants to operate in-place on arbitrarily large models with almost no memory overhead (see \
              Algorithm 1 and Section 2).
            2. We conduct comprehensive experiments across model types (masked LM and autoregressive LM), model scales (from 350M to 66B), and downstream tasks \
              (classification, multiple-choice, and generation). MeZO consistently outperforms zero-shot, ICL, and linear probing. With RoBERTa-large, MeZO achieves \
                performance close to standard fine-tuning within 5%. With OPT-13B, MeZO outperforms or performs comparably to fine-tuning on 7 out of 11 tasks, \
                  despite requiring roughly 12× less memory (Figure 1 and Section 3). In our implementation, MeZO requires only half as many GPU-hours as Adam \
                    fine-tuning for a 30B model (see Appendix F.6).
            3. We demonstrate MeZO’s compatibility with full-parameter tuning and PEFT (e.g., LoRA and prefix-tuning) in Section 3.
            4. Further exploration showcases that MeZO can optimize non-differentiable objectives such as accuracy or F1 score while still requiring only the same \
              memory as inference (Section 3.3).
            5. Our theory suggests that adequate pre-training ensures a better optimization rate (Theorem 1) and a stable convergence rate (Lemma 3.4) of MeZO under \
              proper settings of hyperparameters. This matches practical experience, with Assumption 3.1 specifying sparsity of parameters."

            ALGORITHM 1:
            The MeZO algorithm uses perturbation-based gradient estimation:
            1. Parameters are perturbed to estimate gradients using forward passes only.
            2. Parameters are updated in place, avoiding memory overhead from gradient storage.
            3. Memory efficiency is achieved, requiring only inference-level resources.

            ---

            ### Few-shot Examples:
            Original premise: "My body cast a shadow over the grass."
            Choice 1: "The sun was rising."
            Choice 2: "The grass was cut."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The shadow of my body fell upon the grass."

            Original premise: "The woman tolerated her friend's difficult behavior."
            Choice 1: "The woman knew her friend was going through a hard time."
            Choice 2: "The woman felt that her friend took advantage of her kindness."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The woman endured her friend's challenging behavior."

            Original premise: "The women met for coffee."
            Choice 1: "The cafe reopened in a new location."
            Choice 2: "They wanted to catch up with each other."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The women gathered to have coffee together."

            Original premise: "The runner wore shorts."
            Choice 1: "The forecast predicted high temperatures."
            Choice 2: "She planned to run along the beach."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The runner decided to wear shorts."

            Original premise: "The guests of the party hid behind the couch."
            Choice 1: "It was a surprise party."
            Choice 2: "It was a birthday party."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The party attendees concealed themselves behind the couch."

            Original premise: "The stain came out of the shirt."
            Choice 1: "I patched the shirt."
            Choice 2: "I bleached the shirt."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The shirt became clean after the stain was removed."

            Original premise: "The man got a discount on his groceries."
            Choice 1: "He greeted the cashier."
            Choice 2: "He used a coupon."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The man received a reduction in the price of his groceries."

            Original premise: "The physician misdiagnosed the patient."
            Choice 1: "The patient filed a malpractice lawsuit against the physician."
            Choice 2: "The patient disclosed confidential information to the physician."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The doctor incorrectly diagnosed the patient."

            Original premise: "The customer filed a complaint with the store manager."
            Choice 1: "The sales associate undercharged the customer."
            Choice 2: "The sales associate acted rude to the customer."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The customer lodged a complaint with the store manager."

            Original premise: "The woman repaired her faucet."
            Choice 1: "The faucet was leaky."
            Choice 2: "The faucet was turned off."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The woman fixed her faucet."

            Original premise: "The elderly woman suffered a stroke."
            Choice 1: "The woman's daughter came over to clean her house."
            Choice 2: "The woman's daughter moved in to take care of her."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The elderly woman experienced a stroke."

            Original premise: "The pond froze over for the winter."
            Choice 1: "People skated on the pond."
            Choice 2: "People brought boats to the pond."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "During the winter, the pond became frozen."

            Original premise: "The offender violated parole."
            Choice 1: "She was sent back to jail."
            Choice 2: "She stole money from a church."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The offender breached parole conditions."

            Original premise: "I poured water on my sleeping friend."
            Choice 1: "My friend awoke."
            Choice 2: "My friend snored."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "Water was poured on my friend as they slept."

            Original premise: "The girl gasped."
            Choice 1: "Her friend stuck an ice cube down her back."
            Choice 2: "Her friend gave her a pat on the back."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The girl let out a gasp."

            Original premise: "The shirt shrunk."
            Choice 1: "I poured bleach on it."
            Choice 2: "I put it in the dryer."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The shirt reduced in size."

            Original premise: "It got dark outside."
            Choice 1: "Snowflakes began to fall from the sky."
            Choice 2: "The moon became visible in the sky."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "Nighttime set in."

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
            Rephrased premise: "The woman's ring fell off in the shower."

            ---

            Your task:
            Rephrase the given premise without altering its meaning. The rephrased premise must remain consistent with the provided question and choices, such that \
              the correct answer remains unchanged.

            Original premise: "{premise}"
            Choice 1: "{choice1}"
            Choice 2: "{choice2}"
            Question type: "{question}" (cause or effect)
            Correct answer: "{correct_choice}"

            Directly output only one rephrased sentence without any other characters and other explanatory statements like "The rephrased sentence is:":
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
