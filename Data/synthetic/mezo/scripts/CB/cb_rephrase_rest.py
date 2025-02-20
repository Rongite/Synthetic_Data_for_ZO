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
def generate_prompt(premise, hypothesis, label):
    return f"""
            You are tasked with rephrasing the given **premise** while preserving its original meaning and ensuring consistency with its associated **hypothesis**. Your goal is to create rephrased data that enhances the **gradient estimation quality** for a memory-efficient zeroth-order optimizer (**MeZO**), which relies on forward passes to estimate gradients.

            ### Key Requirements for Premise Rephrasing:
            1. **Task-Specific Context**:
              - The **CB dataset** focuses on **textual entailment**, where the goal is to determine whether the **hypothesis** is **entailed, contradicted, or neutral** with respect to the **premise**.
              - The rephrased **premise** must **not alter the logical relationship** between the **premise** and **hypothesis**.
              - The labels are defined as follows:
                - **0 (Entailment)**: The **premise** supports the **hypothesis**.
                - **1 (Contradiction)**: The **premise** contradicts the **hypothesis**.
                - **2 (Neutral)**: The **premise** neither supports nor contradicts the **hypothesis**.

            2. **Consistency with the Original Label**:
              - Ensure that the logical relationship between the **premise** and **hypothesis** remains unchanged.
              - If the label is **0 (Entailment)**, the rephrased **premise** must still entail the **hypothesis**.
              - If the label is **1 (Contradiction)**, the rephrased **premise** must still contradict the **hypothesis**.
              - If the label is **2 (Neutral)**, the rephrased **premise** must remain neutral.

            3. **Optimized for MeZO Training**:
              - **Enhance Gradient Sensitivity**: Enhance the model's **gradient sensitivity** to subtle variations in input by ensuring clear **semantic boundaries** and well-defined relationships in the rephrased data. Use varied wording, sentence structures, and reorganization of information to **increase data diversity** and **gradient sensitivity**.
              - **Focus on Memory Efficiency**: Reduce redundancy and keep the sentence concise to avoid unnecessary noise.
              - **Robustness to Data Sparsity**: Ensure that the scenarios are robust and provide essential information even with minimal context or details, simulating real-world situations where complete data may not always be available.
              - **Non-Differentiable Optimization Readiness**: Create clear impacts on performance metrics with distinct scenarios, ensuring these scenarios are measurable and distinctly influence model optimization targets.

            4. **Maintain Neutral Stance**:
              - Maintain the **neutral stance** of the original **premise**, carefully avoiding any explicit cues about whether the **hypothesis** is **entailed, contradicted, or neutral**.
              - Ensure that the rephrased **premise** does not explicitly indicate its relationship to the **hypothesis**. The inference must still be a reasoning task based on the **hypothesis**.
              - The rephrased **premise** should only set the stage for inference without resolving or hinting at the correct relationship.
              - Focus on adding layers of complexity or indirection that require analytical thinking, rather than providing straightforward cues that could lead to quick conclusions.

            5. **High-Quality Data Generation**:
              - Produce a **natural, fluent, and coherent** rephrased **premise**.
              - Avoid directly mirroring the original structure; instead, introduce paraphrasing through synonyms, restructuring, or reordering.

            ---

            ### Few-shot Examples:
            Original premise: "It was a complex language. Not written down but handed down. One might say it was peeled down."
            hypothesis: "the language was peeled down"
            label: "0"
            Rephrased premise: "The language was intricate, transmitted orally rather than documented, and could be described as being stripped down."

            Original premise: "It is part of their religion, a religion I do not scoff at as it holds many elements which match our own even though it lacks the truth of ours. At one of their great festivals they have the ritual of driving out the devils from their bodies. First the drummers come on - I may say that no women are allowed to take part in this ritual and the ladies here will perhaps agree with me that they are fortunate in that omission."
            hypothesis: "no women are allowed to take part in this ritual"
            label: "0"
            Rephrased premise: "Their religion, which I respect for its similarities to ours despite lacking our truth, includes a significant festival where they perform a ritual to expel devils from their bodies. Initially, the drummers appear, and it is noteworthy that women are not permitted to participate in this ritual, a fact that some ladies might consider a fortunate exclusion."

            Original premise: "The Paris to Rouen railway was being extended to Le Havre, and the line cut straight through Dr Flaubert's land. Part of it was to be compulsorily purchased. You could say that Gustave was shepherded into creative retreat at Croisset by epilepsy."
            hypothesis: "Gustave was shepherded into creative retreat at Croisset by epilepsy"
            label: "0"
            Rephrased premise: "The railway from Paris to Rouen was being extended to reach Le Havre, cutting directly through Dr. Flaubert's property, necessitating the compulsory purchase of part of it. This situation, combined with his epilepsy, led Gustave to seek a creative retreat in Croisset."

            Original premise: "Part of it was to be compulsorily purchased. You could say that Gustave was shepherded into creative retreat at Croisset by epilepsy. You could also say he was driven there by the railway."
            hypothesis: "Gustave was driven to creative retreat in Croisset by the railway"
            label: "0"
            Rephrased premise: "A section of it was to be acquired compulsorily. It could be said that Gustave found himself in a creative retreat at Croisset due to epilepsy, or alternatively, that the railway led him there."

            Original premise: "Some of them, like for instance the farm in Connecticut, are quite small. If I like a place I buy it. I guess you could say it's a hobby."
            hypothesis: "buying places is a hobby"
            label: "0"
            Rephrased premise: "Occasionally, I purchase properties, such as the small farm in Connecticut, simply because I enjoy them. You might consider this a hobby of mine."

            Original premise: "Look, my dear, I'm not in my dotage yet, and I know I'm a grumbler and a complainer. You could say the only form of comfort I 've got are my complaints."
            hypothesis: "the only form of comfort he has are his complaints"
            label: "0"
            Rephrased premise: "While I may be known for my grumbling and complaining, rest assured, I'm not senile yet, and it's fair to say that my complaints are my sole source of solace."

            Original premise: "Then the silence in the Zoo became complete. Woil stared around him and then suddenly with a push of his wings raised himself into the air, turned, and landed ten feet away on the back of a green bench. Creggan could see that he was afraid and that his fear was making him terribly uncertain."
            hypothesis: "Woil was afraid"
            label: "0"
            Rephrased premise: "In the Zoo, an all-encompassing silence took over. Woil glanced around nervously before suddenly flapping his wings to lift off, landing a short distance away on a green bench. It was evident to Creggan that Woil's fear was causing him significant hesitation."

            Original premise: "But, of course, that just wasn't possible. Her disappointment over Jonathan, which had driven her to France in the first place, had been relegated somewhere to the back of her mind. Now in retrospect she could see that marriage to him would have been a ghastly mistake and that her reaction to discovering that he had been seeing other women while he had been engaged to her had had far more to do with wounded pride than with a wounded heart."
            hypothesis: "marriage to Jonathan would have been a ghastly mistake"
            label: "0"
            Rephrased premise: "Certainly, here's the rephrased premise:\n\n\"Naturally, that was simply not feasible. Her initial distress over Jonathan, which had prompted her move to France, had been pushed to the recesses of her mind. Looking back, she realized that marrying him would have been a terrible error, and her response to finding out about his infidelity during their engagement was more about injured pride than a broken heart."

            Original premise: "Like now. The Community in Knockglen would defend Eve vociferously. Even some of the Sisters here in Dublin might see that the girl had a point."
            hypothesis: "the girl had a point"
            label: "0"
            Rephrased premise: "In the present moment, the people of Knockglen would passionately support Eve. Even a few of the Sisters in Dublin might acknowledge that the girl had a valid argument."

            Original premise: "They have to be crushed, Bobkins! So saying, she marched off down the gravel path, making the kind of crunching noise Robert had thought could only be produced by the BBC sound-effects department. As they rounded the edge of the building he could see that behind the house was a vast garden."
            hypothesis: "behind the house was a vast garden"
            label: "0"
            Rephrased premise: "''They must be destroyed, Bobkins!'' With that declaration, she strode down the gravel path, creating a crunching sound that Robert had believed only the BBC sound-effects team could replicate. As they turned the corner of the building, he noticed a sprawling garden behind the house."

            Original premise: "Do you mind if I use your phone?'' Ronni could see that Guido's brain was whirring."
            hypothesis: "Guido's brain was whirring"
            label: "0"
            Rephrased premise: "Ronni noticed that Guido was deep in thought when she asked to borrow his phone."

            Original premise: "Look, lady, give me a break. I just deliver the stuff, I don't interview it for the Sunday papers.'' He waved the paper at her and even at a distance she could see that it said very little."
            hypothesis: "the paper said very little"
            label: "0"
            Rephrased premise: "Listen, ma'am, cut me some slack. My job is just to deliver the items, not to write articles about them for the weekend edition. He gestured with the paper, and even from afar, she could discern that it contained minimal information."

            Original premise: "You 've heard something like that before. Haven't you, Jimmy?'' Jimmy had been shaken by those sounds more shaken than the others for good reason but Cardiff could see that he was unprepared to show it as he pushed himself away from the reception counter."
            hypothesis: "Jimmy was unprepared to show he had been shaken"
            label: "0"
            Rephrased premise: "''You've encountered something similar before, haven't you, Jimmy?'' While the others were also affected, Jimmy was particularly disturbed by those noises, yet Cardiff noticed he was not ready to reveal his unease as he distanced himself from the reception desk."

            Original premise: "It was Alan's idea. He made a sour kind of joke out of it, that they must wait until their wedding night. Carolyn agreed because she could see he meant it although she didn't understand why."
            hypothesis: "Alan meant it"
            label: "0"
            Rephrased premise: "Alan proposed the idea, making a somewhat bitter joke about waiting until their wedding night. Carolyn consented, recognizing his sincerity despite not grasping the reason."

            Original premise: "And why bother to write anyway? What was there to say? Mary had some vague idea that Adam's parents might suspect he was down here and come to see him."
            hypothesis: "Adam was down here"
            label: "0"
            Rephrased premise: "Why even write at all? Was there anything worth mentioning? Mary had a faint notion that Adam's parents might think he was present here and might visit him."

            Original premise: "And you're not having this dress,' Nora said, bending down to look at the price tag. It's two and a half guineas!'' she hissed at Louise who could tell that she was genuinely appalled."
            hypothesis: "Nora was genuinely appalled"
            label: "0"
            Rephrased premise: "Nora leaned over to check the price tag and exclaimed to Louise, \"You're definitely not getting this dress. It's two and a half guineas!\" Her tone made it clear that she was truly shocked."

            Original premise: "She said good morning to Alice and Alice said hallo. She was thin and rather tall with a very lined gentle face and hair that was white but which Alice could see had once been blonde. She could also have told this was Tina's mother before Mrs Darne went off down the passage that led to the Headmaster's Flat."
            hypothesis: "this was Tina's mother"
            label: "0"
            Rephrased premise: "Alice was greeted with a good morning, to which she responded with a hello. The woman was slender and quite tall, her face gentle and lined, with hair that was white but had clearly been blonde in the past. Even before Mrs. Darne walked down the corridor leading to the Headmaster's Flat, Alice could tell that this woman was Tina's mother."

            Original premise: "She might have sat all afternoon, nibbling and stuporous, exhausted but not sleepy. But the glazier finally came down from the upper floor, cheerfully announcing that all was now right and tight and he would be on his way. Maggie could tell that he would have liked to stop for a chat that he felt sorry for her left on her own but she lacked either her grandmother's grace or her mother's energy so she did not offer him tea."
            hypothesis: "the glazier would have liked to stop for a chat"
            label: "0"
            Rephrased premise: "Maggie might have spent the entire afternoon in a daze, nibbling and weary, yet not drowsy. Eventually, the glazier descended from the upper floor, happily declaring that everything was fixed and he would be leaving. Maggie sensed his desire to pause for a conversation, feeling sympathy for her solitude, but lacking her grandmother's charm or her mother's vigor, she refrained from inviting him for tea."

            Original premise: "Jim waited a long time, and when the young doctor finally came out, it was almost dark. Jim could nonetheless tell by his anxious face that something was wrong."
            hypothesis: "something was wrong"
            label: "0"
            Rephrased premise: "Jim stood by, enduring a lengthy wait until the young doctor emerged, just as dusk was settling in. Despite the dim light, Jim could discern from the doctor's worried expression that there was an issue."

            Original premise: "Yes?'' Nathan?'' He could tell it was long-distance the line was so gravelly and hollow but he didn't recognise the voice."
            hypothesis: "the call was long-distance"
            label: "0"
            Rephrased premise: "Hello? Nathan? The voice on the other end was unfamiliar, and the connection was so rough and echoing that it was clear the call was from far away."

            ---

            ### Your Task:
            Rephrase the following **premise** while ensuring that it remains logically consistent with the given **hypothesis** and its **label**. Ensure that the rephrased **premise** enhances **gradient estimation** for **MeZO** by being logically clear, semantically stable, and appropriately diverse.

            **Original premise**: "{premise}"  
            **Hypothesis**: "{hypothesis}"  
            **Label**: "{label}" (0 = **entailment**, 1 = **contradiction**, 2 = **neutral**)  

            **Directly output only one rephrased premise** without any other characters or explanatory statements like "The rephrased sentence is:":
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
with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/CB/cb_train.jsonl', 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/CB/cb_train_rest.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    if progress <= 20: # delete
        continue # delete
    # if progress > 20: # delete
    #     break # delete

    prompt = generate_prompt(data[i]["premise"], data[i]["hypothesis"], data[i]["label"])
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # print(output)
    sentence = response.choices[0].message.content

    result = data[i]
    # result["sentence"] = temp
    result["premise"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
