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

def generate_prompt(question, choices, answerKey):
    return f"""
            ARC-Challenge Rephrasing Prompt for MeZO Training

            You are tasked with rephrasing the given science question while preserving its original meaning and ensuring consistency with its answer choices. \
              Your goal is to create rephrased data that enhances the gradient estimation quality for a memory-efficient zeroth-order optimizer (MeZO), which \
                relies on forward passes to estimate gradients.

            ### Key Requirements for Question Rephrasing:
            1. **Task-Specific Context**:
              - The **ARC-Challenge dataset** consists of multiple-choice science questions that require reasoning and factual knowledge to answer correctly.
              - The rephrased **question** must **not alter the meaning** or the **correct answer**.
              - The structure of the multiple-choice format must be preserved, ensuring that the **rephrased question** still aligns with the **same answer \
                choices**.

            2. **Consistency with the Correct Answer**:
              - Ensure that the **correct answer** remains the **same** after rephrasing..
              - The **other incorrect choices** must still be plausible but incorrect, maintaining the challenge of the original question.
              - Avoid introducing hints or changes that make the question **easier or harder** than the original.

            3. **Optimized for MeZO Training**:
              - **Enhance Gradient Sensitivity**: Enhance the model's **gradient sensitivity** to subtle variations in input by ensuring clear **semantic \
                boundaries** and well-defined relationships in the rephrased data. Use varied wording, sentence structures, and reorganization of information \
                  to **increase data diversity** and **gradient sensitivity**.
              - **Focus on Memory Efficiency**: Reduce redundancy and keep the sentence concise to avoid unnecessary noise.
              - **Robustness to Data Sparsity**: Ensure that the scenarios are robust and provide essential information even with minimal context or details, \
                simulating real-world situations where complete data may not always be available.
              - **Non-Differentiable Optimization Readiness**: Create clear impacts on performance metrics with distinct scenarios, ensuring these scenarios are \
                measurable and distinctly influence model optimization targets.

            4. **Maintain Question Format & Logical Integrity**:
              - Ensure that the **question format** remains a **multiple-choice science question**.
              - Do not modify the number of **choices** or change their **labels (A, B, C, D)**.
              - Do not introduce explicit hints or clues that make the answer more obvious or obscure.
              - The **difficulty level should remain unchanged**—preserve the reasoning complexity required to determine the correct answer.

            5. **High-Quality Data Generation**:
              - The **rephrased question** should be **fluent, natural, and scientifically accurate**.
              - Avoid **word-for-word mirroring**; instead, introduce **synonyms, restructuring, and varied phrasing**.
              - Ensure that the **scientific reasoning required to solve the question remains unchanged**.

            ---

            ### Few-shot Examples:
            Original question: "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?"
            Answer choices:  
            - **A:** dry palms  
            - **B:** wet palms  
            - **C:** palms covered with oil  
            - **D:** palms covered with lotion  
            Correct answer: "A"
            Rephrased question: "Which type of skin surface will generate the most heat when George rubs his hands together to warm them quickly?"

            Original question: "Which of the following statements best explains why magnets usually stick to a refrigerator door?"
            Answer choices:  
            - **A:** The refrigerator door is smooth.  
            - **B:** The refrigerator door contains iron.  
            - **C:** The refrigerator door is a good conductor.  
            - **D:** The refrigerator door has electric wires in it.  
            Correct answer: "B"
            Rephrased question: "Why do magnets typically adhere to a refrigerator door?"

            Original question: "A fold observed in layers of sedimentary rock most likely resulted from the"
            Answer choices:  
            - **A:** cooling of flowing magma.  
            - **B:** converging of crustal plates.  
            - **C:** deposition of river sediments.  
            - **D:** solution of carbonate minerals.  
            Correct answer: "B"
            Rephrased question: "What is the most probable cause of a fold found in sedimentary rock layers?"

            Original question: "Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the \
              Mesozoic era?"
            Answer choices:  
            - **A:** worldwide disease  
            - **B:** global mountain building  
            - **C:** rise of mammals that preyed upon plants and animals  
            - **D:** impact of an asteroid created dust that blocked the sunlight  
            Correct answer: "D"
            Rephrased question: "What is the latest scientific explanation for the mass extinction of numerous plant and animal species at the conclusion of the \
              Mesozoic era?"

            Original question: "A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels northeast. In which \
              direction is the wind most likely applying force to the sails of the boat?"
            Answer choices:  
            - **A:** west  
            - **B:** east  
            - **C:** north  
            - **D:** south  
            Correct answer: "B"
            Rephrased question: "A boat is influenced by a river current moving northward and a wind impacting its sails. The boat's path is toward the \
              northeast. Which direction is the wind most likely exerting force on the boat's sails?"

            Original question: "Which land form is the result of the constructive force of a glacier?"
            Answer choices:  
            - **A:** valleys carved by a moving glacier  
            - **B:** piles of rocks deposited by a melting glacier  
            - **C:** grooves created in a granite surface by a glacier  
            - **D:** bedrock hills roughened by the passing of a glacier  
            Correct answer: "B"
            Rephrased question: "What type of landform is produced by the constructive action of a glacier?"

            Original question: "Which statement best compares single-celled and multi-celled organisms?"
            Answer choices:  
            - **A:** Tissues in a single-celled organism are like the cells in a multi-celled organism.  
            - **B:** The nucleus in a single-celled organism is like the skin of a multi-celled organism.  
            - **C:** Organelles in a single-celled organism are like the organs in a multi-celled organism.  
            - **D:** The cytoplasm in a single-celled organism is like the nervous system in a multi-celled organism.  
            Correct answer: "C"
            Rephrased question: "How can single-celled organisms be most accurately compared to multi-celled organisms?"

            Original question: "As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut \
              has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?"
            Answer choices:  
            - **A:** 31 pounds and 14 kilograms  
            - **B:** 31 pounds and 84 kilograms  
            - **C:** 186 pounds and 14 kilograms  
            - **D:** 186 pounds and 84 kilograms  
            Correct answer: "D"
            Rephrased question: "An astronaut conducts an experiment by using a scale on the Moon, which shows his weight as 31 pounds. Given that his mass is \
              approximately 84 kilograms, what would be his estimated weight and mass when he is on Earth?"

            Original question: "Which of the following is a trait that a dog does NOT inherit from its parents?"
            Answer choices:  
            - **A:** the length of its fur  
            - **B:** the shape of its nose  
            - **C:** the size of its appetite  
            - **D:** the color of its fur  
            Correct answer: "C"
            Rephrased question: "Which characteristic listed below is not genetically passed down to a dog from its parents?"

            Original question: "Hatchling sea turtles are typically dark in color. Occasionally, a sea turtle hatches that is almost white in color. When \
              crawling from the nest on the beach to the ocean, this light-colored sea turtle could be at risk for sunburn. The light color of the turtles would \
                most likely"
            Answer choices:  
            - **A:** help the turtles have better chances at reproducing.  
            - **B:** cause the shell of the sea turtles to become stronger.  
            - **C:** reduce the chances of turtles surviving to reproduce.  
            - **D:** help in the development of a new species of sea turtles.  
            Correct answer: "C"
            Rephrased question: "What is the most likely impact of a nearly white sea turtle hatchling's coloration on its survival when moving from its nest on \
              the beach to the ocean, considering the risk of sunburn?"
            
            Original question: "Photosynthesis is a process involving carbon dioxide, water, glucose, oxygen, and sunlight. Which is the correct chemical \
              equation for photosynthesis?"
            Answer choices:  
            - **A:** O_{2} + H_{2}O + energy -> C_{6}H_{12}O_{6} + CO_{2}  
            - **B:** CO_{2} + H_{2}O -> C_{6}H_{6}O_{3} + O_{2} + energy  
            - **C:** 6O_{2} + 6H_{2}O -> C_{6}H_{12}O_{6} + 6CO_{2} + energy  
            - **D:** 6CO_{2} + 6H_{2}O + energy -> C_{6}H_{12}O_{6} + 6O_{2}  
            Correct answer: "D"
            Rephrased question: "Which chemical equation accurately represents the process of photosynthesis, involving carbon dioxide, water, glucose, oxygen, \
              and sunlight?"

            Original question: "Which of the following is an example of a physical change?"
            Answer choices:  
            - **A:** lighting a match  
            - **B:** breaking a glass  
            - **C:** burning of gasoline  
            - **D:** rusting of iron  
            Correct answer: "B"
            Rephrased question: "Which option below illustrates a physical change?"

            Original question: "Which of the following areas is most likely to form metamorphic rocks such as gneiss and schist?"
            Answer choices:  
            - **A:** a sea floor  
            - **B:** a windblown desert  
            - **C:** a site deep underground  
            - **D:** a site covered by a glacier  
            Correct answer: "C"
            Rephrased question: "In which of the following locations are metamorphic rocks like gneiss and schist most likely to develop?"

            Original question: "The male insects in a population are treated to prevent sperm production. Would this reduce this insect population?"
            Answer choices:  
            - **A:** No, because the insects would still mate.  
            - **B:** No, because it would not change the offspring mutation rate.  
            - **C:** Yes, because it would sharply decrease the reproduction rate.  
            - **D:** Yes, because the males would die.  
            Correct answer: "C"
            Rephrased question: "If male insects in a group are treated to inhibit sperm production, would this action lead to a decline in their population?"

            Original question: "On Earth, water can be a solid, a liquid, or a gas. Which energy source has the greatest influence on the state of matter of water?"
            Answer choices:  
            - **A:** the sun  
            - **B:** the wind  
            - **C:** ocean currents  
            - **D:** the metal core  
            Correct answer: "A"
            Rephrased question: "What energy source most significantly affects the physical state of water on Earth, whether it is solid, liquid, or gas?"

            Original question: "A ship leaks a large amount of oil near a coastal area. Which statement describes how the oil most likely will affect the coastal habitat?"
            Answer choices:  
            - **A:** Fish reproduction rates will increase.  
            - **B:** Water birds will be unable to use their wings.  
            - **C:** Water plants will be exposed to more sunlight.  
            - **D:** Coastal plants will have access to more nutrients.  
            Correct answer: "B"
            Rephrased question: "How is the coastal environment most likely impacted when a ship spills a significant amount of oil nearby?"

            Original question: "A chemical property of a mineral is evident if the mineral"
            Answer choices:  
            - **A:** breaks easily when struck with a hammer  
            - **B:** bubbles when acid is placed on it  
            - **C:** is easily scratched by a fingernail  
            - **D:** reflects light from its surface  
            Correct answer: "2"
            Rephrased question: "What indicates a mineral's chemical property?"

            Original question: "One evening as it is getting dark, Alex sits on the front porch and watches the sun slowly disappear behind the neighbor's house across the street. Which explains this observation?"
            Answer choices:  
            - **A:** The sun's light is reflected by the clouds.  
            - **B:** The sun's light is refracted by the atmosphere.  
            - **C:** The sun moves from west to east each day.  
            - **D:** The sun appears to move due to Earth's rotation.  
            Correct answer: "D"
            Rephrased question: "As evening falls and darkness approaches, Alex observes from the front porch as the sun gradually vanishes behind the neighbor's house across the street. What accounts for this phenomenon?"

            Original question: "A toothpaste commercial states that a brand of toothpaste has a higher concentration of fluoride than any other toothpaste available. The commercial is most likely inferring that the advertised toothpaste"
            Answer choices:  
            - **A:** has a pleasant flavor.  
            - **B:** is recommended by dentists.  
            - **C:** promotes good dental hygiene.  
            - **D:** is the most expensive brand sold.  
            Correct answer: "C"
            Rephrased question: "A toothpaste advertisement claims that their brand contains more fluoride than any other toothpaste on the market. What is the advertisement most likely suggesting about this toothpaste?"

            Original question: "What do cells break down to produce energy?"
            Answer choices:  
            - **A:** food  
            - **B:** water  
            - **C:** chlorophyll  
            - **D:** carbon dioxide  
            Correct answer: "A"
            Rephrased question: "Which substance do cells decompose to generate energy?"

            ---

            ### Your Task:
            Rephrase the following **science question** while ensuring that it remains logically consistent with the given **answer choices** and **correct answer**. Ensure that the rephrased question enhances **gradient estimation** for **MeZO** by being logically clear, semantically stable, and appropriately diverse.

            **Original question**: 
            "{question}"  

            **Answer choices**:  
            - **A:** {choices["text"][0]}  
            - **B:** {choices["text"][1]}  
            - **C:** {choices["text"][2]}  
            - **D:** {choices["text"][3]}  
            
            **Correct answer**: "{answerKey}"  

            **Directly output only one rephrased question** without any other characters or explanatory statements like “The rephrased question is:":
            """


data = []
with open("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcC_Cloze/ARC-Challenge_train.jsonl", 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/ArcC/ARC-Challenge_train_rest.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True) # no need to create the output path mannually
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    if progress <= 462: # delete
        continue # delete
    # if progress > 462: # delete
    #     break # delete

    prompt = generate_prompt(data[i]["question"], data[i]["choices"], data[i]["answerKey"])
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
    result["question"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
