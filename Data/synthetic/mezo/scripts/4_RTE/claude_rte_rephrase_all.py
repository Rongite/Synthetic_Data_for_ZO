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

cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/synthetic/mezo
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
              - The **RTE dataset** focuses on **textual entailment**, where the goal is to determine whether the **hypothesis** is **entailed or contradicted** with respect to the **premise**.
              - The rephrased **premise** must **not alter the logical relationship** between the **premise** and **hypothesis**.
              - The labels are defined as follows:
                - **0 (Entailment)**: The **premise** supports the **hypothesis**.
                - **1 (Contradiction)**: The **premise** contradicts the **hypothesis**.

            2. **Consistency with the Original Label**:
              - Ensure that the logical relationship between the **premise** and **hypothesis** remains unchanged.
              - If the label is **0 (Entailment)**, the rephrased **premise** must still entail the **hypothesis**.
              - If the label is **1 (Contradiction)**, the rephrased **premise** must still contradict the **hypothesis**.

            3. **Optimized for MeZO Training**:
              - **Enhance Gradient Sensitivity**: Enhance the model's **gradient sensitivity** to subtle variations in input by ensuring clear **semantic boundaries** and well-defined relationships in the rephrased data. Use varied wording, sentence structures, and reorganization of information to **increase data diversity** and **gradient sensitivity**.
              - **Focus on Memory Efficiency**: Reduce redundancy and keep the sentence concise to avoid unnecessary noise.
              - **Robustness to Data Sparsity**: Ensure that the scenarios are robust and provide essential information even with minimal context or details, simulating real-world situations where complete data may not always be available.
              - **Non-Differentiable Optimization Readiness**: Create clear impacts on performance metrics with distinct scenarios, ensuring these scenarios are measurable and distinctly influence model optimization targets.

            4. **Maintain Neutral Stance**:
              - Maintain the **neutral stance** of the original **premise**, carefully avoiding any explicit cues about whether the **hypothesis** is **entailed or contradicted**.
              - Ensure that the rephrased **premise** does not explicitly indicate its relationship to the **hypothesis**. The inference must still be a reasoning task based on the **hypothesis**.
              - The rephrased **premise** should only set the stage for inference without resolving or hinting at the correct relationship.
              - Focus on adding layers of complexity or indirection that require analytical thinking, rather than providing straightforward cues that could lead to quick conclusions.

            5. **High-Quality Data Generation**:
              - Produce a **natural, fluent, and coherent** rephrased **premise**.
              - Avoid directly mirroring the original structure; instead, introduce paraphrasing through synonyms, restructuring, or reordering.

            ---

            ### Few-shot Examples:
            Original premise: "No Weapons of Mass Destruction Found in Iraq Yet."
            hypothesis: "Weapons of Mass Destruction Found in Iraq."
            label: "1"
            Rephrased premise: "Weapons of Mass Destruction have not been discovered in Iraq so far."

            Original premise: "A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI."
            hypothesis: "Pope Benedict XVI is the new leader of the Roman Catholic Church."
            label: "0"
            Rephrased premise: "Following the death of Pope John Paul II, a site once filled with mourning transformed into a scene of joy, as Roman Catholic believers assembled in downtown Chicago to celebrate the inauguration of Pope Benedict XVI."

            Original premise: "Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients."
            hypothesis: "Herceptin can be used to treat breast cancer."
            label: "0"
            Rephrased premise: "Herceptin had received approval for use in treating the most severe cases of breast cancer, and on Monday, the company announced plans to consult with federal regulators about expanding its use to a broader range of breast cancer patients."

            Original premise: "Judie Vivian, chief executive at ProMedica, a medical service company that helps sustain the 2-year-old Vietnam Heart Institute in Ho Chi Minh City (formerly Saigon), said that so far about 1,500 children have received treatment."
            hypothesis: "The previous name of Ho Chi Minh City was Saigon."
            label: "0"
            Rephrased premise: "Judie Vivian, who leads ProMedica, a healthcare organization supporting the 2-year-old Vietnam Heart Institute located in Ho Chi Minh City, which was formerly called Saigon, mentioned that approximately 1,500 children have been treated thus far."

            Original premise: "A man is due in court later charged with the murder 26 years ago of a teenager whose case was the first to be featured on BBC One's Crimewatch. Colette Aram, 16, was walking to her boyfriend's house in Keyworth, Nottinghamshire, on 30 October 1983 when she disappeared. Her body was later found in a field close to her home. Paul Stewart Hutchinson, 50, has been charged with murder and is due before Nottingham magistrates later."
            hypothesis: "Paul Stewart Hutchinson is accused of having stabbed a girl."
            label: "1"
            Rephrased premise: "Paul Stewart Hutchinson, aged 50, is expected to appear in Nottingham court, facing charges related to the murder of a teenager that occurred 26 years ago. Colette Aram, who at the age of 16 was headed to her boyfriend's home in Keyworth, Nottinghamshire, vanished on October 30, 1983. Her remains were discovered in a field near her residence. This case marked the first to be highlighted on BBC One's Crimewatch."

            Original premise: "Britain said, Friday, that it has barred cleric, Omar Bakri, from returning to the country from Lebanon, where he was released by police after being detained for 24 hours."
            hypothesis: "Bakri was briefly detained, but was released."
            label: "0"
            Rephrased premise: "On Friday, Britain announced that cleric Omar Bakri is prohibited from re-entering the country after being held for a day and subsequently released by police in Lebanon."

            Original premise: "Nearly 4 million children who have at least one parent who entered the U.S. illegally were born in the United States and are U.S. citizens as a result, according to the study conducted by the Pew Hispanic Center. That's about three quarters of the estimated 5.5 million children of illegal immigrants inside the United States, according to the study. About 1.8 million children of undocumented immigrants live in poverty, the study found."
            hypothesis: "Three quarters of U.S. illegal immigrants have children."
            label: "1"
            Rephrased premise: "Approximately 4 million children, born in the United States to at least one parent who entered the country illegally, are U.S. citizens, as reported by the Pew Hispanic Center. This figure represents about 75% of the estimated 5.5 million children of undocumented immigrants residing in the U.S. The study also indicates that around 1.8 million of these children live in poverty."

            Original premise: "Like the United States, U.N. officials are also dismayed that Aristide killed a conference called by Prime Minister Robert Malval in Port-au-Prince in hopes of bringing all the feuding parties together."
            hypothesis: "Aristide had Prime Minister Robert Malval  murdered in Port-au-Prince."
            label: "1"
            Rephrased premise: "U.N. representatives, much like those from the United States, expressed frustration over Aristide's decision to cancel Prime Minister Robert Malval's conference aimed at uniting conflicting factions in Port-au-Prince."

            Original premise: "WASHINGTON --  A newly declassified narrative of the Bush administration's advice to the CIA on harsh interrogations shows that the small group of Justice Department lawyers who wrote memos authorizing controversial interrogation techniques were operating not on their own but with direction from top administration officials, including then-Vice President Dick Cheney and national security adviser Condoleezza Rice. At the same time, the narrative suggests that then-Defense Secretary Donald H. Rumsfeld and then-Secretary of State Colin Powell were largely left out of the decision-making process."
            hypothesis: "Dick Cheney was the Vice President of Bush."
            label: "0"
            Rephrased premise: "During the Bush administration, a recently revealed account highlights that the Justice Department attorneys drafting memos for contentious interrogation practices were guided by senior administration figures, notably the Vice President at the time, Dick Cheney, along with national security adviser Condoleezza Rice. Meanwhile, the account indicates that key figures such as Defense Secretary Donald H. Rumsfeld and Secretary of State Colin Powell were mostly excluded from the decision-making process."

            Original premise: "Only a week after it had no comment on upping the storage capacity of its Hotmail e-mail service, Microsoft early Thursday announced it was boosting the allowance to 250MB to follow similar moves by rivals such as Google, Yahoo, and Lycos."
            hypothesis: "Microsoft's Hotmail has raised its storage capacity to 250MB."
            label: "0"
            Rephrased premise: "A mere week after remaining silent on increasing the storage capacity of its Hotmail email service, Microsoft revealed early Thursday that it was expanding the limit to 250MB, aligning with similar actions taken by competitors like Google, Yahoo, and Lycos."

            Original premise: "Lina Joy, 42, was born Azlina Jailani to Malay parents, and was raised as a Muslim. Malaysia's constitution guarantees freedom of religion, but by law, all ethnic Malays are Muslim. Joy converted to Christianity at age 26, and after some bureaucratic difficulties had her named legally changed in 1999. However, on her MyKad national ID, the National Registration Department retained her stated religion as Islam. In order to have her religion changed, the National Registration Department said Joy would have to obtain a certificate of apostasy from the Muslim Sharia Court."
            hypothesis: "Lina Joy's parents are from Malaysia."
            label: "0"
            Rephrased premise: "Lina Joy, originally named Azlina Jailani, was born to Malay parents and brought up in the Islamic faith. Although Malaysia's constitution allows for religious freedom, the law mandates that all ethnic Malays are Muslims. At 26, Joy embraced Christianity and, after overcoming some administrative hurdles, legally changed her name in 1999. Despite this, her MyKad national ID still lists her religion as Islam, and the National Registration Department requires her to obtain an apostasy certificate from the Muslim Sharia Court to officially change her religious status."

            Original premise: "November 9, 1989 , the day the Berlin Wall fell and the world changed forever . Not even the most astute saw it coming . As Hungary's foreign minister in the late summer of 1989 , Gyula Horn gave the order to let visiting East Germans use his country to do a 400-mile end run around the Berlin Wall , a move now seen as the beginning of the end for hard-line communism in Europe ."
            hypothesis: "The Berlin Wall was torn down in 1989."
            label: "0"
            Rephrased premise: "On November 9, 1989, the Berlin Wall came down, marking a pivotal moment in history. This event was unforeseen by even the most insightful observers. In the late summer of 1989, Gyula Horn, then Hungary's foreign minister, authorized East German visitors to traverse Hungary, effectively bypassing the Berlin Wall—a decision now recognized as a catalyst for the collapse of strict communist regimes in Europe."

            Original premise: "Valero Energy Corp., on Monday, said it found \"extensive\" additional damage at its 250,000-barrel-per-day Port Arthur refinery."
            hypothesis: "Valero Energy Corp. produces 250,000 barrels per day."
            label: "0"
            Rephrased premise: "Valero Energy Corp. announced on Monday that it discovered significant further damage at its Port Arthur refinery, which has a capacity of 250,000 barrels per day."

            Original premise: "Oil prices fall back as Yukos oil threat lifted"
            hypothesis: "Oil prices rise."
            label: "1"
            Rephrased premise: "The threat concerning Yukos oil has been resolved, leading to a decline in oil prices."

            Original premise: "Brian Brohm, the Louisville quarterback, threw for 368 yards and five touchdowns as the Cardinals beat visiting Oregon State 63-27."
            hypothesis: "The quarterback threw for 413 yards and three touchdowns, and then ran to the end zone two more times."
            label: "1"
            Rephrased premise: "Louisville's quarterback, Brian Brohm, passed for 368 yards and achieved five touchdowns, leading the Cardinals to a 63-27 victory over Oregon State at home."

            Original premise: "Greg Page, a former heavyweight boxing champion who suffered a severe brain injury in a 2001 fight, has died at 50. His wife Patricia said the one-time WBA champion had died at his home in Kentucky, USA, of complications related to injuries he suffered in the fight. Page was in a coma for a week after the 9 March 2001 fight against Dale Crowe which was stopped in the 10th round. Patricia Page said he was \"in a better place now\" after announcing on Monday he had died overnight in his sleep."
            hypothesis: "Greg Page was a WBA champion."
            label: "0"
            Rephrased premise: "Greg Page, once a champion in heavyweight boxing who endured a major brain injury during a fight in 2001, has passed away at 50. His spouse Patricia revealed that the former WBA titleholder succumbed at his residence in Kentucky, USA, due to complications from the injuries acquired in the bout. After the 9th March 2001 fight with Dale Crowe, which ended in the 10th round, Page was in a coma for a week. Patricia Page expressed that he is \"in a better place now,\" after announcing his peaceful passing in his sleep overnight on Monday."

            Original premise: "Sierra is likely to remain in jail at the Hillsborough County jail in her native Tampa until her next hearing on December 20, where she is being held without bail, which would prevent her attending the Washington event on Friday even if she still had permission to perform. Sierra has been in jail since the start of the month after an altercation with police officers outside a Tampa nightclub, which she had been ejected from. She is charged with disorderly intoxication and resisting arrest."
            hypothesis: "Sierra once reached the finals of \"American Idol\"."
            label: "1"
            Rephrased premise: "Sierra is expected to stay in the Hillsborough County jail in Tampa, her hometown, until her upcoming hearing on December 20. She is being detained without bail, which means she cannot attend the Washington event on Friday, even if she were still allowed to perform. Sierra has been incarcerated since early this month following a confrontation with police officers outside a Tampa nightclub from which she was removed. She faces charges of disorderly intoxication and resisting arrest."

            Original premise: "Since 1987, however, Brazil has taken steps to dramatically reduce the destruction, including stepped-up enforcement and the elimination of tax incentives that led to large-scale land clearing."
            hypothesis: "In the early 1990s Brazil began to take action to save the rainforest."
            label: "1"
            Rephrased premise: "Brazil, from 1987 onwards, has implemented measures aimed at significantly curbing deforestation, such as enhancing legal measures and removing tax benefits that previously encouraged widespread land clearance."

            Original premise: "FIFA has received 11 bids to host the 2018 and 2022 FIFA World Cup tournaments, an international football competition contested by the men's national teams. The countries vying to host the tournament are Australia, England, Indonesia, Japan, Mexico, Qatar, Russia, South Korea and United States, who have individual bids and the joint bids are from Belgium-Netherlands and Spain-Portugal.  Select bids are for 2018 and 2022 tournaments and two bids are just for the 2022 tournament. Qatar and South Korea are vying just for the 2022 tournament. The two winning bids will be chosen on December 2010 by the 24-man executive committee of FIFA.  Said FIFA president Sepp Blatter: \"We are very pleased about the fantastic level of interest in our flagship competition, with all initial bidders confirming their candidature.\""
            hypothesis: "Sepp Blatter is the president of FIFA."
            label: "0"
            Rephrased premise: "In pursuit of hosting the 2018 and 2022 FIFA World Cup tournaments, an international competition for men's national football teams, FIFA has attracted 11 bids. The contenders include Australia, England, Indonesia, Japan, Mexico, Qatar, Russia, South Korea, and the United States with their individual bids, while Belgium-Netherlands and Spain-Portugal have submitted joint bids. There are specific bids for both the 2018 and 2022 tournaments, and Qatar along with South Korea are focusing solely on the 2022 event. The decision for the two successful bids will be made in December 2010 by FIFA's 24-member executive committee. FIFA's president, Sepp Blatter, expressed satisfaction with the outstanding interest in their premier tournament, with all initial bidders confirming their participation."

            Original premise: "U.S. crude settled $1.32 lower at $42.83 a barrel."
            hypothesis: "Crude the light American lowered to the closing 1.32 dollars, to 42.83 dollars the barrel."
            label: "1"
            Rephrased premise: "The price of U.S. crude oil closed down by $1.32, reaching $42.83 per barrel."

                        ---

            ### Your Task:
            Rephrase the following **premise** while ensuring that it remains logically consistent with the given **hypothesis** and its **label**. Ensure that the rephrased **premise** enhances **gradient estimation** for **MeZO** by being logically clear, semantically stable, and appropriately diverse.

            **Original premise**: "{premise}"  
            **Hypothesis**: "{hypothesis}"  
            **Label**: "{label}" (0 = **entailment**, 1 = **contradiction**)  

            **Directly output only one rephrased premise** without any other characters or explanatory statements like "The rephrased sentence is:":
            """

data = []
with open('/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/RTE/rte_train.jsonl', 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/synthetic/mezo/RTE/rte_train.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    # if progress <= 20: # delete
    #     continue # delete
    if progress > 500: # delete
        break # delete

    prompt = generate_prompt(data[i]["premise"], data[i]["hypothesis"], data[i]["label"])
    response = client.chat.completions.create( # change
        model="claude-3-7-sonnet-latest",
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
