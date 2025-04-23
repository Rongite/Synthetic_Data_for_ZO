### evaluate using open ai api
import json
from openai import OpenAI
from tqdm import tqdm
import os

client = OpenAI(
    # api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    base_url = "https://api2.aigcbest.top/v1"
)

eval_list = []

def generate_prompt(premise, hypothesis, label, rephrased_premise):
    return f"""
            Task Description:

            You are tasked with verifying whether a rephrased premise maintains consistency with its associated **hypothesis**. Specificqally, \
                determine if the Rephrased premise remains logically consistent with the given **hypothesis** and its **label** as the same as the Original \
                    premise. Below are correctly rephrased examples to guide you:

            ### Few-shot Examples:
            
            Original premise: "WINNENDEN, Germany \u2015A teenage gunman killed 15 people, most of them female, on Wednesday in a rampage that began at a school near Stuttgart in southern Germany and ended in a nearby town, where he then killed himself after the police wounded him. The attack left Germany, which tightened tough gun controls after a similar attack at a school seven years ago, struggling to understand the carnage that had again befallen it, a country with relatively little violent crime. In 2002, a gunman killed 16 people before killing himself at a school in Erfurt, in eastern Germany."
            hypothesis: "In 2002 near Stuttgart a boy shot 16 people."
            label: "1"
            Rephrased premise: "In Winnenden, Germany, a teenage shooter took the lives of 15 individuals, predominantly female, on Wednesday. The violent spree started at a school near Stuttgart in southern Germany and concluded in a nearby town, where the gunman ended his own life after being injured by police. This incident left Germany, which had already implemented stringent gun control measures following a similar school attack seven years prior, grappling with the tragedy in a nation known for its low levels of violent crime. In 2002, a shooter in Erfurt, eastern Germany, killed 16 people before committing suicide."

            Original premise: "Many hopes are riding on the sale of Talisman's holdings in Palm Beach and Hendry counties, which Vice President Al Gore announced with much fanfare last year at the 50th anniversary of Everglades National Park."
            hypothesis: "Everglades National Park is located in Florida."
            label: "1"
            Rephrased premise: "Vice President Al Gore, with considerable publicity, announced last year the anticipated sale of Talisman's properties in Palm Beach and Hendry counties during the 50th anniversary celebration of Everglades National Park."

            Original premise: "Rabies virus infects the central nervous system, causing encephalopathy and ultimately death. Early symptoms of rabies in humans are nonspecific, consisting of fever, headache, and general malaise."
            hypothesis: "Rabies is fatal in humans."
            label: "0"
            Rephrased premise: "The rabies virus targets the central nervous system, leading to encephalopathy and eventually resulting in death. Initial symptoms in humans are vague, including fever, headache, and a general feeling of discomfort."

            Original premise: "American tobacco companies were showing a profit most quarters due to export sales of cigarettes and diversification of products sold including food."
            hypothesis: "PM often entered markets with both cigarettes and food."
            label: "1"
            Rephrased premise: "Tobacco companies in America frequently reported profits in most quarters, attributed to cigarette exports and a diverse range of products, including food."

            Original premise: "The development of agriculture by early humans, roughly 10,000 years ago, was also harmful to many natural ecosystems as they were systematically destroyed and replaced with artificial versions."
            hypothesis: "Humans existed 10,000 years ago."
            label: "0"
            Rephrased premise: "Approximately 10,000 years ago, early humans initiated agricultural practices, which led to the systematic destruction and replacement of numerous natural ecosystems with artificial ones."

            Original premise: "The two young leaders of the coup, Pibul Songgram and Pridi Phanomyang, both educated in Europe and influenced by Western ideas, came to dominate Thai politics in the ensuing years."
            hypothesis: "Pibul was a young leader."
            label: "0"
            Rephrased premise: "Educated in Europe and inspired by Western ideologies, Pibul Songgram and Pridi Phanomyang, the youthful leaders of the coup, rose to prominence in Thai politics in the following years."

            Original premise: "Lin Piao, after all, was the creator of Mao's \"Little Red Book\" of quotations."
            hypothesis: "Lin Piao wrote the \"Little Red Book\"."
            label: "0"
            Rephrased premise: "Lin Piao was indeed the mastermind behind the compilation of Mao's \"Little Red Book\" of quotations."

            Original premise: "When patients interrupt a course of antibiotics, the surviving bacteria return with a vengeance, often having rapidly mutated to resist the therapy."
            hypothesis: "Bacteria is winning the war against antibiotics."
            label: "1"
            Rephrased premise: "\"Discontinuing antibiotics allows the remaining bacteria to come back aggressively, frequently having quickly evolved to withstand the treatment.\""

            Original premise: "Initially the Bundesbank opposed the introduction of the euro but was compelled to accept it in light of the political pressure of the capitalist politicians who supported its introduction."
            hypothesis: "The introduction of the euro has been opposed."
            label: "0"
            Rephrased premise: "The Bundesbank was initially against the euro's introduction but had to concede due to the political influence exerted by capitalist politicians who favored its adoption."

            Original premise: "The Armed Forces Press Committee (COPREFA) admitted that the government troops sustained 11 casualties in these clashes, adding that they inflicted three casualties on the rebels."
            hypothesis: "Three rebels were killed by government troops."
            label: "0"
            Rephrased premise: "The Armed Forces Press Committee (COPREFA) acknowledged that government forces suffered 11 casualties during these confrontations, while also reporting that they caused three casualties among the rebels."

            Original premise: "One economic study will not be the basis of Canada's public policy decisions, but Easton's research does conclusively show that there are economic benefits in the legalization of marijuana."
            hypothesis: "Drug legalization has benefits."
            label: "0"
            Rephrased premise: "Easton's research definitively demonstrates the economic advantages of legalizing marijuana, although a single economic study will not solely determine Canada's public policy decisions."

            Original premise: "Archaeologists have found approximately 30 beautifully preserved mummies in a 4,000 year old Egyptian necropolis which held 53 tombs. Supervisor of Antiquities for Middle Egypt Dr. Abdel-Rahman El-Ayedi's team established his archaeological site in the Faiyum Oasis near the El-Lahun Egyptian pyramid which is just south of Cairo. Besides the mummies, the team found masks, amulets, clay pots and an offering table located in a funerary chapel. The chapel dates back to about 30 BC to 337 AD."
            hypothesis: "30 beautifully preserved mummies have been located in the south of Cairo."
            label: "0"
            Rephrased premise: "In a 4,000-year-old Egyptian necropolis containing 53 tombs, archaeologists have uncovered around 30 remarkably well-preserved mummies. Dr. Abdel-Rahman El-Ayedi, the Supervisor of Antiquities for Middle Egypt, led his team to establish the archaeological site in the Faiyum Oasis, near the El-Lahun Egyptian pyramid, located just south of Cairo. Alongside the mummies, the team discovered masks, amulets, clay pots, and an offering table within a funerary chapel, which dates from approximately 30 BC to 337 AD."

            Original premise: "Gastrointestinal bleeding can happen as an adverse effect of non-steroidal anti-inflammatory drugs such as aspirin or ibuprofen."
            hypothesis: "Aspirin prevents gastrointestinal bleeding."
            label: "1"
            Rephrased premise: "Non-steroidal anti-inflammatory drugs, including aspirin and ibuprofen, can cause gastrointestinal bleeding as a side effect."

            Original premise: "A former employee of the company, David Vance of South Portland, said Hooper spent a lot of time on the road, often meeting with customers between Portland and Kittery."
            hypothesis: "Hooper is a citizen of Portland."
            label: "1"
            Rephrased premise: "David Vance, a former company employee from South Portland, mentioned that Hooper frequently traveled, regularly engaging with clients from Portland to Kittery."

            Original premise: "Cyprus, divided or not, joins the EU on the 1st of May."
            hypothesis: "Cyprus was divided into two parts on May 1."
            label: "1"
            Rephrased premise: "Cyprus, regardless of its division status, became a member of the EU on May 1."

            Original premise: "World leaders expressed concern on Thursday that North Korea will quit six-party nuclear disarmament talks and will bolster its nuclear weapons arsenal."
            hypothesis: "North Korea says it has a stockpile of nuclear weapons and is building more."
            label: "1"
            Rephrased premise: "World leaders voiced apprehension on Thursday about North Korea's potential withdrawal from the six-party nuclear disarmament discussions and its intention to expand its nuclear weapons stockpile."

            Original premise: "In addition to establishing the electoral commission, the laws also concern nationality, a contentious issue since citizenship laws were tightened to exclude one of Gbagbo's main competitors from the 2000 presidential race, former prime minister Alassane Ouattara."
            hypothesis: "Gbagbo is a competitor of Ouattara."
            label: "0"
            Rephrased premise: "The laws, besides setting up the electoral commission, also address nationality, a divisive topic since citizenship regulations were tightened to exclude Alassane Ouattara, a key rival of Gbagbo, from the 2000 presidential election."

            Original premise: "Mohandas Karamchand Gandhi never received the Nobel Peace Prize, though he was nominated for it five times between 1937 and 1948."
            hypothesis: "Mohandas received the Nobel Prize in 1989."
            label: "1"
            Rephrased premise: "Mohandas Karamchand Gandhi was nominated for the Nobel Peace Prize on five occasions from 1937 to 1948, yet he never won the award."

            Original premise: "The Christian Democrats (CDU) won 35.2% of the vote, or 225 seats, against 34.3% for Chancellor Gerhard Schroeder's Social Democrats (SPD)."
            hypothesis: "It seems unlikely that there will be a coalition between Gerhard Schroeder's Social Democrats and Angela Merkel's Christian Democratic Union."
            label: "1"
            Rephrased premise: "The Christian Democratic Union (CDU) secured 35.2% of the votes, translating to 225 seats, while Chancellor Gerhard Schroeder's Social Democratic Party (SPD) garnered 34.3%."

            Original premise: "Edward VIII became King in January of 1936 and abdicated in December."
            hypothesis: "King Edward VIII abdicated in December 1936."
            label: "0"
            Rephrased premise: "Edward VIII ascended to the throne in January 1936 and stepped down in December of the same year."

            ---

            Your Task:

            Evaluate the following data and determine whether the rephrased premise maintains consistency with the original premise regarding the given \
                **hypothesis** and its **label** choice.

            **Original premise**: "{premise}"  
            **Hypothesis**: "{hypothesis}"  
            **Label**: "{label}" (0 = **entailment**, 1 = **contradiction**) 
            **Rephrased premise**: "{rephrased_premise}"

            Example Input:
            **Original premise**: "WINNENDEN, Germany \u2015A teenage gunman killed 15 people, most of them female, on Wednesday in a rampage that began at a school near Stuttgart in southern Germany and ended in a nearby town, where he then killed himself after the police wounded him. The attack left Germany, which tightened tough gun controls after a similar attack at a school seven years ago, struggling to understand the carnage that had again befallen it, a country with relatively little violent crime. In 2002, a gunman killed 16 people before killing himself at a school in Erfurt, in eastern Germany."
            **Hypothesis**: "In 2002 near Stuttgart a boy shot 16 people." 
            **Label**: "1" (0 = **entailment**, 1 = **contradiction**) 
            **Rephrased premise**: "In Winnenden, Germany, a teenage shooter took the lives of 15 individuals, predominantly female, on Wednesday. The violent spree started at a school near Stuttgart in southern Germany and concluded in a nearby town, where the gunman ended his own life after being injured by police. This incident left Germany, which had already implemented stringent gun control measures following a similar school attack seven years prior, grappling with the tragedy in a nation known for its low levels of violent crime. In 2002, a shooter in Erfurt, eastern Germany, killed 16 people before committing suicide."

            Expected Output: same

            Directly output [same/not the same] without any explanation:

            """


eval_list = []
count = 0
with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/RTE/rte_train.jsonl', 'r', encoding="utf-8") as f1: # modify
    for line in f1:
        data = json.loads(line)
        temp = {}
        temp["original"] = data["premise"]
        temp["hypothesis"] = data["hypothesis"]
        temp["idx"] = data["idx"]
        temp["label"] = data["label"]
        eval_list.append(temp)


with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/RTE/rte_train.jsonl', 'r', encoding="utf-8") as f2: # modify
    for line in f2:
        data = json.loads(line)
        eval_list[count]["rephrased"] = data["premise"]
        count += 1

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/RTE/rte_train.jsonl") # output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")

correct_answer = 0
total_answer = 0

for i in tqdm(range(len(eval_list))):
    output_data = {}
    if 20 <= i < 40:
        eval_list[i]['eval_result'] = "same"
        output_data["premise"] = eval_list[i]["rephrased"]
        output_data["hypothesis"] = eval_list[i]["hypothesis"]
        output_data["idx"] = eval_list[i]["idx"]
        output_data["label"] = eval_list[i]["label"]
        correct_answer += 1
        total_answer += 1
        out_file.write(json.dumps(output_data) + "\n")
        out_file.flush()
        continue

    prompt = generate_prompt(eval_list[i]["original"], eval_list[i]["hypothesis"], eval_list[i]["label"], eval_list[i]["rephrased"])
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
        output_data["hypothesis"] = eval_list[i]["hypothesis"]
        output_data["idx"] = eval_list[i]["idx"]
        output_data["label"] = eval_list[i]["label"]
        total_answer += 1
        out_file.write(json.dumps(output_data) + "\n")
        out_file.flush()
        continue
    
    # print(response)
    eval_list[i]['eval_result'] = response.choices[0].message.content # change
    output_data["premise"] = eval_list[i]["rephrased"]
    output_data["hypothesis"] = eval_list[i]["hypothesis"]
    output_data["idx"] = eval_list[i]["idx"]
    output_data["label"] = eval_list[i]["label"]
    correct_answer += 1
    total_answer += 1
    out_file.write(json.dumps(output_data) + "\n")
    out_file.flush()

acc = correct_answer / total_answer
print("The prompt accuracy is ", acc)
