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
test_set = [
    {
        "original_premise": "The tenant misplaced his keys to his apartment.",
        "rephrased_premise": "The tenant lost track of where his apartment keys were.",
        "choice1": "His landlord unlocked the door.",
        "choice2": "His landlord repaired the door.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "My favorite song came on the radio.",
        "rephrased_premise": "The radio began playing my favorite song.",
        "choice1": "I covered my ears.",
        "choice2": "I sang along to it.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The executive decided not to hire the applicant.",
        "rephrased_premise": "The executive opted against offering employment to the applicant.",
        "choice1": "The applicant failed a background check.",
        "choice2": "The applicant had experience for the job.",
        "question_type": "cause",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The man's eye became infected.",
        "rephrased_premise": "A man's eye developed an infection.",
        "choice1": "He went blind.",
        "choice2": "He put on glasses.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The bird couldn't fly.",
        "rephrased_premise": "The bird was unable to take flight.",
        "choice1": "It migrated for the winter.",
        "choice2": "It injured its wing.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The girl made a wish.",
        "rephrased_premise": "A wish was made by the girl.",
        "choice1": "She saw a black cat.",
        "choice2": "She saw a shooting star.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The woman shivered as she got out the pool.",
        "rephrased_premise": "\nStepping out of the pool, the woman felt a chill run through her body.",
        "choice1": "She wrapped herself in a towel.",
        "choice2": "She poured herself some lemonade.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The nurse prepared the needle for the patient's injection.",
        "rephrased_premise": "The nurse readied the syringe for the patient's shot.",
        "choice1": "The patient bled.",
        "choice2": "The patient tensed up.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man threw out the bread.",
        "rephrased_premise": "The man discarded the bread.",
        "choice1": "It was fresh.",
        "choice2": "It was stale.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The children knocked over a lamp.",
        "rephrased_premise": "The children caused a lamp to topple over.",
        "choice1": "They had a pillow fight.",
        "choice2": "They jumped on the bed.",
        "question_type": "cause",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "I drank from the water fountain.",
        "rephrased_premise": "I took a sip from the water fountain.",
        "choice1": "I was thirsty.",
        "choice2": "I felt nauseous.",
        "question_type": "cause",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The homeowners disliked their nosy neighbors.",
        "rephrased_premise": "The homeowners were not fond of their inquisitive neighbors.",
        "choice1": "They built a fence around their property.",
        "choice2": "They hosted a barbeque in their backyard.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The bodybuilder lifted weights.",
        "rephrased_premise": "A bodybuilder engaged in weightlifting.",
        "choice1": "The gym closed.",
        "choice2": "Her muscles became fatigued.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The cook stirred the ingredients in the bowl.",
        "rephrased_premise": "The chef mixed the contents of the bowl.",
        "choice1": "The ingredients melted.",
        "choice2": "The ingredients blended together.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man signed the document.",
        "rephrased_premise": "The man put his signature on the document.",
        "choice1": "The transaction was voided.",
        "choice2": "The transaction became official.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The police officer dropped the gun.",
        "rephrased_premise": "The gun slipped from the police officer's hands.",
        "choice1": "The gun recoiled.",
        "choice2": "The gun went off.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The woman felt compelled to help someone in need.",
        "rephrased_premise": "Driven by a sense of duty, the woman took action to assist someone in distress.",
        "choice1": "She donated blood.",
        "choice2": "She wrote a poem.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The woman felt lonely.",
        "rephrased_premise": "The woman experienced a sense of solitude.",
        "choice1": "She renovated her kitchen.",
        "choice2": "She adopted a cat.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "I rubbed sandpaper on the wood.",
        "rephrased_premise": "Applying sandpaper to the wood surface.",
        "choice1": "The wood became smooth.",
        "choice2": "The wood became sticky.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The crowd gave the band a standing ovation.",
        "rephrased_premise": "The audience rose to their feet in applause for the band.",
        "choice1": "The band signed autographs.",
        "choice2": "The band reappeared on the stage.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man threw his empty can onto the street.",
        "rephrased_premise": "The man discarded his empty can onto the street.",
        "choice1": "He was jumped from behind.",
        "choice2": "He was fined for littering.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "My stomach growled.",
        "rephrased_premise": "Hunger pangs emanated from my stomach.",
        "choice1": "I forgot to eat breakfast.",
        "choice2": "I was full from breakfast.",
        "question_type": "cause",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The boaters set off a flare.",
        "rephrased_premise": "A flare was launched by the boaters.",
        "choice1": "Their boat drifted to shore.",
        "choice2": "Their boat was rescued.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man removed his coat.",
        "rephrased_premise": "The man took off his coat.",
        "choice1": "He entered the house.",
        "choice2": "He loosened his tie.",
        "question_type": "cause",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The family took their dog to the veterinarian.",
        "rephrased_premise": "The family brought their dog to the animal doctor.",
        "choice1": "The dog chewed on a bone.",
        "choice2": "The dog injured his paw.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The woman dangled the biscuit above the dog.",
        "rephrased_premise": "The woman held a biscuit above the dog's head.",
        "choice1": "The dog jumped up.",
        "choice2": "The dog scratched its fur.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "I learned how to play the board game.",
        "rephrased_premise": "Knowledge of the board game was gained by me.",
        "choice1": "My friend explained the rules to me.",
        "choice2": "My friend got the rules wrong.",
        "question_type": "cause",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The man went away for the weekend.",
        "rephrased_premise": "The man left town for the weekend.",
        "choice1": "He wanted to relax.",
        "choice2": "He felt content.",
        "question_type": "cause",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The shop was closed.",
        "rephrased_premise": "The store was not open.",
        "choice1": "The owner was helping customers.",
        "choice2": "The shop was undergoing renovation.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The boy tuned the radio.",
        "rephrased_premise": "The boy adjusted the radio settings.",
        "choice1": "The station was playing rock music.",
        "choice2": "The station was coming in with static.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The terrorist set off the bomb.",
        "rephrased_premise": "A bomb was triggered by the terrorist.",
        "choice1": "The bomb exploded.",
        "choice2": "The bomb was deactivated.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The police handcuffed the suspect.",
        "rephrased_premise": "The suspect was placed in handcuffs by the police.",
        "choice1": "The police called for backup.",
        "choice2": "The suspect resisted arrest.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The authorities vowed to protect the identity of the crime victim.",
        "rephrased_premise": "The officials committed to keeping the crime victim's identity confidential.",
        "choice1": "The victim struggled to recall details about the crime.",
        "choice2": "They withheld the victim's name from the public.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man's clothes fit loosely.",
        "rephrased_premise": "The man's clothes appeared oversized on him.",
        "choice1": "He bought them on sale.",
        "choice2": "He lost weight.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The clock stopped ticking.",
        "rephrased_premise": "The clock ceased its ticking motion.",
        "choice1": "I took extra time to get ready.",
        "choice2": "The clock showed the wrong time.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man closed the umbrella.",
        "rephrased_premise": "The umbrella was folded by the man.",
        "choice1": "He got out of the car.",
        "choice2": "He approached the building.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man craved a cigarette.",
        "rephrased_premise": "A cigarette was intensely desired by the man.",
        "choice1": "His family urged him to quit smoking.",
        "choice2": "He was addicted to nicotine.",
        "question_type": "cause",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The man dropped food on the floor.",
        "rephrased_premise": "Food fell from the man's hands onto the floor.",
        "choice1": "His dog jumped up on him.",
        "choice2": "His dog ran over to eat the food.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    },
    {
        "original_premise": "The girl was angry with her friend.",
        "rephrased_premise": "The girl felt upset with her friend.",
        "choice1": "The girl spread a rumor about her friend.",
        "choice2": "The girl told a secret to her friend.",
        "question_type": "effect",
        "correct_answer": "Choice 1",
        "eval": "same"
    },
    {
        "original_premise": "The fugitive hid from the police.",
        "rephrased_premise": "The fugitive concealed themselves to avoid police detection.",
        "choice1": "The police dropped the case.",
        "choice2": "The fugitive remained at large.",
        "question_type": "effect",
        "correct_answer": "Choice 2",
        "eval": "same"
    }
]


def few_shot(i, example):
    return example[i]

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

correct_answer = 0
total_answer = 0

for i in tqdm(range(len(test_set))):
    output_data = {}
    prompt = generate_prompt(test_set[i]["original_premise"], test_set[i]["choice1"], test_set[i]["choice2"], test_set[i]["question_type"], \
                    test_set[i]["correct_answer"], test_set[i]["rephrased_premise"])
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0 
    )
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
