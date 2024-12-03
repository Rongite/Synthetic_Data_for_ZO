import json
from openai import OpenAI
from tqdm import tqdm
import os

client = OpenAI(
    # api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    base_url = "https://api2.aigcbest.top/v1"
)

# Example hypothesis and reference
test_set = [
    {
        "original_premise": "The tenant misplaced his keys to his apartment.",
        "rephrased_premise": "The tenant lost his apartment keys.",
        "choice1": "His landlord unlocked the door.",
        "choice2": "His landlord repaired the door.",
        "question_type": "effect",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "My favorite song came on the radio.",
        "rephrased_premise": "The radio played my favorite song.",
        "choice1": "I covered my ears.",
        "choice2": "I sang along to it.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The executive decided not to hire the applicant.",
        "rephrased_premise": "The executive chose not to employ the applicant.",
        "choice1": "The applicant failed a background check.",
        "choice2": "The applicant had experience for the job.",
        "question_type": "cause",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "The man's eye became infected.",
        "rephrased_premise": "The man's eye developed an infection.",
        "choice1": "He went blind.",
        "choice2": "He put on glasses.",
        "question_type": "effect",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "The bird couldn't fly.",
        "rephrased_premise": "The bird was unable to fly.",
        "choice1": "It migrated for the winter.",
        "choice2": "It injured its wing.",
        "question_type": "cause",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The girl made a wish.",
        "rephrased_premise": "The girl wished upon seeing a shooting star.",
        "choice1": "She saw a black cat.",
        "choice2": "She saw a shooting star.",
        "question_type": "cause",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The woman shivered as she got out the pool.",
        "rephrased_premise": "Exiting the pool, the woman shivered.",
        "choice1": "She wrapped herself in a towel.",
        "choice2": "She poured herself some lemonade.",
        "question_type": "effect",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "The nurse prepared the needle for the patient's injection.",
        "rephrased_premise": "The nurse set up the needle for administering the patient's injection.",
        "choice1": "The patient bled.",
        "choice2": "The patient tensed up.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The man threw out the bread.",
        "rephrased_premise": "The man discarded the bread.",
        "choice1": "It was fresh.",
        "choice2": "It was stale.",
        "question_type": "cause",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The children knocked over a lamp.",
        "rephrased_premise": "A lamp was knocked over by the children.",
        "choice1": "They had a pillow fight.",
        "choice2": "They jumped on the bed.",
        "question_type": "cause",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "I drank from the water fountain.",
        "rephrased_premise": "I took a drink from the water fountain.",
        "choice1": "I was thirsty.",
        "choice2": "I felt nauseous.",
        "question_type": "cause",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "The homeowners disliked their nosy neighbors.",
        "rephrased_premise": "The homeowners were annoyed by their prying neighbors.",
        "choice1": "They built a fence around their property.",
        "choice2": "They hosted a barbeque in their backyard.",
        "question_type": "effect",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "The bodybuilder lifted weights.",
        "rephrased_premise": "The bodybuilder engaged in weightlifting.",
        "choice1": "The gym closed.",
        "choice2": "Her muscles became fatigued.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The cook stirred the ingredients in the bowl.",
        "rephrased_premise": "The cook mixed the bowl's contents.",
        "choice1": "The ingredients melted.",
        "choice2": "The ingredients blended together.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The man signed the document.",
        "rephrased_premise": "The man put his signature on the document.",
        "choice1": "The transaction was voided.",
        "choice2": "The transaction became official.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The police officer dropped the gun.",
        "rephrased_premise": "The gun was dropped by the police officer.",
        "choice1": "The gun recoiled.",
        "choice2": "The gun went off.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "The woman felt compelled to help someone in need.",
        "rephrased_premise": "The woman felt driven to assist someone who required help.",
        "choice1": "She donated blood.",
        "choice2": "She wrote a poem.",
        "question_type": "effect",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "The woman felt lonely.",
        "rephrased_premise": "The woman experienced loneliness.",
        "choice1": "She renovated her kitchen.",
        "choice2": "She adopted a cat.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    },
    {
        "original_premise": "I rubbed sandpaper on the wood.",
        "rephrased_premise": "I applied sandpaper to the wood.",
        "choice1": "The wood became smooth.",
        "choice2": "The wood became sticky.",
        "question_type": "effect",
        "correct_answer": "choice1",
        "eval": "same"
    },
    {
        "original_premise": "The crowd gave the band a standing ovation.",
        "rephrased_premise": "The audience stood to applaud the band.",
        "choice1": "The band signed autographs.",
        "choice2": "The band reappeared on the stage.",
        "question_type": "effect",
        "correct_answer": "choice2",
        "eval": "same"
    }
]

def few_shot(i, example):
    return example[i]

def generate_prompt(original_premise, choice1, choice2, question, correct_choice, rephrased_premise):
    return f"""
            You are tasked with judging if the rephrased premise has the same meaning as the original premise and checking if the rephrased premise remains \
                consistent with its associated question and choices. If the rephrased premise remains consistent with its associated questions and options, the \
                    choice inferred from the rephrased premise will be the same as the choice inferred from the original premise, meaning that the correct answer \
                        remains the same. Below are the correctly rephrased examples:

            ### Few-shot Examples:
            
            Original premise: "The girl received a trophy."
            Choice 1: "She won a spelling bee."
            Choice 2: "She made a new friend."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The girl was awarded a trophy."

            Original premise: "The woman's date wanted to look like a gentleman."
            Choice 1: "He opened the door for her."
            Choice 2: "He asked her if she liked sushi."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The woman's date aimed to present himself as a gentleman."

            Original premise: "The farmland needed irrigation."
            Choice 1: "A canal was constructed."
            Choice 2: "A flood occurred."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The farmland required a water supply."

            Original premise: "The host cancelled the party."
            Choice 1: "She was certain she had the flu."
            Choice 2: "She worried she would catch the flu."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The host called off the party."

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
            Rephrased premise: "The skydiver landed safely on the ground."

            Original premise: "The toddler became cranky."
            Choice 1: "Her mother put her down for a nap."
            Choice 2: "Her mother fixed her hair into pigtails."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The toddler got irritable."

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
            Rephrased premise: "The grape juice underwent fermentation."

            Original premise: "The friends' debate dragged on interminably."
            Choice 1: "The friends saw eye to eye."
            Choice 2: "The friends were splitting hairs."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The friends' discussion seemed endless."

            Original premise: "The woman hummed to herself."
            Choice 1: "She was nervous."
            Choice 2: "She was in a good mood."
            Question type: "cause"
            Correct answer: Choice 2
            Rephrased premise: "The woman quietly sang to herself."

            Original premise: "The man hated his new haircut."
            Choice 1: "He wore a hat."
            Choice 2: "He grew a beard."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The man disliked his new haircut."

            Original premise: "The police aimed their weapons at the fugitive."
            Choice 1: "The fugitive fell to the ground."
            Choice 2: "The fugitive dropped his gun."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The police pointed their guns at the fugitive."

            Original premise: "The patient was dehydrated."
            Choice 1: "The nurse tested his reflexes."
            Choice 2: "The nurse gave him an IV."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The patient experienced dehydration."

            Original premise: "The girl found the missing puzzle piece."
            Choice 1: "She completed the puzzle."
            Choice 2: "She took apart the puzzle."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The girl discovered the lost puzzle piece."

            Original premise: "The man urgently leaped out of bed."
            Choice 1: "He wanted to shut off the alarm clock."
            Choice 2: "He wanted to iron his pants before work."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The man quickly sprang out of bed."

            Original premise: "The papers were disorganized."
            Choice 1: "I made photocopies of them."
            Choice 2: "I put them into alphabetical order."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The papers were in disarray."

            Original premise: "The woman won the lottery."
            Choice 1: "She bought a yacht."
            Choice 2: "She joined a church."
            Question type: "effect"
            Correct answer: Choice 1
            Rephrased premise: "The woman hit the jackpot in the lottery."

            Original premise: "The seamstress pushed the threaded needle into the fabric."
            Choice 1: "The thread wrapped around the needle."
            Choice 2: "The thread went through the fabric."
            Question type: "effect"
            Correct answer: Choice 2
            Rephrased premise: "The seamstress inserted the threaded needle into the fabric."

            Original premise: "The woman hired a lawyer."
            Choice 1: "She decided to sue her employer."
            Choice 2: "She decided to run for office."
            Question type: "cause"
            Correct answer: Choice 1
            Rephrased premise: "The woman engaged legal counsel."

            ---

            Your task:
            For the given information below, determine whether the Rephrased premise and Original premise are the same. If the meaning and tone of the Rephrased \
                premise and the Original premise are consistent, and the rephrased premise remains consistent with the provided questions and choices so that the \
                    Correct answer remains unchanged, then the rephrased premise and the original premise can be determined to be the same, otherwise they are not \
                        the same.

            Original premise: "{original_premise}"
            Choice 1: "{choice1}"
            Choice 2: "{choice2}"
            Question type: "{question}" (cause or effect)
            Correct answer: "{correct_choice}"
            Rephrased premise: "{rephrased_premise}"

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
