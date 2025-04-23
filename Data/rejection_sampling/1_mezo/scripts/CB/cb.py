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
            
            Original premise: "Chopra stood unsteadily on his feet. The shapechanger bounded around with excitement. Chopra could tell something had happened."
            hypothesis: "something had happened"
            label: "0"
            Rephrased premise: "Chopra wobbled as he stood, while the shapechanger leapt about energetically. It was evident to Chopra that an event had occurred."

            Original premise: "It seemed impossible that anyone could endure such pain for so long, but at last the doors of the Renault slammed and there was comparative silence. The engine was started up, revving violently as the car was turned round on the narrow road. John could tell that it was being driven back up the hill towards Putna."
            hypothesis: "the car was being driven back up the hill towards Putna"
            label: "0"
            Rephrased premise: "The pain seemed unbearable for such an extended period, yet eventually, the Renault's doors slammed shut, bringing a relative quiet. The engine roared to life, revving intensely as the vehicle maneuvered on the narrow road. John could discern that it was heading back up the hill towards Putna."

            Original premise: "Just when you think you 've got it straight, along comes the Fool with his pig's bladder and whops you on the nose. By the way, I'm no idiot. I could tell Gillian and Stuart weren't thrilled to see me at the airport."
            hypothesis: "Gillian and Stuart weren't thrilled to see her at the airport"
            label: "0"
            Rephrased premise: "When you think everything is clear, the Fool appears with his pig's bladder and surprises you. For the record, I'm not foolish. I noticed that Gillian and Stuart were less than pleased to see me at the airport."

            Original premise: "He's weird enough to have undressed me without thinking, according to some mad notion of the ``proper'' thing to do. Perhaps he thought I couldn't lie in bed with my clothes on."
            hypothesis: "she couldn't lie in bed with her clothes on"
            label: "1"
            Rephrased premise: "He is peculiar enough to have disrobed me based on some irrational idea of what is \"appropriate.\" Maybe he believed it was impossible for me to remain in bed fully dressed."

            Original premise: "That is to say, I did not take sufficient account of the fact that at that time of the day, what Mr Farraday enjoys is a conversation of a lighthearted, humorous sort. Knowing this to be his likely mood when I brought in the tea yesterday afternoon, and being aware of his general propensity to talk with me in a bantering tone at such moments, it would certainly have been wiser not to have mentioned Miss Kenton at all. But you will perhaps understand that there was a natural tendency on my part in asking what was after all a generous favour from my employer to hint that there was a good professional motive behind my request."
            hypothesis: "there was a natural tendency on her part to hint that there was a good professional motive behind her request"
            label: "0"
            Rephrased premise: "I failed to adequately consider that Mr. Farraday prefers lighthearted and humorous conversations at that time of day. Aware of his usual inclination to engage in playful banter when I served tea yesterday afternoon, it would have been more prudent not to mention Miss Kenton. However, you might understand that when requesting a generous favor from my employer, there was a natural inclination to suggest a valid professional reason for my request."

            Original premise: "But don't dilly-dally for too long. Once it's published we are all going to look a little risible if we have made no adjustments to what is after all known as being predominantly my own design of gallery. Also I am a bit older than the rest of you but you can perhaps understand that I don't want to drop dead without a proper and public recantation."
            hypothesis: "he doesn't want to drop dead without a proper and public recantation"
            label: "0"
            Rephrased premise: "Please don't delay for too long. Once it's released, we'll all appear somewhat foolish if we haven't made any changes to what is largely recognized as my own gallery design. Additionally, I'm a bit older than the rest of you, and you might understand that I wish to avoid passing away without making a proper and public retraction."

            Original premise: "I should dearly have liked to know whether they were Europeans or Americans, but I couldn't hear the accents. They appeared to be arguing. I hoped the white men weren't telling him to eliminate all witnesses because I don't believe it would have needed much persuasion."
            hypothesis: "eliminating all witnesses would have needed much persuasion"
            label: "1"
            Rephrased premise: "I was quite curious to find out if they were Europeans or Americans, but their accents were indistinct. They seemed to be in a dispute. I hoped the white men weren't suggesting he eliminate all witnesses, as I doubt it would have required much convincing."

            Original premise: "But the damage was done as far as my faith was concerned, which is probably why I went mad. So anyway, that Christmas Eve night confirmed my worst fears, it was like a kind of ``royal flush'' for the infant Jimbo. All three kings - Pa Santa and the King of Kings - all down the pan together... And to be honest I don't believe any of them stands a chance of ever making a comeback with me."
            hypothesis: "any of the three kings stands a chance of ever making a comeback with him"
            label: "1"
            Rephrased premise: "The damage to my faith was irreversible, likely leading to my breakdown. That Christmas Eve only solidified my deepest fears, akin to a \"royal flush\" for young Jimbo. All three figures - Pa Santa, the King of Kings - were dismissed simultaneously, and frankly, I doubt any of them will ever regain my belief."

            Original premise: "He had seen something I should have, which was a car turning in from Soho Square and coming up behind me. My right foot hovered over the accelerator pedal and I balanced Armstrong on the clutch. I wasn't as convinced as Malpass that Nevil was out of harm's way."
            hypothesis: "Nevil was out of harm's way"
            label: "1"
            Rephrased premise: "Having noticed something I should have, he saw a car turning in from Soho Square and approaching from behind. My right foot hovered over the accelerator while I kept Armstrong balanced on the clutch. Unlike Malpass, I wasn't convinced that Nevil was safe."

            Original premise: "Jed wondered. He 'd scarcely set eyes on him since the night they 'd had dinner together at the house in Westwood. Nobody had mentioned him either and Jed didn't feel he should ask."
            hypothesis: "Jed should ask"
            label: "1"
            Rephrased premise: "Jed was curious. He hadn't seen him since their dinner at the Westwood house, and no one had brought him up in conversation. Jed felt it wasn't his place to inquire."

            Original premise: "Yet what good came from knowing that a woman had killed herself? The children who had suffered a trauma would survive the experience, scarred by it and a little flawed by it. They would never forget that for a week they had imagined the act of murder had been committed."
            hypothesis: "for a week the children had imagined the act of murder had been committed"
            label: "0"
            Rephrased premise: "\"What benefit was there in knowing a woman had taken her own life? The children, having endured a traumatic event, would carry the scars and imperfections from it. They would always remember that for a week, they had believed a murder had occurred.\""

            Original premise: "Nora calculated that there must be lots of single men up there so she decided it was ideal for the `manhunt'', as we called it, even though the train fare was a serious consideration. I remember we went up to Euston together one Saturday morning, very excited, to buy the ticket in advance. It was a secret - if my parents had known she was going away alone they would have soon put the kybosh on it."
            hypothesis: "Nora was going away alone"
            label: "0"
            Rephrased premise: "Nora figured there were likely many single men up there, making it perfect for what we dubbed the \"manhunt,\" despite the train fare being a significant factor. I recall us heading to Euston together one Saturday morning, full of excitement, to purchase the ticket in advance. It was kept a secret, as my parents would have quickly stopped her if they knew she was traveling alone."

            Original premise: "His mother driving the car, so happy, young-looking and fashionably dressed and his father, a big, confident man in a smart suit, smiling and turning round to say something to Simon in the back seat. Marie thought of her own mother with her frumpy clothes and ageing, lined face. No one would have guessed that she was only forty-two."
            hypothesis: "Marie's mother was only forty-two"
            label: "0"
            Rephrased premise: "Marie observed Simon's parents, with his mother looking joyful, youthful, and stylish as she drove, and his father, a large, self-assured man in a sharp suit, smiling and speaking to Simon in the back seat. This made Marie reflect on her own mother, whose outdated attire and aged, lined face belied her true age of just forty-two."

            Original premise: "`I hope you are settling down and the cat is well.'' This was a lie. She did not hope the cat was well."
            hypothesis: "the cat was well"
            label: "2"
            Rephrased premise: "\"''I trust you're getting comfortable and that the cat is doing fine.'' This statement was untrue. She did not actually wish for the cat's well-being.\""

            Original premise: "Jane ate without pausing. Hunger was an unknown experience. She had never imagined it could actually hurt."
            hypothesis: "hunger could actually hurt"
            label: "0"
            Rephrased premise: "Jane ate continuously, as hunger was a sensation she had never encountered before, and she had not anticipated that it could genuinely cause pain."

            Original premise: "Miss Martindale had had a school, but her rigid ideas and stern manner had frightened the children, and their parents had taken them away. And gradually the school declined, until she had to give it up and retire to end her days in the white cottage with the inevitable cat as her only companion. Breeze had never imagined that digging was such hard work."
            hypothesis: "digging was such hard work"
            label: "0"
            Rephrased premise: "Miss Martindale once ran a school, but her strict principles and severe demeanor scared the children away, leading their parents to withdraw them. Over time, the school dwindled, forcing her to close it and retire to a white cottage, where her only company was an inevitable cat. Breeze had never realized that digging could be so laborious."

            Original premise: "`You don't need to worry. We're quite adequately chaperoned. Rosa is a woman of strict moral principles.'' If she knows I'm in here she's probably hovering outside the door right now."
            hypothesis: "he's in here"
            label: "0"
            Rephrased premise: "\"We're well supervised, so there's no need for concern. Rosa, who upholds strong moral values, is likely standing just outside the door if she's aware that I'm inside.\""

            Original premise: "I said you were mad to come over at this time. It's a world event. Don't you know that Venice is packed with visitors?"
            hypothesis: "Venice is packed with visitors"
            label: "0"
            Rephrased premise: "Visiting at this time is madness, I told you. It's a global event, and Venice is teeming with tourists."

            Original premise: "And what she had said, and went on saying quietly, calmly, efficiently, was that she loved Maggie. She paid attention. At eight Maggie had not known that her grandmother was famous but she had seen that people had something in their manner when they looked at Rachel."
            hypothesis: "Maggie's grandmother was famous"
            label: "0"
            Rephrased premise: "Quietly and calmly, she expressed her love for Maggie, showing attentiveness. At the age of eight, Maggie was unaware of her grandmother's fame, but she noticed a certain demeanor in people when they observed Rachel."

            Original premise: "They 'd seen Miss Lavant on the promenade and about the town, always walking slowly, sometimes with a neat wicker basket. Kate had often thought she was beautiful. She hadn't known she was in love with Dr Greenslade who had a wife already and three children."
            hypothesis: "Miss Lavant was in love with Dr Greenslade"
            label: "0"
            Rephrased premise: "Miss Lavant was frequently seen by them strolling leisurely along the promenade and around town, occasionally carrying a tidy wicker basket. Kate often considered her to be beautiful. She was unaware that Miss Lavant was romantically involved with Dr. Greenslade, who was already married with three children."

            ---

            Your Task:

            Evaluate the following data and determine whether the rephrased premise maintains consistency with the original premise regarding the given \
                **hypothesis** and its **label** choice.

            **Original premise**: "{premise}"  
            **Hypothesis**: "{hypothesis}"  
            **Label**: "{label}" (0 = **entailment**, 1 = **contradiction**, 2 = **neutral**) 
            **Rephrased premise**: "{rephrased_premise}"

            Example Input:
            **Original premise**: "Chopra stood unsteadily on his feet. The shapechanger bounded around with excitement. Chopra could tell something had happened."
            **Hypothesis**: "something had happened" 
            **Label**: "0" (0 = **entailment**, 1 = **contradiction**, 2 = **neutral**) 
            **Rephrased premise**: "Chopra wobbled as he stood, while the shapechanger leapt about energetically. It was evident to Chopra that an event had occurred."

            Expected Output: same

            Directly output [same/not the same] without any explanation:

            """


eval_list = []
count = 0
with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/CB/cb_train.jsonl', 'r', encoding="utf-8") as f1: # modify
    for line in f1:
        data = json.loads(line)
        temp = {}
        temp["original"] = data["premise"]
        temp["hypothesis"] = data["hypothesis"]
        temp["idx"] = data["idx"]
        temp["label"] = data["label"]
        eval_list.append(temp)


with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/CB/cb_train.jsonl', 'r', encoding="utf-8") as f2: # modify
    for line in f2:
        data = json.loads(line)
        eval_list[count]["rephrased"] = data["premise"]
        count += 1

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/rejection_sampling/mezo/CB/cb_train.jsonl") # output file
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
