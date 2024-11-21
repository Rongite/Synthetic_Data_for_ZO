### evaluate using open ai api
import json
from openai import OpenAI
from tqdm import tqdm
import os

client = OpenAI(
    api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    # api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    base_url = "https://api2.aigcbest.top/v1"
)

eval_list = []

# Example hypothesis and reference
example = [ 
    { 
        "id": "1",
        "original": "It was a complex language. Not written down but handed down. One might say it was peeled down.", # modify
        "rephrased": "The language was intricate, transmitted orally rather than documented, akin to being uncovered layer by layer.", # modify 
        "eval": "same"
    }, 
    { 
        "id": "2",
        "original": "It is part of their religion, a religion I do not scoff at as it holds many elements which match our own even though it lacks the truth of ours. At one of their great festivals they have the ritual of driving out the devils from their bodies. First the drummers come on - I may say that no women are allowed to take part in this ritual and the ladies here will perhaps agree with me that they are fortunate in that omission.", 
        "rephrased": "Their religion, which I respect for sharing numerous aspects with ours despite lacking our truth, includes a ritual at one of their major festivals where they expel devils from their bodies. Initially, the drummers appear\u2014it's worth noting that women are not permitted to participate, and the women present might agree that they're lucky to be excluded.", 
        "eval": "same" 
    }, 
    {
        "id": "3",
        "original": "The Paris to Rouen railway was being extended to Le Havre, and the line cut straight through Dr Flaubert's land. Part of it was to be compulsorily purchased. You could say that Gustave was shepherded into creative retreat at Croisset by epilepsy.",
        "rephrased": "The railway from Paris to Rouen was extended to Le Havre, running directly through the property of Dr. Flaubert, necessitating part of it be bought out, thereby pushing Gustave into a creative retreat at Croisset due to his epilepsy.",
        "eval": "same"
    },
    {
        "id": "4",
        "original": "Part of it was to be compulsorily purchased. You could say that Gustave was shepherded into creative retreat at Croisset by epilepsy. You could also say he was driven there by the railway.",
        "rephrased": "A portion of it was taken by compulsory purchase; one might suggest that epilepsy guided Gustave into a creative retreat at Croisset, or that he was compelled there by the railway.",
        "eval": "same"
    },
    {
        "id": "5",
        "original": "Some of them, like for instance the farm in Connecticut, are quite small. If I like a place I buy it. I guess you could say it's a hobby.",
        "rephrased": "For example, some properties, such as the farm in Connecticut, are rather modest in size. If a location appeals to me, I purchase it. You might consider it a hobby.",
        "eval": "same"
    },
    {
        "id": "6",
        "original": "Look, my dear, I'm not in my dotage yet, and I know I'm a grumbler and a complainer. You could say the only form of comfort I 've got are my complaints.",
        "rephrased": "I'm not elderly yet, my dear, and I admit I'm prone to grumbling and complaining, but you might say that my grievances are my sole source of solace.",
        "eval": "same"
    },
    {
        "id": "7",
        "original": "Then the silence in the Zoo became complete. Woil stared around him and then suddenly with a push of his wings raised himself into the air, turned, and landed ten feet away on the back of a green bench. Creggan could see that he was afraid and that his fear was making him terribly uncertain.",
        "rephrased": "The Zoo fell into total silence as Woil took a glance around and abruptly, with a thrust of his wings, lifted himself airborne, pivoted, and settled ten feet away atop a green bench, where Creggan noticed his fear causing profound uncertainty.",
        "eval": "same"
    },
    {
        "id": "8",
        "original": "But, of course, that just wasn't possible. Her disappointment over Jonathan, which had driven her to France in the first place, had been relegated somewhere to the back of her mind. Now in retrospect she could see that marriage to him would have been a ghastly mistake and that her reaction to discovering that he had been seeing other women while he had been engaged to her had had far more to do with wounded pride than with a wounded heart.",
        "rephrased": "However, naturally, that was simply not feasible. Her disillusionment with Jonathan, which had initially led her to France, had been pushed to the recesses of her mind. Looking back now, she realized that marrying him would have been a dreadful error and that her response to finding out he had been involved with other women during their engagement was more about injured ego than a broken heart.",
        "eval": "same"
    },
    {
        "id": "9",
        "original": "Like now. The Community in Knockglen would defend Eve vociferously. Even some of the Sisters here in Dublin might see that the girl had a point.",
        "rephrased": "At this moment, the Knockglen community would ardently support Eve, and even a few of the Sisters in Dublin might acknowledge that the girl had a valid argument.",
        "eval": "same"
    },
    {
        "id": "10",
        "original": "``They have to be crushed, Bobkins!'' So saying, she marched off down the gravel path, making the kind of crunching noise Robert had thought could only be produced by the BBC sound-effects department. As they rounded the edge of the building he could see that behind the house was a vast garden.",
        "rephrased": "\"They must be defeated, Bobkins!\" With that declaration, she strode down the gravel path, creating a crunching sound similar to what Robert believed only the BBC sound-effects department could make. As they turned the corner of the building, he noticed a sprawling garden behind the house.",
        "eval": "same"
    },
    {
        "id": "11",
        "original": "``Do you mind if I use your phone?'' Ronni could see that Guido's brain was whirring.",
        "rephrased": "\"``Is it alright if I borrow your phone?'' Ronni noticed Guido's mind racing with thoughts.\"",
        "eval": "same"
    },
    {
        "id": "12",
        "original": "``Look, lady, give me a break. I just deliver the stuff, I don't interview it for the Sunday papers.'' He waved the paper at her and even at a distance she could see that it said very little.",
        "rephrased": "\"Listen, ma'am, cut me some slack. I'm just the delivery guy, not a journalist writing a Sunday feature.\" He gestured with the paper, and from afar, she could tell it had scant information.",
        "eval": "same"
    },
    {
        "id": "13",
        "original": "``You 've heard something like that before. Haven't you, Jimmy?'' Jimmy had been shaken by those sounds more shaken than the others for good reason but Cardiff could see that he was unprepared to show it as he pushed himself away from the reception counter.",
        "rephrased": "\"``You\u2019ve heard something similar in the past, haven\u2019t you, Jimmy?'' Although Jimmy was more unsettled by those sounds than the others, with good cause, Cardiff observed that he was determined not to reveal it as he moved away from the reception desk.\"",
        "eval": "same"
    },
    {
        "id": "14",
        "original": "It was Alan's idea. He made a sour kind of joke out of it, that they must wait until their wedding night. Carolyn agreed because she could see he meant it although she didn't understand why.",
        "rephrased": "Alan came up with the idea, humorously suggesting they should wait until their wedding night, and Carolyn agreed, sensing his seriousness despite not understanding his reasoning.",
        "eval": "same"
    },
    {
        "id": "15",
        "original": "And why bother to write anyway? What was there to say? Mary had some vague idea that Adam's parents might suspect he was down here and come to see him.",
        "rephrased": "What was the point of writing at all? Was there anything worth saying? Mary had a hazy thought that Adam's parents might guess he was here and pay him a visit.",
        "eval": "same"
    },
    {
        "id": "16",
        "original": "``And you're not having this dress,'' Nora said, bending down to look at the price tag. ``It's two and a half guineas!'' she hissed at Louise who could tell that she was genuinely appalled.",
        "rephrased": "\"You won't be getting this dress,\" Nora declared, bending down to inspect the price tag. \"It's two and a half guineas!\" she whispered to Louise, who could tell she was truly shocked.",
        "eval": "same"
    },
    {
        "id": "17",
        "original": "She said good morning to Alice and Alice said hallo. She was thin and rather tall with a very lined gentle face and hair that was white but which Alice could see had once been blonde. She could also have told this was Tina's mother before Mrs Darne went off down the passage that led to the Headmaster's Flat.",
        "rephrased": "\"She greeted Alice with a good morning, and Alice responded with a hello. The woman was slender and quite tall, with a kind, heavily lined face and hair that was white, but Alice could tell it had been blonde in her younger years. Before Mrs. Darne walked down the hallway towards the Headmaster\u2019s Flat, Alice could have recognized her as Tina's mother.\"",
        "eval": "same"
    },
    {
        "id": "18",
        "original": "She might have sat all afternoon, nibbling and stuporous, exhausted but not sleepy. But the glazier finally came down from the upper floor, cheerfully announcing that all was now right and tight and he would be on his way. Maggie could tell that he would have liked to stop for a chat that he felt sorry for her left on her own but she lacked either her grandmother's grace or her mother's energy so she did not offer him tea.",
        "rephrased": "Maggie could have spent the entire afternoon in a state of lethargy, nibbling but not sleeping, until the glazier cheerfully descended from the upper floor, announcing that everything was fixed and that he would be on his way, and though he seemed inclined to stay and converse out of pity for her solitude, she neither possessed her grandmother's poise nor her mother's vigor to invite him for tea.",
        "eval": "same"
    },
    {
        "id": "19",
        "original": "Jim waited. He waited a long time, and when the young doctor finally came out, it was almost dark. Jim could nonetheless tell by his anxious face that something was wrong.",
        "rephrased": "Jim sat in anticipation for what felt like an eternity, and by the time the young doctor finally emerged, it was nearly evening; however, Jim could discern from his worried expression that something was amiss.",
        "eval": "same"
    },
    {
        "id": "20",
        "original": "``Yes?'' ``Nathan?'' He could tell it was long-distance the line was so gravelly and hollow but he didn't recognise the voice.",
        "rephrased": "\"\u2018Yes?\u2019 \u2018Nathan?\u2019 The call had a distant, rough quality that made it clear it was long-distance, but the voice was unfamiliar to him.\"",
        "eval": "same"
    }
] 

def few_shot(i, example):
    return example[i]

def prompt(sample_example):
    return f"Given the original sentence '{sample_example['original']}' and the rephrased sentence '{sample_example['rephrased']}', judge if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."

def messages(example):
    messages = [{"role": "system", "content": "You are a helpful judge."}]
    for i in range(len(example)):
        sample_example = few_shot(i, example)
        the_prompt = prompt(sample_example)
        messages.append({"role": "user", "content": the_prompt})
        messages.append({"role": "assistant", "content": sample_example["eval"]})
    return messages

messages = messages(example)

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
    if i < 20:
        eval_list[i]['eval_result'] = example[i]["eval"]
        if example[i]["eval"] == "same":
            output_data["premise"] = eval_list[i]["rephrased"]
            output_data["hypothesis"] = eval_list[i]["hypothesis"]
            output_data["idx"] = eval_list[i]["idx"]
            output_data["label"] = eval_list[i]["label"]
            correct_answer += 1
            total_answer += 1
            out_file.write(json.dumps(output_data) + "\n")
            out_file.flush()
        else:
            print(example[i])
            output_data["premise"] = eval_list[i]["original"]
            output_data["hypothesis"] = eval_list[i]["hypothesis"]
            output_data["idx"] = eval_list[i]["idx"]
            output_data["label"] = eval_list[i]["label"]
            total_answer += 1
            out_file.write(json.dumps(output_data) + "\n")
            out_file.flush()
        continue
    
    prompt = f"Given the original sentence '{eval_list[i]['original']}' and the rephrased sentence '{eval_list[i]['rephrased']}', judge if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=messages,
        temperature=0.0 
    )
    messages.pop()
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
