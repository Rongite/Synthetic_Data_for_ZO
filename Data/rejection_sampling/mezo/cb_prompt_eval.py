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

test_set = [
   {
        "id": "21",
        "original": "Chopra stood unsteadily on his feet. The shapechanger bounded around with excitement. Chopra could tell something had happened.",
        "rephrased": "Unsteadily, Chopra stood on his feet as the shapechanger leaped around with enthusiasm, making it clear to Chopra that something had occurred.",
        "eval": "same"
   },
   {
        "id": "22",
        "original": "It seemed impossible that anyone could endure such pain for so long, but at last the doors of the Renault slammed and there was comparative silence. The engine was started up, revving violently as the car was turned round on the narrow road. John could tell that it was being driven back up the hill towards Putna.",
        "rephrased": "Enduring such agony for that length of time seemed inconceivable, but eventually, the Renault's doors banged shut, bringing relative tranquility. The engine roared to life, revving fiercely as the car was maneuvered around on the tight road. John sensed it was heading back up the incline towards Putna.",
        "eval": "same"
   },
   {
        "id": "23",
        "original": "Just when you think you 've got it straight, along comes the Fool with his pig's bladder and whops you on the nose. By the way, I'm no idiot. I could tell Gillian and Stuart weren't thrilled to see me at the airport.",
        "rephrased": "Whenever you believe everything is settled, the Fool shows up with his pig's bladder and taps you on the nose. For the record, I'm not clueless; I could sense that Gillian and Stuart were less than excited to see me at the airport.",
        "eval": "same"
   },
   {
        "id": "24",
        "original": "He's weird enough to have undressed me without thinking, according to some mad notion of the ``proper'' thing to do. Perhaps he thought I couldn't lie in bed with my clothes on.",
        "rephrased": "He\u2019s peculiar enough to have disrobed me on a whim, based on some bizarre idea of what is \u201cappropriate.\u201d Maybe he believed I couldn\u2019t lie in bed wearing my clothes.",
        "eval": "same"
   },
   {
        "id": "25",
        "original": "That is to say, I did not take sufficient account of the fact that at that time of the day, what Mr Farraday enjoys is a conversation of a lighthearted, humorous sort. Knowing this to be his likely mood when I brought in the tea yesterday afternoon, and being aware of his general propensity to talk with me in a bantering tone at such moments, it would certainly have been wiser not to have mentioned Miss Kenton at all. But you will perhaps understand that there was a natural tendency on my part in asking what was after all a generous favour from my employer to hint that there was a good professional motive behind my request.",
        "rephrased": "I failed to sufficiently consider that Mr. Farraday typically prefers lighthearted and humorous conversations at that time of day. Knowing he would likely be in that mood when I served the tea yesterday afternoon, and understanding his habit of speaking to me in a joking manner during such occasions, it would have undoubtedly been more prudent not to mention Miss Kenton. However, you might appreciate that, given I was requesting a generous favor from my employer, there was a natural inclination to suggest there was a commendable professional reason behind my appeal.",
        "eval": "same"
   },
   {
        "id": "26",
        "original": "But don't dilly-dally for too long. Once it's published we are all going to look a little risible if we have made no adjustments to what is after all known as being predominantly my own design of gallery. Also I am a bit older than the rest of you but you can perhaps understand that I don't want to drop dead without a proper and public recantation.",
        "rephrased": "Please try not to delay excessively. We'll appear somewhat foolish once it's published if we haven't made any updates to what is largely recognized as my personal gallery design. Additionally, although I'm older than most of you, you might appreciate that I'd prefer not to pass away without a formal and public retraction.",
        "eval": "same"
   },
   {
        "id": "27",
        "original": "I should dearly have liked to know whether they were Europeans or Americans, but I couldn't hear the accents. They appeared to be arguing. I hoped the white men weren't telling him to eliminate all witnesses because I don't believe it would have needed much persuasion.",
        "rephrased": "I would have really liked to discern if they were Europeans or Americans, but I couldn't catch their accents. They seemed to be arguing, and I sincerely hoped the white men weren't suggesting he eliminate all witnesses, as I don't think much convincing would have been required.",
        "eval": "same"
   },
   {
        "id": "28",
        "original": "But the damage was done as far as my faith was concerned, which is probably why I went mad. So anyway, that Christmas Eve night confirmed my worst fears, it was like a kind of `royal flush'' for the infant Jimbo. All three kings - Pa Santa and the King of Kings - all down the pan together... And to be honest I don't believe any of them stands a chance of ever making a comeback with me.",
        "rephrased": "As far as my faith was involved, the damage had been done, which likely explains my descent into madness. That Christmas Eve night only reinforced my deepest fears, acting as a sort of \"royal flush\" for young Jimbo. The trio of kings - Father Christmas and the King of Kings - were dismissed together... To be frank, I doubt any of them will ever regain my belief.",
        "eval": "same"
   },
   {
        "id": "29",
        "original": "He had seen something I should have, which was a car turning in from Soho Square and coming up behind me. My right foot hovered over the accelerator pedal and I balanced Armstrong on the clutch. I wasn't as convinced as Malpass that Nevil was out of harm's way.",
        "rephrased": "He noticed something I should have, a car turning in from Soho Square and approaching me from behind, while my right foot hovered over the accelerator and I kept Armstrong steady on the clutch; I wasn't as certain as Malpass that Nevil was safe.",
        "eval": "same"
   },
   {
        "id": "30",
        "original": "Jed wondered. He 'd scarcely set eyes on him since the night they 'd had dinner together at the house in Westwood. Nobody had mentioned him either and Jed didn't feel he should ask.",
        "rephrased": "Jed pondered; he had barely seen him since the evening they dined at the Westwood house, and no one had spoken of him either, leaving Jed reluctant to inquire.",
        "eval": "same"
   },
   {
        "id": "31",
        "original": "Yet what good came from knowing that a woman had killed herself? The children who had suffered a trauma would survive the experience, scarred by it and a little flawed by it. They would never forget that for a week they had imagined the act of murder had been committed.",
        "rephrased": "What benefit was there in realizing that a woman had taken her own life? The children, having endured a traumatic event, would continue on, marked and slightly damaged by the ordeal, never erasing the memory of a week when they believed a murder had occurred.",
        "eval": "same"
   },
   {
        "id": "32",
        "original": "Nora calculated that there must be lots of single men up there so she decided it was ideal for the ``manhunt'', as we called it, even though the train fare was a serious consideration. I remember we went up to Euston together one Saturday morning, very excited, to buy the ticket in advance. It was a secret - if my parents had known she was going away alone they would have soon put the kybosh on it.",
        "rephrased": "Nora reckoned there were plenty of unattached men in that area, making it perfect for what we jokingly called the \"manhunt,\" despite the train fare being a significant factor. I recall accompanying her to Euston on a Saturday morning, eagerly purchasing her ticket ahead of time. We kept it a secret, knowing my parents would have likely stopped her trip if they knew she was going solo.",
        "eval": "same"
   },
   {
        "id": "33",
        "original": "His mother driving the car, so happy, young-looking and fashionably dressed and his father, a big, confident man in a smart suit, smiling and turning round to say something to Simon in the back seat. Marie thought of her own mother with her frumpy clothes and ageing, lined face. No one would have guessed that she was only forty-two.",
        "rephrased": "Simon's mother, who was behind the wheel, appeared joyful, youthful, and stylishly attired, while his father, a robust and self-assured man in an elegant suit, turned back with a smile to speak to Simon in the rear seat. Meanwhile, Marie reflected on her own mother, whose dowdy attire and aged, wrinkled visage belied her mere forty-two years.",
        "eval": "same"
   },
   {
        "id": "34",
        "original": "``I hope you are settling down and the cat is well.'' This was a lie. She did not hope the cat was well.",
        "rephrased": "She falsely professed, \"I trust you\u2019re getting comfortable and that the cat is doing fine,\" while, in truth, she harbored no such wish for the cat\u2019s well-being.",
        "eval": "same"
   },
   {
        "id": "35",
        "original": "Jane ate without pausing. Hunger was an unknown experience. She had never imagined it could actually hurt.",
        "rephrased": "Jane devoured her meal nonstop, as hunger was foreign to her; she had never conceived that it could cause real pain.",
        "eval": "same"
   },
   {
        "id": "36",
        "original": "Miss Martindale had had a school, but her rigid ideas and stern manner had frightened the children, and their parents had taken them away. And gradually the school declined, until she had to give it up and retire to end her days in the white cottage with the inevitable cat as her only companion. Breeze had never imagined that digging was such hard work.",
        "rephrased": "Miss Martindale once ran a school, but her strict approach and harsh demeanor scared off both the children and their parents, leading to its eventual decline and prompting her to close it down and retire to spend her remaining years in a white cottage with just her inevitable cat for company. Breeze had never expected that digging would be so exhausting.",
        "eval": "same"
   },
   {
        "id": "37",
        "original": "``You don't need to worry. We're quite adequately chaperoned. Rosa is a woman of strict moral principles.'' If she knows I'm in here she's probably hovering outside the door right now.",
        "rephrased": "\"There's no need to be concerned. We are under proper supervision. Rosa is a woman with strong moral standards. If she knows I'm here, she's likely waiting just outside the door.\"",
        "eval": "same"
   },
   {
        "id": "38",
        "original": "I said you were mad to come over at this time. It's a world event. Don't you know that Venice is packed with visitors?",
        "rephrased": "I mentioned that it was crazy for you to visit at this hour. It's an international event. Aren't you aware that Venice is crowded with tourists?",
        "eval": "same"
   },
   {
        "id": "39",
        "original": "And what she had said, and went on saying quietly, calmly, efficiently, was that she loved Maggie. She paid attention. At eight Maggie had not known that her grandmother was famous but she had seen that people had something in their manner when they looked at Rachel.",
        "rephrased": "She consistently expressed her love for Maggie in a quiet, calm, and effective manner, attentively. At the age of eight, Maggie wasn't aware of her grandmother's fame, yet she noticed a distinct behavior in people when they regarded Rachel.",
        "eval": "same"
   },
   {
        "id": "40",
        "original": "They 'd seen Miss Lavant on the promenade and about the town, always walking slowly, sometimes with a neat wicker basket. Kate had often thought she was beautiful. She hadn't known she was in love with Dr Greenslade who had a wife already and three children.",
        "rephrased": "Miss Lavant, often seen strolling leisurely along the promenade or around town with a tidy wicker basket, was considered beautiful by Kate. Kate was unaware of her romantic feelings for Dr. Greenslade, who was already married and father to three children.",
        "eval": "same"
   },
   
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
correct_answer = 0
total_answer = 0

for i in tqdm(range(len(test_set))):
    output_data = {}
    prompt = f"Given the original sentence '{test_set[i]['original']}' and the rephrased sentence '{test_set[i]['rephrased']}', judge if the rephrased \
  sentence has the same meaning as the original sentence Directly output [same/not the same] without any explanation."
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=messages,
        temperature=0.0 
    )
    messages.pop()
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
