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
        "Original premise": "Did he intend everyone in the castle to know he did not want the wife he had married in such a hurry? Did he intend to ignore her completely? Then Isabel saw Ellen's stunned face and realised that her maid at least did not know she had spent the night alone.",
        "hypothesis": "Isabel had spent the night alone",
        "label": "0",
        "Rephrased premise": "Was it his plan for everyone in the castle to be aware that he had no desire for the wife he had hastily married? Did he plan to completely disregard her? It was then that Isabel noticed Ellen's shocked expression and realized that her maid, at least, was unaware that she had spent the night by herself.",
        "eval": "same"
    },
    { 
        "Original premise": "Most of them young, about his age, stood and talked and drank and laughed. The two girls he had noticed earlier were standing talking to some other girls. Graham hoped they all realised that just because he was standing talking to Slater that didn't mean he was gay too.",
        "hypothesis": "Graham was gay too",
        "label": "1",
        "Rephrased premise": "Young people, mostly around his age, were gathered, engaging in conversation, drinking, and laughing. The two girls he had noticed earlier were chatting with some other girls. Graham hoped everyone understood that his conversation with Slater didn't imply he was also gay.",
        "eval": "same"
    },
    { 
        "Original premise": "He said that maybe I wasn't ready to join the Party just yet. Terry's passion for equality appealed to my purer mind, and his hatred of existing authority appealed to my resentments. But although I hated inequality it didn't mean I wanted to be treated like everyone else.",
        "hypothesis": "he wanted to be treated like everyone else",
        "label": "1",
        "Rephrased premise": "Terry suggested that perhaps I wasn't quite ready to become a Party member. While his fervor for equality resonated with my idealistic side, and his disdain for the current authority matched my grievances, my dislike for inequality didn't equate to a desire to be treated the same as everyone else.",
        "eval": "same"
    },
    { 
        "Original premise": "Oh, my poor Folly... We 've been together for five years, Lexy and I - she practically holds that company together. Of course I gave her an A''. But that doesn't mean I'm having an affair with her.",
        "hypothesis": "he is having an affair with Lexy",
        "label": "1",
        "Rephrased premise": "\"My dear Folly, Lexy and I have been working closely for five years, and she is essentially the backbone of the company. Naturally, I awarded her an 'A'. However, this does not imply that I am romantically involved with her.\"",
        "eval": "same"
    },
    { 
        "Original premise": "Nevertheless, her heart sank at the thought of spending an evening with him in his present state of mind, and she was tempted to invent a sore throat in order to get out of it. But he had been very helpful over the Puddephat business, she admitted to herself, and his moods were unpredictable - he might be on top of the world by the time he arrived at the cinema. `I'm still keen if you are '' she said brightly pretending she had not noticed anything was amiss.",
        "hypothesis": "something was amiss",
        "label": "0",
        "Rephrased premise": "Despite feeling disheartened at the prospect of spending the evening with him given his current mood, she considered feigning a sore throat to avoid it. However, she acknowledged his significant help with the Puddephat situation and recognized his unpredictable nature—he might be in high spirits by the time they reached the cinema. \"I'm still eager if you are,\" she said cheerfully, acting as though she hadn't noticed anything was wrong.",
        "eval": "same"
    },
    { 
        "Original premise": "What must it be like to be imprisoned here, day after day, month after month? I wonder, does he keep them chained and manacled, thought Fenella, or does he use sorcery? And so utterly immersed was she in this strange blue and green land that was not feeling strange any more that she did not even notice that she was weighing sorcery against steel chains and seriously considering the likely outcome.",
        "hypothesis": "Fenella was weighing sorcery against steel chains",
        "label": "0",
        "Rephrased premise": "In contemplating the experience of being confined here continuously, Fenella pondered whether he restrained them with chains and manacles or employed magic. So absorbed was she in this peculiar blue and green realm, which no longer felt unfamiliar, that she found herself earnestly comparing the effectiveness of sorcery to that of physical restraints.",
        "eval": "same"
    },
    { 
        "Original premise": "Strasbourg, Vienna, Bucharest, Istanbul, not stopping, not looking back. I saw her tossing newly gauffred curls as the open roadster headed east, away from Ollie... Temporarily I managed to re-erect my jocular facade, but inside I was panicking. He could take her away I thought he could just do that he has such power to hurt me this little furry creature who hasn't even noticed that I 've given up the weed.",
        "hypothesis": "he has given up the weed",
        "label": "0",
        "Rephrased premise": "As the roadster sped eastward through cities like Strasbourg, Vienna, Bucharest, and Istanbul, without pausing or looking back, I watched her flick her freshly styled curls. Despite temporarily restoring my cheerful exterior, internally, I was in a state of panic. I feared he could simply take her away, wielding the power to wound me deeply, this small, oblivious creature, unaware that I had quit smoking.",
        "eval": "same"
    },
    { 
        "Original premise": "It - the tractor, the boys and the bulbs in the earth - knew she had chosen for them and was coming back to them. Of course there was still love, there was healthy, growing love and its name was called Work. She had fallen in love with it so slowly and gently and sweetly that she had never noticed it had happened.",
        "hypothesis": "falling in love had happened",
        "label": "0",
        "Rephrased premise": "The tractor, the boys, and the bulbs in the soil were aware that she had made a choice for them and was returning. Naturally, love persisted, a robust and flourishing love named Work. She had gradually and tenderly fallen in love with it, so subtly that she hadn't realized it had occurred.",
        "eval": "same"
    },
    { 
        "Original premise": "After the twelfth dot, two thirds of the way down the page, the transcript of this long session tails away into blank paper. I suppose what's happened is this. He has gone on staring out of the window thinking and she has gone on staring at him waiting with such absorption that neither of them noticed the tape had run out.",
        "hypothesis": "the tape had run out",
        "label": "0",
        "Rephrased premise": "\"Following the twelfth dot, about two-thirds down the page, the transcript of this lengthy session fades into empty space. I assume the situation is this: he continued gazing out the window, lost in thought, while she kept watching him intently, so absorbed that neither realized the tape had ended.\"",
        "eval": "same"
    },
    {
        "Original premise": "`His name is Matthew Blake,'' Mandy informed Charity as they descended the steps from their cabin on to the paved pathway that led to the lodge. Thankfully she hadn't even noticed that Charity had changed from the blue wrap-around skirt and was now wearing red shorts with her white silk blouse.",
        "hypothesis": "Charity had changed from the blue wrap-around skirt",
        "label": "0",
        "Rephrased premise": "\"As they stepped down from their cabin onto the paved path leading to the lodge, Mandy mentioned to Charity, 'His name is Matthew Blake.' Fortunately, she didn't even notice that Charity had switched from the blue wrap-around skirt to red shorts paired with her white silk blouse.\"",
        "eval": "same"
    },
    {
        "Original premise": "These ` ere smugglers is a dangerous bunch from wot I 've  eard!'' If only we could devise a safe way of laying our hands on all that money,'' murmured Pugwash, whose greed was as proverbial as his cowardice. And the pirates were so busy discussing the problem and what they would do with the reward if they won it that they didn't notice that they were being observed from the window above by none other than the new Mayor and his entourage.",
        "hypothesis": "the pirates were being observed from the window above by none other than the new Mayor and his entourage",
        "label": "0",
        "Rephrased premise": "The smugglers are known to be a perilous group, according to what I've heard! If only we could find a secure method to get our hands on all that money, Pugwash muttered, his greed as well-known as his cowardice. Meanwhile, the pirates were so engrossed in their discussion about the problem and their plans for the reward that they failed to notice they were being watched from the window above by the new Mayor and his entourage.",
        "eval": "same"
    },
    {
        "Original premise": "Ockleton, Morpurgo, Cornelius, Dysart and half a dozen others too drunk to mention. But there was so much coming and going that any one of us could have slipped out, pushed Everett through the window and slipped back again without being noticed. Damn it all we didn't even notice Everett was missing until a porter tripped over him in the quad so anything's theoretically possible.",
        "hypothesis": "Everett was missing",
        "label": "0",
        "Rephrased premise": "Several individuals, including Ockleton, Morpurgo, Cornelius, Dysart, and others too inebriated to recall, were present. However, with the constant movement, any one of us could have easily slipped out, pushed Everett through the window, and returned unnoticed. In fact, we only realized Everett was absent when a porter stumbled over him in the quad, indicating that anything could have happened.",
        "eval": "same"
    },
    {
        "Original premise": "His hair was white, as my daughters reported when they went to view the body before it was given to the Odonata. Now he is known as The Man Who Changed the World, and there are statues to him everywhere. No one remembers he had a younger brother.",
        "hypothesis": "The Man Who Changed the World had a younger brother",
        "label": "0",
        "Rephrased premise": "His hair was white, as my daughters noted when they went to see the body before it was handed over to the Odonata. Now, he is celebrated as The Man Who Changed the World, with statues erected in his honor everywhere. Yet, the fact that he had a younger brother is forgotten.",
        "eval": "same"
    },
    {
        "Original premise": "Oh, I did, I did! I was lucky. I would have liked brothers and sisters but I don't remember that I was ever lonely.",
        "hypothesis": "she was ever lonely",
        "label": "1",
        "Rephrased premise": "I certainly did! I was fortunate. Although I wished for siblings, I don't recall ever feeling lonely.",
        "eval": "same"
    },
    {
        "Original premise": "Her priggishness. I admire it. I know she does wrong things she tries to organize other people's lives she can't see Mr Knightley is a man in a million.",
        "hypothesis": "Mr Knightley is a man in a million",
        "label": "0",
        "Rephrased premise": "Her self-righteousness is something I respect. Despite her faults, such as meddling in others' affairs, she fails to recognize that Mr. Knightley is truly one of a kind.",
        "eval": "same"
    },
    {
        "Original premise": "Only she herself knew the evil thoughts she had and how effortlessly they could be translated into action. `I 'll make a cup of tea.'' No she would not tell Peter that the person he loved most in the world was dead.",
        "hypothesis": "the person Peter loved most in the world was dead",
        "label": "0",
        "Rephrased premise": "She alone was aware of her malicious thoughts and how easily they could be put into action. \"I'll make a cup of tea,\" she decided, choosing not to inform Peter that the person he cherished most had passed away.",
        "eval": "same"
    },
    {
        "Original premise": "`It's not your day, is it, dear?'' No, but they 've got to be done, and Shirley's making the tea.'' Ianthe had not told her mother that she sometimes had to dust the books in the library.",
        "hypothesis": "Ianthe sometimes had to dust the books in the library",
        "label": "0",
        "Rephrased premise": "\"''It's not your day, is it, dear?'' ''No, but these tasks need to be completed, and Shirley is preparing the tea.'' Ianthe had not mentioned to her mother that she occasionally needed to dust the library books.\"",
        "eval": "same"
    },
    {
        "Original premise": "He pulled occasionally, his arms tiring. Conker slowed a little, but the branches were coming too fast, he had to lean right forward and couldn't use his hands. He remembered the gate at the end of the track had time to hope it was open because he didn't think Conker could jump it.",
        "hypothesis": "Conker could jump the gate",
        "label": "1",
        "Rephrased premise": "Occasionally, he tugged, feeling his arms grow weary. Conker's pace decreased slightly, but the branches approached too quickly, forcing him to lean forward and preventing the use of his hands. He recalled the gate at the track's end and hoped it was open, doubting Conker's ability to leap over it.",
        "eval": "same"
    },
    {
        "Original premise": "But he ended up eating it himself. I was reluctant to kiss my mother, afraid that somehow her weakness and unhappiness would infect me. Naturally I didn't think for a minute that my life and spirit could stimulate her.",
        "hypothesis": "her life and spirit could stimulate her mother",
        "label": "1",
        "Rephrased premise": "He ultimately consumed it himself. I hesitated to kiss my mother, fearing that her frailty and sadness might somehow affect me. Of course, I never considered that my vitality and energy could invigorate her.",
        "eval": "same"
    },
    {
        "Original premise": "`Ely,'' I said (that was her name and the first time I 'd ever used it), I want to be free.'' She looked stunned. I don't think she 'd considered this.",
        "hypothesis": "Ely had considered him wanting to be free",
        "label": "1",
        "Rephrased premise": "\"Ely,\" I addressed her by name for the first time, \"I desire freedom.\" Her expression was one of shock, indicating she hadn't anticipated this.",
        "eval": "same"
    },
    {
        "Original premise": "I'm sorry, I 've put you in an invidious position. If you're being run by Morton, he 'll want to hear all this. It won't do any harm but I 'd rather not give him food for thought because I consider him an idiot and I don't think he's capable of interpreting it correctly.",
        "hypothesis": "Morton is capable of interpreting this food for thought correctly",
        "label": "1",
        "Rephrased premise": "\"I apologize for placing you in a difficult situation. If Morton is in charge, he'll want to know all of this. While it won't cause any harm, I'd prefer not to provide him with material to ponder, as I believe he's an idiot and lacks the ability to interpret it accurately.\"",
        "eval": "same"
    },
    {
        "Original premise": "The big Norwegian shook his head, frowning. `Jeg fonstAr ikke.'' I don't think he found Ward's accent at all easy and anyway like many foreigners he found it easier to speak English than to understand it.",
        "hypothesis": "the big Norwegian found Ward's accent at all easy",
        "label": "1",
        "Rephrased premise": "The large Norwegian shook his head with a frown. \"Jeg forstår ikke.\" It seemed that Ward's accent was challenging for him, and like many non-native speakers, he found speaking English easier than comprehending it.",
        "eval": "same"
    },
    {
        "Original premise": "You really don't know anything about me, do you, despite all that wallowing in my mind? As it happens I don't think I'm the right person to lead humanity into the future no.",
        "hypothesis": "she is the right person to lead humanity into the future",
        "label": "1",
        "Rephrased premise": "Despite your attempts to delve into my thoughts, you truly know nothing about me. In fact, I believe I am not the suitable candidate to guide humanity forward.",
        "eval": "same"
    },
    {
        "Original premise": "It's where the bands practise. I can't remember what band Petra's in, but I seen them practise once. They were OK but I didn't think they was brilliant.",
        "hypothesis": "Petra's band was brilliant",
        "label": "1",
        "Rephrased premise": "The location is used for band rehearsals. I can't recall which band Petra belongs to, but I watched them practice once. They were decent, though I wouldn't say they were outstanding.",
        "eval": "same"
    },
    {
        "Original premise": "She swallowed hard, unsure if she had the nerve to go ahead. The memory of the pain in Tara's eyes last night decided her. Did he really expect her to believe that Tara was only the housekeeper?",
        "hypothesis": "Tara was only the housekeeper",
        "label": "1",
        "Rephrased premise": "Uncertain if she had the courage to proceed, she swallowed hard. The recollection of the hurt in Tara's eyes from the previous night made her decision. Was he truly expecting her to accept that Tara was merely the housekeeper?",
        "eval": "same"
    },
    {
        "Original premise": "If there are spirits at work at the time, they come only from yourself, not from the fume of the incense. Why should spirits aid living beings? What arrogance is it that drives people to believe they can have power over them?",
        "hypothesis": "people can have power over spirits",
        "label": "1",
        "Rephrased premise": "The presence of spirits, if any, originates from within oneself rather than from the incense's smoke. What makes people think spirits would assist the living? What conceit leads them to believe they can exert control over spirits?",
        "eval": "same"
    },
    {
        "Original premise": "Why should this topic matter? You talked about everything else as you usually do. Why should I feel Maelmuire is important?",
        "hypothesis": "Maelmuire is important",
        "label": "1",
        "Rephrased premise": "\"Why should this subject be significant? You've discussed all other matters as you typically do. What makes Maelmuire seem important to me?\"",
        "eval": "same"
    },
    {
        "Original premise": "It is all very well, in these changing times, to adapt one's work to take in duties not traditionally within one's realm. But bantering is of another dimension altogether. For one thing how would one know for sure that at any given moment a response of the bantering sort is truly what is expected?",
        "hypothesis": "at any given moment a response of the bantering sort is truly what is expected",
        "label": "2",
        "Rephrased premise": "In these evolving times, it's commendable to expand one's responsibilities beyond traditional roles. However, engaging in banter is an entirely different matter. How can one be certain that a bantering response is indeed anticipated at any particular moment?",
        "eval": "same"
    },
    {
        "Original premise": "But the horror of losing was as much to do with money as with pride. Biddy had never let them down, come without fail all through the bad weather, and now was giving Nails an intensive course on her own horse which - in terms of money - was worth another couple of hundred pounds. Yet surely she knew they had no way of paying should she demand it?",
        "hypothesis": "they had no way of paying",
        "label": "0",
        "Rephrased premise": "The fear of losing was tied to both financial concerns and pride. Biddy had consistently supported them, showing up without fail even in harsh weather, and was now providing Nails with an intensive course on her own horse, which financially amounted to an additional few hundred pounds. Surely, she must have realized they had no means to pay if she were to request it?",
        "eval": "same"
    },
    {
        "Original premise": "I ducked so fast I wasn't sure whether he 'd seen me or not, but it gave me a prickly feeling just to imagine it, so I scuttled for the door and legged it up the spiral stairway three steps at a time, just in case. As I ran, I remember thinking stupid thoughts like. How did he know I was up here looking down?",
        "hypothesis": "he was up there looking down",
        "label": "0",
        "Rephrased premise": "In a swift motion, I ducked, uncertain if he had noticed me, but the mere thought made me uneasy, prompting me to dash for the door and sprint up the spiral staircase, taking three steps at a time, just to be safe. As I hurried, I found myself pondering absurd questions like, \"How did he know I was up here observing?\"",
        "eval": "same"
    },
    {
        "Original premise": "At the heart of the universe there is cruelty. We are predators and are preyed upon, every living thing. Did you know that wasps lay their eggs in ladybirds piercing the weak spot in their armour?",
        "hypothesis": "wasps lay their eggs in ladybirds",
        "label": "0",
        "Rephrased premise": "Cruelty is central to the universe, where all living beings are both hunters and hunted. For instance, wasps deposit their eggs inside ladybirds by penetrating a vulnerable area in their exoskeleton.",
        "eval": "same"
    },
    {
        "Original premise": "I wanted to tell you. But the Bookman asked me to keep our meeting a secret. How did you know I 'd met him?",
        "hypothesis": "he had met the Bookman",
        "label": "0",
        "Rephrased premise": "\"I intended to inform you, but the Bookman requested that I keep our encounter confidential. How did you find out that I had met him?\"",
        "eval": "same"
    },
    {
        "Original premise": "She rubbed them away with an angry fist. She was a fool to let anyone get round her. How long before she learned that folk 'll always take advantage of weakness?",
        "hypothesis": "folk 'll always take advantage of weakness",
        "label": "0",
        "Rephrased premise": "Using an irritated hand, she wiped them away, berating herself for being naive enough to let others manipulate her. When would she finally understand that people invariably exploit vulnerability?",
        "eval": "same"
    },
    {
        "Original premise": "At length she decided that there was nothing to be gained by worrying her. Probably there was some quite innocent explanation, which Roger Kenyon would give her when she returned the wallet - if, indeed, it were his. And yet why had his manner changed so abruptly when he learned that the girl whose hat he had rescued was going to live at Sunset Cottage?",
        "hypothesis": "the girl whose hat Roger Kenyon had rescued was going to live at Sunset Cottage",
        "label": "0",
        "Rephrased premise": "Eventually, she concluded that worrying her was pointless. There was likely a simple, innocent reason that Roger Kenyon would explain when she returned the wallet—assuming it was his. However, she wondered why his demeanor had shifted so suddenly upon discovering that the girl whose hat he had retrieved was moving to Sunset Cottage.",
        "eval": "same"
    },
    {
        "Original premise": "Anna looked at Peter again and said to herself in a guilty whisper, 'Will he become even more difficult?' She wondered if a stranger could tell that he was difficult, just by looking at him. Would such a person watching Peter now reading the prayers of Rite B in his level pleasant voice notice that resentment lay like his blood just under his skin because the life he had chosen had not turned out as he had expected it to?",
        "hypothesis": "resentment lay just under Peter's skin",
        "label": "0",
        "Rephrased premise": "Anna glanced at Peter once more and guiltily whispered to herself, \"Will he become even more challenging?\" She pondered whether an outsider could discern his difficult nature just by observing him. Would someone watching Peter now, as he read the prayers of Rite B in his calm, pleasant voice, perceive that resentment simmered beneath his skin because the life he had chosen hadn't met his expectations?",
        "eval": "same"
    },
    {
        "Original premise": "Jean was tough and liked to drink. She would endure for a long while yet. But what would she do when she realized that with things as they were she was on a life sentence not just a temporary suspension of essential pleasure?",
        "hypothesis": "Jean was on a life sentence",
        "label": "0",
        "Rephrased premise": "Jean was resilient and enjoyed drinking. She would persist for quite some time. However, what would happen when she came to understand that her situation was a permanent deprivation of essential joy, not merely a temporary pause?",
        "eval": "same"
    },
    {
        "Original premise": "Clever. Klug means clever. Would you say that Abie was clever?",
        "hypothesis": "Abie was clever",
        "label": "2",
        "Rephrased premise": "\"Klug translates to 'clever.' Would you describe Abie as clever?\"",
        "eval": "same"
    },
    {
        "Original premise": "The Susweca. It means dragonfly in Sioux, you know. Did I ever tell you that's where Paul and I met?",
        "hypothesis": "Susweca is where she and Paul met",
        "label": "0",
        "Rephrased premise": "\"Susweca translates to 'dragonfly' in Sioux. Have I mentioned before that it's the place where Paul and I first met?\"",
        "eval": "same"
    },
    {
        "Original premise": "How do you know? she was going to ask, but his smile was answer enough. If DeVore said there was going to be a vacancy there would be a vacancy.",
        "hypothesis": "there was going to be a vacancy",
        "label": "0",
        "Rephrased premise": "She was about to question him, \"How do you know?\" but his confident smile provided all the assurance needed. If DeVore claimed a vacancy was imminent, then a vacancy was indeed forthcoming.",
        "eval": "same"
    },
    {
        "Original premise": "She didn't know if they had given themselves sufficient time to think things over before they married - that was the kind of question her sister Louise asked. Edward stayed in the Engineers for a bit, then came out and was not very successful in finding a job to suit him. That wasn't his fault and if anyone said that it was Nenna would still feel like poking a hole in them.",
        "hypothesis": "it was Edward's fault",
        "label": "1",
        "Rephrased premise": "Nenna was uncertain if they had allowed enough time for reflection before marrying, a question often posed by her sister Louise. Edward remained in the Engineers for a while, but upon leaving, he struggled to find a suitable job. Despite this, Nenna would still feel like confronting anyone who claimed it was Edward's fault.",
        "eval": "same"
    }
]


def few_shot(i, example):
    return example[i]

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

correct_answer = 0
total_answer = 0

for i in tqdm(range(len(test_set))):
    output_data = {}
    prompt = generate_prompt(test_set[i]["Original premise"], test_set[i]["hypothesis"], test_set[i]["label"], test_set[i]["Rephrased premise"]) #
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
