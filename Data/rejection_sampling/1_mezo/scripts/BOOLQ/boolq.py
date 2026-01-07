### evaluate using open ai api
import json
from openai import OpenAI
from tqdm import tqdm
import os

client = OpenAI(
    # 钱多多
    # # api_key = 'sk-eYvV5PulUcRh5gX40d10873c274b41C3B596F4F1F06e1a34', # office
    # api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c', # my
    # base_url = "https://api2.aigcbest.top/v1"

    # OpenRouter
    api_key = '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', 
    base_url = "https://openrouter.ai/api/v1"
)

eval_list = []

def generate_prompt(question, answer, original_passage, rephrased_passage):
    return f"""
            Task Description:

            The BoolQ Dataset structure is as follow:
            - The **BoolQ dataset** consists of yes/no questions and short passages. The task is to determine whether the **passage** supports a **"true"** or **"false"** answer to the **question**.
            - The rephrased **passage** must **not change the answerability or truth value** of the original passage with respect to the question.
            - If the original **answer** is **true**, the rephrased passage must continue to support a "yes" response.
            - If the original **answer** is **false**, the rephrased passage must not provide justification to answer "yes".

            We rephrased the passage section of a BoolQ data. You are tasked with verifying whether the Rephrased **passage** maintains consistency with its associated **question**. Specificqally, \
                determine if the Rephrased **passage** remains logically consistent with the given **question** and its **answer** as the same as the Original \
                    **passage**. Below are correctly rephrased examples to guide you:

            ### Few-shot Examples:

            question: "science begins with the premise that knowledge should first be acquired through observation"
            answer: true
            Original passage: "These terms are used with respect to reasoning (epistemology) to distinguish ``necessary conclusions from first premises'' (i.e., what must come before sense observation) from ``conclusions based on sense observation'' (which must follow it). Thus, the two kinds of knowledge, justification, or argument, may be glossed:"
            Rephrased passage: "\"These concepts are applied in epistemology to differentiate 'necessary conclusions arising from initial premises'—which precede sensory observation—from 'conclusions derived from sensory observation,' which come afterwards. In this way, the two forms of knowledge, justification, or reasoning can be outlined:\""

            question: "is tim brown in the hall of fame"
            answer: true
            Original passage: "Timothy Donell Brown (born July 22, 1966) is a former American football wide receiver who played professionally in the National Football League (NFL). He played college football for Notre Dame, where he won the Heisman Trophy, becoming the first wide receiver to win the award. He spent sixteen years with the Los Angeles/Oakland Raiders, during which he established himself as one of the NFL's most prolific wide receivers. Brown has also played for the Tampa Bay Buccaneers. In 2015, he was inducted into the Pro Football Hall of Fame."
            Rephrased passage: "Timothy Donell Brown (born July 22, 1966) is a former NFL wide receiver who played professionally for the National Football League. He was a standout college player for Notre Dame, where he made history as the first wide receiver to win the Heisman Trophy. Brown spent the majority of his professional career—sixteen years—with the Los Angeles/Oakland Raiders, establishing himself as one of the league's top wide receivers. He also had a stint with the Tampa Bay Buccaneers. In 2015, he was enshrined in the Pro Football Hall of Fame."

            question: "can you drink alcohol in public in denmark"
            answer: true
            Original passage: "Drinking in public in Denmark is legal in general. The law forbids ``disturbing of the public law and order''. Thus general consumption is accepted. Several cafes have outdoor serving in the same zones."
            Rephrased passage: "\"Public consumption of alcohol is generally permitted in Denmark. Laws primarily focus on preventing disturbances to public order, allowing regular drinking practices. Many cafes even offer outdoor seating in these areas.\""

            question: "is jersey currency legal tender in the uk"
            answer: false
            Original passage: "Both Jersey and Bank of England notes are legal tender in Jersey and circulate together, alongside the Guernsey pound and Scottish banknotes. The Jersey notes are not legal tender in the United Kingdom but are legal currency, so creditors and traders may accept them if they so choose."
            Rephrased passage: "Jersey notes, along with Bank of England notes, serve as legal tender in Jersey, circulating together with Guernsey pounds and Scottish banknotes. However, in the United Kingdom, Jersey notes are not considered legal tender; they are legal currency, meaning businesses and creditors can opt to accept them if desired."

            question: "have the milwaukee bucks ever won a championship"
            answer: true
            Original passage: "The Bucks have won one league title (1971), two conference titles (1971 and 1974), and 13 division titles (1971--1974, 1976, 1980--1986, 2001). They have featured such notable players as Kareem Abdul-Jabbar, Sidney Moncrief, Oscar Robertson, Bob Dandridge, Bob Lanier, Glenn Robinson, Ray Allen, Sam Cassell, Junior Bridgeman, Michael Redd, Terry Cummings, Vin Baker, Jon McGlocklin, Marques Johnson, and Brian Winters."
            Rephrased passage: "The Milwaukee Bucks have secured one league championship in 1971, along with two conference titles in 1971 and 1974, and 13 division titles spanning the years 1971 to 1974, 1976, 1980 to 1986, and 2001. The team has been home to prominent players such as Kareem Abdul-Jabbar, Sidney Moncrief, Oscar Robertson, Bob Dandridge, Bob Lanier, Glenn Robinson, Ray Allen, Sam Cassell, Junior Bridgeman, Michael Redd, Terry Cummings, Vin Baker, Jon McGlocklin, Marques Johnson, and Brian Winters."

            question: "does the world cup final go to penalties"
            answer: true
            Original passage: "This is a list of all penalty shoot-outs that have occurred in the Finals tournament of the FIFA World Cup. Penalty shoot-outs were introduced as tie-breakers in the 1978 World Cup but did not occur before 1982. The first time a World Cup title was won by penalty shoot-out was in 1994. The only other time was in 2006. By the end of the 2018 edition, 30 shoot-outs have taken place in the World Cup. Of these, only two reached the sudden death stage after still being tied at the end of ``best of five kicks''."
            Rephrased passage: "The Finals tournament of the FIFA World Cup has featured several penalty shoot-outs, which were implemented as tie-breaking methods starting with the 1978 World Cup, though they were first used in 1982. A World Cup was first decided through a penalty shoot-out in 1994, with another instance occurring in 2006. By the close of the 2018 tournament, 30 shoot-outs had been recorded, with only two advancing to sudden death after ending in a draw following the initial five attempts."

            question: "is kingdom manga based on a true story"
            answer: true
            Original passage: "Kingdom (\u30ad\u30f3\u30b0\u30c0\u30e0, Kingudamu) is a Japanese manga series written and illustrated by Yasuhisa Hara (\u539f\u6cf0\u4e45, Hara Yasuhisa). The manga provides a fictionalized account of the Warring States period primarily through the experiences of the war orphan Xin and his comrades as he fights to become the greatest general under the heavens, and in doing so, unifying China for the first time in history. The series was adapted into a thirty-eight episode anime series by studio Pierrot that aired from June 4, 2012 to February 25, 2013. A second season was announced and aired from June 8, 2013 to March 1, 2014. An English language release of the anime was licensed by Funimation."
            Rephrased passage: "\"Kingdom (\u30ad\u30f3\u30b0\u30c0\u30e0, Kingudamu) is a Japanese manga series authored and illustrated by Yasuhisa Hara (\u539f\u6cf0\u4e45, Hara Yasuhisa). It depicts a fictionalized version of the Warring States era, focusing on the journey of Xin, a war orphan, who aspires to become the supreme general and achieve the unification of China for the first time in history. The manga was later adapted into an anime series by Studio Pierrot, consisting of thirty-eight episodes, which aired from June 4, 2012, to February 25, 2013. A second season followed, broadcasting from June 8, 2013, to March 1, 2014, with Funimation acquiring the rights for an English language release.\""

            question: "is croatia part of the european economic area"
            answer: true
            Original passage: "As of 2014 the contracting parties to the EEA are 3 of the 4 EFTA member states and 27 of the 28 EU member states. The 28th and newest EU member, Croatia, finished negotiating their accession to the EEA in November 2013, and since 12 April 2014 is provisionally applying the agreement pending its ratification by all EEA member states."
            Rephrased passage: "As of 2014, the EEA comprises three of the four EFTA countries and 27 EU nations out of the total 28. Croatia, the latest EU member, completed negotiations for joining the EEA in November 2013 and began provisionally implementing the agreement on April 12, 2014, while awaiting ratification by all EEA member states."

            question: "can u marry a dead person in france"
            answer: true
            Original passage: "Posthumous marriage (or necrogamy) is a marriage in which one of the participating members is deceased. It is legal in France and similar forms are practiced in Sudan and China. Since World War I, France has had hundreds of requests each year, of which many have been accepted."
            Rephrased passage: "Posthumous marriage, also known as necrogamy, is a ceremony involving one deceased individual as a participant. This practice is permitted in France, where similar traditions are observed in Sudan and China. Since World War I, France has received numerous requests annually for such marriages, with a significant number being approved."

            question: "is saline and sodium chloride the same thing"
            answer: false
            Original passage: "Saline, also known as saline solution, is a mixture of sodium chloride in water and has a number of uses in medicine. Applied to the affected area it is used to clean wounds, help remove contact lenses, and help with dry eyes. By injection into a vein it is used to treat dehydration such as from gastroenteritis and diabetic ketoacidosis. It is also used to dilute other medications to be given by injection."
            Rephrased passage: "\"Saline solution, often simply called saline, is a combination of sodium chloride and water used for various medical purposes. It is applied to clean wounds, assist in removing contact lenses, and alleviate dry eyes. When administered intravenously, it treats dehydration associated with conditions like gastroenteritis and diabetic ketoacidosis. Additionally, it serves as a diluent for medications administered by injection.\""

            question: "is there a treatment for the bubonic plague"
            answer: true
            Original passage: "Several classes of antibiotics are effective in treating bubonic plague. These include aminoglycosides such as streptomycin and gentamicin, tetracyclines (especially doxycycline), and the fluoroquinolone ciprofloxacin. Mortality associated with treated cases of bubonic plague is about 1--15%, compared to a mortality of 40--60% in untreated cases."
            Rephrased passage: "\"Various types of antibiotics can successfully treat bubonic plague. Effective options include aminoglycosides like streptomycin and gentamicin, tetracyclines, particularly doxycycline, and fluoroquinolones such as ciprofloxacin. When treated, the mortality rate for bubonic plague ranges from 1% to 15%, whereas it reaches 40% to 60% in cases that go untreated.\""

            question: "does buffy's mom know she's a slayer"
            answer: true
            Original passage: "The premise of the series is that Buffy is the latest Slayer, a young woman endowed by mystical forces with superhuman powers to fight and defeat vampires, demons, and other evil forces in the fictional town of Sunnydale. Like every Slayer before her, she was chosen and informed of her destiny when she was 15 years old. Her mother is unaware of her daughter's powers and responsibilities until Buffy is forced to tell her at the end of the second season of the television series. Although Joyce is shocked at this revelation, she recovers quickly and remains a source of stability for Buffy and Buffy's small circle of friends who assist her, dubbed the Scooby Gang. Eventually Joyce is able to take Buffy's dangerous demon-fighting in stride and even become proud and respectful of her daughter's abilities. Her natural death from an illness in the fifth season forces Buffy to face becoming an adult."
            Rephrased passage: "Buffy is the latest Slayer, a young woman endowed with superhuman abilities to combat vampires, demons, and other evil entities in Sunnydale. She was selected and informed of her fate at 15. Buffy's mother initially has no knowledge of her daughter's powers until Buffy reveals them at the end of the second season. Although Joyce is initially shocked, she soon adapts and becomes a source of stability for Buffy and her friends, known as the Scooby Gang, who help in her mission. Joyce eventually accepts Buffy's demon-fighting role with pride and respect for her abilities. Joyce's death from illness in the fifth season necessitates Buffy to embrace adulthood."

            question: "was the leaning tower of pisa built leaning"
            answer: false
            Original passage: "The tower's tilt began during construction in the 12th century, caused by an inadequate foundation on ground too soft on one side to properly support the structure's weight. The tilt increased in the decades before the structure was completed in the 14th century. It gradually increased until the structure was stabilized (and the tilt partially corrected) by efforts in the late 20th and early 21st centuries."
            Rephrased passage: "During its construction in the 12th century, the leaning of the tower began, resulting from a foundation that was insufficient on one side due to the soft soil, unable to bear the building's weight. This tilt progressed over the years before the tower's completion in the 14th century. The leaning continued to worsen until stabilization and partial correction efforts were undertaken in the late 20th and early 21st centuries."

            question: "is the movie fool's gold a true story"
            answer: false
            Original passage: "Fool's Gold is a 2008 American adventure-romance film from Warner Bros. Pictures about a recently divorced couple who rekindle their romantic life while searching for a lost treasure. The film was directed by Andy Tennant and reunites the How to Lose a Guy in 10 Days stars Matthew McConaughey and Kate Hudson."
            Rephrased passage: "Fool's Gold, released in 2008 by Warner Bros. Pictures, is an adventure-romance film that follows a divorced couple as they reignite their romance during their quest for missing treasure. Directed by Andy Tennant, the film stars Matthew McConaughey and Kate Hudson, known for their previous collaboration in How to Lose a Guy in 10 Days."

            question: "has any nfl team played the superbowl at home"
            answer: false
            Original passage: "The home field curse affects the host team of the Super Bowl. So far no team has yet managed to reach the Super Bowl in their home stadium. Four teams with Super Bowls in their home venue have qualified for the divisional playoffs: the 1994 Miami Dolphins, the 1998 Miami Dolphins, the 2016 Houston Texans, and the 2017 Minnesota Vikings, the Vikings being the first to qualify for their conference's title game. From 1966--2011 (excluding the six Super Bowl games held in a stadium without a professional team), the Super Bowl host team has had 11 winning seasons, four split seasons, and 25 losing seasons. Mathematically, the probability of that many losing seasons or more occurring by chance (assuming a 50 percent chance of having a losing season (disregarding .500 seasons)) is 7.69 percent. It should be noted, however, that the Super Bowl host stadium is selected several years before the game is played, without regard to the teams that qualify."
            Rephrased passage: "\"The home field disadvantage impacts Super Bowl host teams, as none have reached the Super Bowl in their own stadium. Four teams hosting have reached the divisional playoffs: the Miami Dolphins in 1994 and 1998, the Houston Texans in 2016, and the Minnesota Vikings in 2017, with the Vikings being the first to make it to their conference championship game. Between 1966 and 2011, omitting six Super Bowls held in non-professional team stadiums, host teams experienced 11 winning seasons, 4 even seasons, and 25 losing seasons. Statistically, the likelihood of having as many or more losing seasons by random chance is 7.69 percent, assuming a 50 percent chance of a losing season (ignoring .500 seasons). It is important to note that the Super Bowl venue is chosen several years in advance, regardless of the teams that qualify.\""

            question: "does each australian state have its own constitution"
            answer: true
            Original passage: "In Australia, each state has its own constitution. Each state constitution preceded the Constitution of Australia as constitutions of the then separate British colonies, but all the states ceded powers to the Parliament of Australia as part of federation in 1901."
            Rephrased passage: "Every Australian state possesses its own constitution, which initially served as the governing framework for each British colony before the establishment of the Australian Constitution. Upon federation in 1901, these states transferred certain powers to the Australian Parliament."

            question: "can you have a right and left bundle branch block"
            answer: true
            Original passage: "A bundle branch block can be diagnosed when the duration of the QRS complex on the ECG exceeds 120 ms. A right bundle branch block typically causes prolongation of the last part of the QRS complex, and may shift the heart's electrical axis slightly to the right. The ECG will show a terminal R wave in lead V1 and a slurred S wave in lead I. Left bundle branch block widens the entire QRS, and in most cases shifts the heart's electrical axis to the left. The ECG will show a QS or rS complex in lead V1 and a monophasic R wave in lead I. Another normal finding with bundle branch block is appropriate T wave discordance. In other words, the T wave will be deflected opposite the terminal deflection of the QRS complex. Bundle branch block, especially left bundle branch block, can lead to cardiac dyssynchrony. The simultaneous occurrence of left and right bundle branch block leads to total AV block."
            Rephrased passage: "A bundle branch block diagnosis is confirmed when the QRS complex duration on an ECG surpasses 120 ms. A right bundle branch block generally extends the final part of the QRS complex and may slightly redirect the heart's electrical axis to the right, visible as a terminal R wave in lead V1 and a slurred S wave in lead I. In contrast, a left bundle branch block causes the entire QRS to widen, often shifting the heart's electrical axis to the left, with an ECG showing a QS or rS complex in lead V1 and a monophasic R wave in lead I. Typically, bundle branch blocks also involve T wave discordance, meaning the T wave is deflected in the opposite direction of the QRS terminal deflection. Both branch blocks, especially on the left, can result in cardiac dyssynchrony. When both right and left bundle branch blocks are present, they cause a total AV block."

            question: "do you always have to say check in chess"
            answer: false
            Original passage: "In informal games, it is customary to announce ``check'' when making a move that puts the opponent's king in check. In formal competitions, however, check is rarely announced."
            Rephrased passage: "In casual chess games, players often declare \"check\" when their move threatens the opponent's king. However, in official tournaments, announcing check is uncommon."

            question: "is an australian shepherd the same as an australian cattle dog"
            answer: false
            Original passage: "The Texas Heeler is a cross between the Australian Cattle Dog and the Australian Shepherd that was first registered with the Animal Research Foundation (ARF) in 1970. The ARF has registered Australian Cattle Dogs without papers as ``Australian Cattledog Queensland Heelers'' since 1965 and was the first organisation to recognise the Australian Shepherd. Although originally bred for its ability to work cattle, the Texas Heeler is increasingly used as a pet and a companion in dog sports. As with most cross breeds, the Texas Heeler's size and appearance is a variable combination of the parent breeds."
            Rephrased passage: "The Texas Heeler, a hybrid between the Australian Cattle Dog and the Australian Shepherd, was first listed by the Animal Research Foundation (ARF) in 1970. Since 1965, the ARF has acknowledged Australian Cattle Dogs without documentation as \"Australian Cattledog Queensland Heelers\" and was the pioneer in recognizing the Australian Shepherd. Originally developed for cattle herding, the Texas Heeler has become increasingly popular as a pet and for dog sports. As is typical with mixed breeds, the Texas Heeler's size and appearance can vary based on its parent breeds."

            question: "is tomato puree and tomato sauce the same thing"
            answer: false
            Original passage: "Tomato pur\u00e9e is a thick liquid made by cooking and straining tomatoes. The difference between tomato paste, tomato pur\u00e9e, and tomato sauce is consistency; tomato puree has a thicker consistency and a deeper flavour than sauce."
            Rephrased passage: "\"Tomato pur\u00e9e is created by cooking and straining tomatoes into a dense liquid form. It differs from tomato paste and tomato sauce mainly in consistency; pur\u00e9e is thicker and more flavorful compared to sauce.\""

            ---

            Your Task:

            Evaluate the following data and determine whether the rephrased **passage** maintains consistency with the original **passage** regarding the given \
                **question** and its **answer**.

            **question**: "{question}"  
            **answer**: "{'true' if answer else 'false'}"  
            **Original passage**: "{original_passage}"
            **Rephrased passage**: "{rephrased_passage}"

            Example Input:
            **question**: ""
            **answer**: ""
            **Original passage**: ""
            **Rephrased passage**: ""

            Expected Output: same

            Directly output [same/not the same] without any explanation:

            """


eval_list = []
count = 0
with open('/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/original/BOOLQ/boolq_train.jsonl', 'r', encoding="utf-8") as f1: # modify
    for line in f1:
        data = json.loads(line)
        temp = {}
        temp["original"] = data["passage"]
        temp["answer"] = data["answer"]
        temp["question"] = data["question"]
        eval_list.append(temp)


with open('/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/synthetic/mezo/BOOLQ/boolq_train.jsonl', 'r', encoding="utf-8") as f2: # modify
    for line in f2:
        data = json.loads(line)
        eval_list[count]["rephrased"] = data["passage"]
        count += 1

output_file = os.path.expanduser("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/BOOLQ/boolq_train.jsonl") # output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")

correct_answer = 0
total_answer = 0

for i in tqdm(range(len(eval_list))):
    output_data = {}
    if i >= 500:
        print(eval_list[i]["rephrased"])
        eval_list[i]['eval_result'] = response.choices[0].message.content # change
        output_data["question"] = eval_list[i]["question"]
        output_data["answer"] = eval_list[i]["answer"]
        output_data["passage"] = eval_list[i]["original"]
        out_file.write(json.dumps(output_data) + "\n")
        out_file.flush()
        continue
    
    # if 20 <= i < 40:
    #     eval_list[i]['eval_result'] = "same"
    #     output_data["question"] = eval_list[i]["question"]
    #     output_data["answer"] = eval_list[i]["answer"]
    #     output_data["passage"] = eval_list[i]["rephrased"]
    #     correct_answer += 1
    #     total_answer += 1
    #     out_file.write(json.dumps(output_data) + "\n")
    #     out_file.flush()
    #     continue

    prompt = generate_prompt(eval_list[i]["question"], eval_list[i]["answer"], eval_list[i]["original"], eval_list[i]["rephrased"])
    response = client.chat.completions.create( # change
        model="openai/gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "You are a helpful judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    if response.choices[0].message.content == 'not the same': # change
        print(eval_list[i]["rephrased"])
        eval_list[i]['eval_result'] = response.choices[0].message.content # change
        output_data["question"] = eval_list[i]["question"]
        output_data["answer"] = eval_list[i]["answer"]
        output_data["passage"] = eval_list[i]["original"]
        total_answer += 1
        out_file.write(json.dumps(output_data) + "\n")
        out_file.flush()
        continue
    
    # print(response)
    eval_list[i]['eval_result'] = response.choices[0].message.content # change
    output_data["question"] = eval_list[i]["question"]
    output_data["answer"] = eval_list[i]["answer"]
    output_data["passage"] = eval_list[i]["rephrased"]
    correct_answer += 1
    total_answer += 1
    out_file.write(json.dumps(output_data) + "\n")
    out_file.flush()

acc = correct_answer / total_answer
print("The prompt accuracy is ", acc)
