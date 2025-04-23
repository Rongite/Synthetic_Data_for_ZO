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

def generate_prompt(question, passage, answer):
    return f"""
            You are tasked with rephrasing the given **passage** while preserving its original meaning and ensuring consistency with its associated **question** and **boolean answer**. Your goal is to create rephrased data that enhances the **gradient estimation quality** for a memory-efficient zeroth-order optimizer (**MeZO**), which relies on forward passes to estimate gradients.

            ### Key Requirements for Passage Rephrasing:

            1. **Task-Specific Context (BoolQ Dataset)**:
              - The **BoolQ dataset** consists of yes/no questions and short passages. The task is to determine whether the **passage** supports a **"true"** or **"false"** answer to the **question**.
              - The rephrased **passage** must **not change the answerability or truth value** of the original passage with respect to the question.
              - If the original **answer** is **true**, the rephrased passage must continue to support a "yes" response.
              - If the original **answer** is **false**, the rephrased passage must not provide justification to answer "yes".

            2. **Consistency with the Original Answer**:
              - Ensure that the rephrased **passage** leads to the **same boolean answer** as the original.
              - Do not insert or omit information that would alter the logical basis of the answer.

            3. **Optimized for MeZO Training**:
              - **Enhance Gradient Sensitivity**: Design the rephrased passage to respond distinctly to minor variations in input phrasing or token order. Use varied vocabulary and structure to strengthen gradient signal.
              - **Focus on Memory Efficiency**: Avoid verbose phrasing or redundant elaboration. Keep the passage informative yet concise to reduce noise.
              - **Robustness to Data Sparsity**: Ensure the passage remains informative even if truncated or partially occluded, simulating real-world input loss.
              - **Non-Differentiable Optimization Readiness**: Use language that leads to clear decision boundaries (yes/no), enabling optimization of discrete metrics such as accuracy.

            4. **Maintain Informational Neutrality**:
              - Do not insert direct yes/no claims. Let the **question**-to-**answer** reasoning depend entirely on the passage content.
              - Avoid rhetorical or leading language that may bias the answer beyond what the original passage supports.

            5. **High-Quality Data Generation**:
              - Produce a **natural, fluent, and coherent** rephrased passage.
              - Avoid direct copying of the original structure; instead, use paraphrasing, clause reordering, or syntax transformation.

            ---

            ### Few-shot Examples:
            question: "do iran and afghanistan speak the same language"
            answer: true
            Original passage: "Persian (\/\u02c8p\u025c\u02d0r\u0292\u0259n, -\u0283\u0259n\/), also known by its endonym Farsi (\u0641\u0627\u0631\u0633\u06cc f\u0101rsi (f\u0252\u02d0\u027e\u02c8si\u02d0) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet."
            Rephrased passage: "Persian, also referred to as Farsi, is a Western Iranian language within the Indo-Iranian branch of the Indo-European language family. This language is predominantly used in Iran and Afghanistan\u2014where it is known as Dari since 1958\u2014and in Tajikistan, recognized as Tajiki since the Soviet era. Additionally, it is spoken in various regions that were historically Persianate and part of Greater Iran. The Persian script, derived and modified from the Arabic script and originally the Aramaic alphabet, is used for writing this language."

            question: "do good samaritan laws protect those who help at an accident"
            answer: true
            Original passage: "Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable."
            Rephrased passage: "Good Samaritan laws provide legal protection to individuals who render aid to someone who is, or whom they assume to be, injured, sick, endangered, or otherwise incapacitated. These laws aim to alleviate the reluctance of bystanders to help due to the fear of lawsuits or prosecution for any unintended harm or wrongful death resulting from their assistance. In some common-law regions of Canada, the good Samaritan doctrine ensures that a person who voluntarily assists a victim in distress cannot be successfully sued for any missteps. This principle serves to lessen the apprehension of aiding individuals due to possible legal consequences from errors in help provided. Conversely, a duty to rescue law demands that people provide assistance, holding those who neglect to do so accountable."

            question: "is windows movie maker part of windows essentials"
            answer: true
            Original passage: "Windows Movie Maker (formerly known as Windows Live Movie Maker in Windows 7) is a discontinued video editing software by Microsoft. It is a part of Windows Essentials software suite and offers the ability to create and edit videos as well as to publish them on OneDrive, Facebook, Vimeo, YouTube, and Flickr."
            Rephrased passage: "Windows Movie Maker, originally called Windows Live Movie Maker in Windows 7, is a video editing program no longer supported by Microsoft. It belongs to the Windows Essentials suite, providing tools to create, edit, and share videos on platforms such as OneDrive, Facebook, Vimeo, YouTube, and Flickr."

            question: "is confectionary sugar the same as powdered sugar"
            answer: true
            Original passage: "Powdered sugar, also called confectioners' sugar, icing sugar, and icing cake, is a finely ground sugar produced by milling granulated sugar into a powdered state. It usually contains a small amount of anti-caking agent to prevent clumping and improve flow. Although most often produced in a factory, powdered sugar can also be made by processing ordinary granulated sugar in a coffee grinder, or by crushing it by hand in a mortar and pestle."
            Rephrased passage: "Confectioners' sugar, also known as icing sugar or powdered sugar, is created by grinding granulated sugar into a fine powder. This sugar typically includes a minor amount of anti-caking agent to enhance flow and avoid clumping. While it is mostly manufactured in factories, it can also be created at home by grinding regular granulated sugar with a coffee grinder or by hand using a mortar and pestle."

            question: "is elder scrolls online the same as skyrim"
            answer: false
            Original passage: "As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective."
            Rephrased passage: "Set in the same world as other titles in The Elder Scrolls series, The Elder Scrolls Online unfolds on the continent of Tamriel. The timeline of this game places it approximately 1,000 years before The Elder Scrolls V: Skyrim, and about 800 years prior to The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. While it shares some structural similarities with Skyrim, featuring two main conflicts\u2014one concerning the world's fate and another involving the pursuit of power over Tamriel\u2014it differs in key aspects. In this game, players confront the Daedric Prince Molag Bal, who aims to merge the world of Mundus with his own realm, Coldharbour. Additionally, the struggle for the empty imperial throne sees three mortal alliances vying for control. A central aspect of the game involves the player character, who has been sacrificed to Molag Bal and whose soul has been taken, making its recovery a vital mission."

            question: "can you use oyster card at epsom station"
            answer: false
            Original passage: "Epsom railway station serves the town of Epsom in Surrey. It is located off Waterloo Road and is less than two minutes' walk from the High Street. It is not in the London Oyster card zone unlike Epsom Downs or Tattenham Corner stations. The station building was replaced in 2012\/2013 with a new building with apartments above the station (see end of article)."
            Rephrased passage: "Epsom railway station, situated in the town of Epsom in Surrey, is positioned just off Waterloo Road and is a short walk from the High Street. Unlike Epsom Downs or Tattenham Corner stations, Epsom station is outside the area covered by the London Oyster card zone. In 2012/2013, the station was updated with a new structure that includes apartments above it."

            question: "will there be a season 4 of da vinci's demons"
            answer: false
            Original passage: "The series premiered in the United States on Starz on 12 April 2013, and its second season premiered on 22 March 2014. The series was renewed for a third season, which premiered on 24 October 2015. On 23 July 2015, Starz announced that the third season would be the show's last. However Goyer has left it open for a miniseries return."
            Rephrased passage: "Starz aired the first season of the series in the U.S. starting on April 12, 2013, followed by the second season on March 22, 2014. The show was given a third season, which premiered on October 24, 2015. On July 23, 2015, it was announced by Starz that this third season would conclude the series. Nevertheless, Goyer has mentioned the possibility of a future miniseries."

            question: "is the federal court the same as the supreme court"
            answer: false
            Original passage: "The federal courts are composed of three levels of courts. The Supreme Court of the United States is the court of last resort. It is generally an appellate court that operates under discretionary review, which means that the Court can choose which cases to hear, by granting writs of certiorari. There is therefore generally no basic right of appeal that extends automatically all the way to the Supreme Court. In a few situations (like lawsuits between state governments or some cases between the federal government and a state) it sits as a court of original jurisdiction."
            Rephrased passage: "Federal courts consist of three tiers, with the United States Supreme Court serving as the ultimate judicial authority. Predominantly an appellate body, it exercises discretionary review, selecting cases to hear through the granting of writs of certiorari. Consequently, there is typically no inherent right for a case to be appealed directly to the Supreme Court. However, it occasionally acts as a court of original jurisdiction in specific circumstances, such as lawsuits involving state governments or certain disputes between the federal government and a state."

            question: "did abraham lincoln write the letter in saving private ryan"
            answer: true
            Original passage: "In the 1998 war film Saving Private Ryan, General George Marshall (played by Harve Presnell) reads the Bixby letter to his officers before giving the order to find and send home Private James Francis Ryan after Ryan's three brothers died in battle."
            Rephrased passage: "In the 1998 film Saving Private Ryan, General George Marshall, portrayed by Harve Presnell, recites the Bixby letter to his officers prior to issuing the command to retrieve Private James Francis Ryan, following the combat deaths of Ryan's three siblings."

            question: "is batman and robin a sequel to batman forever"
            answer: true
            Original passage: "With the box office success of Batman Forever in June 1995, Warner Bros. immediately commissioned a sequel. They hired director Joel Schumacher and writer Akiva Goldsman to reprise their duties the following August, and decided it was best to fast track production for a June 1997 target release date, which is a break from the usual 3-year gap between films. Schumacher wanted to homage both the broad camp style of the 1960s television series and the work of Dick Sprang. The storyline of Batman & Robin was conceived by Schumacher and Goldsman during pre-production on A Time to Kill. Portions of Mr. Freeze's back-story were based on the Batman: The Animated Series episode ``Heart of Ice'', written by Paul Dini."
            Rephrased passage: "Warner Bros., seeing the commercial success of Batman Forever in June 1995, promptly initiated plans for a sequel. They brought back Joel Schumacher as director and Akiva Goldsman as writer in August of the same year, aiming to expedite production for a release in June 1997, breaking the typical three-year interval between installments. Schumacher sought to pay tribute to the campy style of the 1960s TV series and the art of Dick Sprang. The plot for Batman & Robin was developed by Schumacher and Goldsman while they were working on A Time to Kill. Elements of Mr. Freeze\u2019s background story were inspired by the Batman: The Animated Series episode \"Heart of Ice,\" penned by Paul Dini."

            question: "is a wolverine the same as a badger"
            answer: false
            Original passage: "Badgers are short-legged omnivores in the family Mustelidae, which also includes the otters, polecats, weasels, and wolverines. They belong to the caniform suborder of carnivoran mammals. The 11 species of badgers are grouped in three subfamilies: Melinae (Eurasian badgers), Mellivorinae (the honey badger or ratel), and Taxideinae (the American badger). The Asiatic stink badgers of the genus Mydaus were formerly included within Melinae (and thus Mustelidae), but recent genetic evidence indicates these are actually members of the skunk family, placing them in the taxonomic family Mephitidae."
            Rephrased passage: "Badgers, which are short-legged omnivores, fall under the Mustelidae family\u2014a group that also contains otters, polecats, weasels, and wolverines. They are categorized within the caniform suborder of carnivorous mammals. There are 11 different badger species divided into three subfamilies: Melinae for Eurasian badgers, Mellivorinae for the honey badger or ratel, and Taxideinae for the American badger. While Asiatic stink badgers, previously part of Melinae and thus Mustelidae, are now considered part of the skunk family, Mephitidae, based on new genetic findings."

            question: "will there be a green lantern 2 movie"
            answer: false
            Original passage: "Green Lantern was released on June 17, 2011, and received generally negative reviews; most criticized the film for its screenplay, inconsistent tone, choice and portrayal of villains, and its use of CGI, while some praised Reynolds' performance. Reynolds would later voice his dissatisfaction with the film. The film underperformed at the box office, grossing $219 million against a production budget of $200 million. Due to the film's negative reception and disappointing box office performance, Warner Bros. canceled any plans for a sequel, instead opting to reboot the character in the DC Extended Universe line with the film Green Lantern Corps, set for release in 2020."
            Rephrased passage: "Green Lantern, released on June 17, 2011, faced largely unfavorable reviews, with many critiques targeting its script, uneven tone, villain choices and portrayals, and excessive CGI. Reynolds' performance was met with some approval, although he later expressed dissatisfaction with the film. Financially, the movie did not meet expectations, earning $219 million against its $200 million budget. Due to this poor reception and unsatisfactory box office results, Warner Bros. decided against a sequel, choosing instead to reboot the Green Lantern character within the DC Extended Universe with the Green Lantern Corps film, planned for a 2020 release."

            question: "does the icc has jurisdiction in the united states"
            answer: false
            Original passage: "The United States should have the chance to observe and assess the functioning of the court, over time, before choosing to become subject to its jurisdiction. Given these concerns, I will not, and do not recommend that my successor, submit the treaty to the Senate for advice and consent until our fundamental concerns are satisfied."
            Rephrased passage: "The United States ought to be allowed to watch and evaluate how the court operates over a period before deciding to accept its jurisdiction. Due to these issues, I will not, nor do I suggest that my successor, present the treaty to the Senate for their advice and approval until our core concerns are addressed."

            question: "calcium carbide cac2 is the raw material for the production of acetylene"
            answer: true
            Original passage: "Calcium carbide is a chemical compound with the chemical formula of CaC. Its main use industrially is in the production of acetylene and calcium cyanamide."
            Rephrased passage: "Calcium carbide, represented by the formula CaC2, is primarily utilized in industries for producing acetylene and calcium cyanamide."

            question: "is there a now you see me 3 coming out"
            answer: true
            Original passage: "Now You See Me is a series of heist thriller film written by Ed Solomon, Boaz Yakin, and Edward Ricourt. They focus on the actions of a team of illusionists named ``The Four Horsemen'' who pull off near impossible heists. The series features an ensemble cast including Jesse Eisenberg, Mark Ruffalo, Woody Harrelson, Isla Fisher, Dave Franco, Michael Caine, Lizzy Caplan, and Morgan Freeman. The first film was released in 2013, while the second was released in 2016, and a third film is currently in development and set to be released in 2019. The series has received mixed reviews from critics and audiences and grossed nearly $700 million worldwide."
            Rephrased passage: "\"Now You See Me is a thriller film series penned by Ed Solomon, Boaz Yakin, and Edward Ricourt, centering on a group of illusionists called 'The Four Horsemen' known for orchestrating nearly impossible heists. The cast includes actors such as Jesse Eisenberg, Mark Ruffalo, Woody Harrelson, Isla Fisher, Dave Franco, Michael Caine, Lizzy Caplan, and Morgan Freeman. The debut film premiered in 2013, followed by a sequel in 2016. Currently, the third installment is under development and anticipated for release in 2019. Despite mixed reviews, the series has garnered about $700 million globally.\""

            question: "does a penalty shoot out goal count towards the golden boot"
            answer: false
            Original passage: "A shoot-out is usually considered for statistical purposes to be separate from the match which preceded it. In the case of a two-legged fixture, the two matches are still considered either as two draws or as one win and one loss; in the case of a single match, it is still considered as a draw. This contrasts with a fixture won in extra time, where the score at the end of normal time is superseded. Converted shoot-out penalties are not considered as goals scored by a player for the purposes of their individual records, or for ``golden boot'' competitions."
            Rephrased passage: "A penalty shoot-out is typically treated as separate from the preceding match for statistical purposes. In two-legged fixtures, matches are counted as two draws or a win and a loss. For single matches, it's treated as a draw. This differs from extra time victories, which replace the score at the end of regular time. Goals from shoot-out penalties are not regarded as player goals for personal records or \"golden boot\" contests."

            question: "will there be a new season of cutthroat kitchen"
            answer: false
            Original passage: "Cutthroat Kitchen is a cooking show hosted by Alton Brown that aired on the Food Network from August 11, 2013 to July 19, 2017. It features four chefs competing in a three-round elimination cooking competition. The contestants face auctions in which they can purchase opportunities to sabotage one another. Each chef is given $25,000 at the start of the show; the person left standing keeps whatever money they have not spent in the auctions. The show ended on its fifteenth season in July 2017. The series shares some basic elements with other four-chef, three-round elimination-style competitions on Food Network including Chopped and Guy's Grocery Games. Numerous Cutthroat Kitchen contestants have competed on these shows."
            Rephrased passage: "Alton Brown hosts Cutthroat Kitchen, a cooking competition series that initially premiered on the Food Network on August 11, 2013. The program concluded its run on July 19, 2017 with its fifteenth season. In each episode, four chefs face off in three elimination rounds, where they can participate in auctions to acquire tools or methods to hinder their competitors. Starting with $25,000, any funds not spent during the auctions are retained by the last chef standing. The show has similarities with other Food Network favorites like Chopped and Guy's Grocery Games, often featuring contestants who appear across these platforms."

            question: "is jack sikma in the hall of fame"
            answer: true
            Original passage: "In 2006, Sikma was voted as one of the 100 Legends of the IHSA Boys Basketball Tournament, a group of former players and coaches in honor of the 100 anniversary of the IHSA boys basketball tournament. On June 27, 2017, Sikma was inducted into the Small College Basketball Hall of Fame as part of their second class . Inducted alongside Sikma were Zelmo Beaty, Walt Frazier, Bob Love, Elmore Smith, Jim Spivey, Rico Swanson, George Tinsley, and Al Tucker."
            Rephrased passage: "In 2006, Jack Sikma was named among the 100 Legends of the IHSA Boys Basketball Tournament, a selection of former players and coaches commemorating the tournament's 100th anniversary. On June 27, 2017, Sikma was inducted into the Small College Basketball Hall of Fame as part of their second induction class. Alongside Sikma, the class included notable figures such as Zelmo Beaty, Walt Frazier, Bob Love, Elmore Smith, Jim Spivey, Rico Swanson, George Tinsley, and Al Tucker."

            question: "can elves and humans mate lord of the rings"
            answer: true
            Original passage: "In J.R.R. Tolkien's fictional universe of Middle-earth, the Half-elven (Sindarin singular Peredhel, plural Peredhil, Quenya singular Perelda) are the children of the union of Elves and Men. Of these, the most significant were the products of couplings between the Eldar (the Elves who followed the Call to Valinor) and the Edain (the Men of the Three Houses of early Men who allied themselves with the Eldar in their war against Morgoth)."
            Rephrased passage: "In the world of Middle-earth created by J.R.R. Tolkien, the offspring known as the Half-elven (Sindarin: singular Peredhel, plural Peredhil; Quenya: singular Perelda) result from unions between Elves and Humans. Notably, these unions often involve the Eldar, Elves who responded to the summons to Valinor, and the Edain, the early Men from the Three Houses who joined forces with the Eldar against Morgoth."

            question: "the boy in the plastic bubble based on true story"
            answer: true
            Original passage: "The Boy in the Plastic Bubble is a 1976 American made-for-television drama film inspired by the lives of David Vetter and Ted DeVita, who lacked effective immune systems. It stars John Travolta, Glynnis O'Connor, Diana Hyland, Robert Reed, Ralph Bellamy & P.J. Soles. It was written by Douglas Day Stewart, executive produced by Aaron Spelling and Leonard Goldberg (who, at the time, produced Starsky and Hutch and Charlie's Angels), and directed by Randal Kleiser, who would work with Travolta again in Grease shortly after. The original music score was composed by Mark Snow. The theme song ``What Would They Say'' was written and sung by Paul Williams. William Howard Taft High School in Woodland Hills was used for filming."
            Rephrased passage: "\"The Boy in the Plastic Bubble is a 1976 American TV drama that draws inspiration from the real stories of David Vetter and Ted DeVita, who had severe immune deficiencies. John Travolta leads the cast, which also includes Glynnis O'Connor, Diana Hyland, Robert Reed, Ralph Bellamy, and P.J. Soles. Douglas Day Stewart wrote the film, with Aaron Spelling and Leonard Goldberg as executive producers who were also behind Starsky and Hutch and Charlie's Angels during that era. Randal Kleiser directed the film and later collaborated with Travolta on Grease. Mark Snow provided the film's original music score while Paul Williams penned and performed the theme song \"What Would They Say.\" Filming occurred at William Howard Taft High School in Woodland Hills.\""

                        ---

            ### Your Task:
            Rephrase the following **passage** while ensuring that it leads to the same answer for the given **question**. The rephrased passage should maintain logical consistency and enhance MeZO’s gradient estimation effectiveness.

            **question**: "{question}"  
            **answer**: "{'true' if answer else 'false'}"  
            **Original passage**: "{passage}"  

            **Directly output only one rephrased passage** without any other characters or explanatory statements like "The rephrased sentence is:":
            """


data = []
with open('/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ/boolq_train.jsonl', 'r', encoding='utf-8') as file: # input file
    for line in file:
            # Parses the JSON data for each row and appends it to the data list
        data.append(json.loads(line.strip()))

output_file = os.path.expanduser("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/BOOLQ/boolq_train_rest.jsonl") # output file

os.makedirs(os.path.dirname(output_file), exist_ok=True)
out_file = open(output_file, "w")
progress = 0 # delete
for i in tqdm(range(len(data))):
    progress += 1 # delete
    if progress <= 20: # delete
        continue # delete
    if progress > 500: # delete
        break # delete

    prompt = generate_prompt(data[i]["question"], data[i]["passage"], data[i]["answer"])
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
    result["passage"] = sentence
    out_file.write(json.dumps(result) + "\n")
    out_file.flush()
    i += 1
