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
    "Original premise": "Australian Guantanamo Bay detainee David Hicks has won British citizenship, opening the door for a possible bid to have him freed from the US detention facility in Cuba. Justice Lawrence Collins of the British High Court has overturned the British Government's refusal of citizenship to Hicks, whose mother was born in England. Justice Collins said the Government had \"no power to withhold or deprive citizenship\". Justice Collins said: \"In my view it would be improper to fail to give assistance which otherwise would have been given, simply because the claimant was believed to be involved in terrorism and has not had any previous connection with this country.\" Hicks' lawyer, Stephen Grosz, said the decision was a breakthrough. He said there was now no reason why Hicks should not enjoy the same protection as the nine other British citizens released without charge from Guantanamo Bay on representations of the British Government.",
    "hypothesis": "Stephen Grosz is the British lawyer of David Hicks.",
    "label": "1",
    "Rephrased premise": "David Hicks, an Australian detainee at Guantanamo Bay, has been granted British citizenship, potentially allowing him to seek release from the U.S. detention center in Cuba. Justice Lawrence Collins of the British High Court has reversed the British Government's denial of citizenship to Hicks, whose mother was English-born. Justice Collins stated that the Government lacked the authority to deny or revoke citizenship. He remarked, \"In my opinion, it would be inappropriate to withhold assistance that would otherwise be provided, merely because the claimant is suspected of terrorism and has no prior ties to this country.\" Hicks' attorney, Stephen Grosz, described the ruling as a significant development, asserting that there is now no justification for Hicks not to receive the same protection as the nine other British nationals who were released from Guantanamo Bay without charges following British Government interventions.",
    "eval": "same"
  },
  {
    "Original premise": "Rockweed has been harvested commercially in Nova Scotia since the late 1950's and is currently the most important commercial seaweed in Atlantic Canada.",
    "hypothesis": "Marine vegetation is harvested.",
    "label": "0",
    "Rephrased premise": "Commercial harvesting of rockweed began in Nova Scotia in the late 1950s, and it now stands as the leading commercial seaweed in Atlantic Canada.",
    "eval": "same"
  },
  {
    "Original premise": "Pakistan President Pervez Musharraf has ordered security forces to take firm action against rioters following the assassination of opposition leader Benazir Bhutto. The violence has left at least 44 people dead and dozens injured. Mr. Musharraf insisted the measures were to protect people. VOA's Ayaz Gul reports from Islamabad that a bitter dispute has also erupted over how the 54-year-old politician died and who was behind her assassination.",
    "hypothesis": "Musharraf has ordered rioters to take firm action against security forces.",
    "label": "1",
    "Rephrased premise": "President Pervez Musharraf of Pakistan has directed security forces to decisively address the unrest following the assassination of opposition leader Benazir Bhutto, which has resulted in at least 44 fatalities and numerous injuries. Musharraf emphasized that these actions are intended to safeguard the public. Meanwhile, a contentious debate has emerged regarding the circumstances of the 54-year-old politician's death and the identity of those responsible, as reported by VOA's Ayaz Gul from Islamabad.",
    "eval": "same"
  },
  {
    "Original premise": "A month after Gov. David A. Paterson dropped his proposal for a soda tax, New York City's health commissioner has written an article advocating \"hefty\" taxes on sodas and sports drinks containing sugar. Such a tax, the article said, could be the biggest boon to public health since tobacco taxes. The commissioner, Dr. Thomas R. Frieden, and Kelly D. Brownell of Yale University, his co-author, argue in the New England Journal of Medicine that a tax of a penny per ounce could reduce consumption by more than 10 percent and raise $1.2 billion a year in New York State alone.",
    "hypothesis": "Michael Bloomberg is the mayor of New York.",
    "label": "1",
    "Rephrased premise": "A month following Gov. David A. Paterson's withdrawal of his soda tax proposal, New York City's health commissioner authored an article supporting substantial taxes on sugary sodas and sports drinks. The article suggests that such a tax could be the most significant public health advancement since tobacco taxes. Dr. Thomas R. Frieden, the commissioner, along with his co-author Kelly D. Brownell from Yale University, argue in the New England Journal of Medicine that a tax of one cent per ounce could cut consumption by over 10 percent and generate $1.2 billion annually in New York State alone.",
    "eval": "same"
  },
  {
    "Original premise": "Officials said the Finnish suspect was among the dead but did not provide a motive for the attack.",
    "hypothesis": "Officials said that the suspect, a Finnish citizen, was among the dead.",
    "label": "0",
    "Rephrased premise": "The authorities confirmed that the Finnish suspect was included among the deceased, though they did not disclose any motive for the assault.",
    "eval": "same"
  },
  {
    "Original premise": "Britain agreed to lift by March 31 a 150-mile military protection zone enforced around the islands since Argentina invaded them in 1982.",
    "hypothesis": "The military protection zone around Falklands was lifted.",
    "label": "0",
    "Rephrased premise": "Britain consented to remove by March 31 the 150-mile military protection zone that had been maintained around the islands since Argentina's 1982 invasion.",
    "eval": "same"
  },
  {
    "Original premise": "The ferry owner PT Nur Budi's spokesman blamed Indonesian port authorities for the tragedy. \"The passenger capacity of the ferry is 205 people but the port administrator accepted more passengers as they thought it was possible,\" he said. The National Meteorological and Geophysics Agency, however, had published and raised an alert signal about high waves on Friday. It specifically stated that \"Saturday 10th and Sunday 11th, Indonesian waters would have witnessed storm force waves,\" but despite the dire warnings KM Teratai set for the seas.",
    "hypothesis": "An Indonesian ferry with 300 passengers sank.",
    "label": "1",
    "Rephrased premise": "PT Nur Budi's spokesperson attributed the disaster to Indonesian port officials, stating that although the ferry's capacity was 205 passengers, the port allowed more on board, believing it was feasible. Meanwhile, the National Meteorological and Geophysics Agency had issued a warning about high waves, noting that on Saturday the 10th and Sunday the 11th, Indonesian waters would experience storm-level waves. Despite these severe warnings, KM Teratai proceeded to sea.",
    "eval": "same"
  },
  {
    "Original premise": "Tension and battles between Greek police and anarchist demonstrators took place in the centre of Athens, Greece, during the anti-war demonstration of the 4th European Social Forum which is taking place in the Greek capital, from 4 to 7 of May 2006. The march of the approximately 1,000 anarchists ended with clashes between groups of anarchists and police. Riot police used tear gas, while a branch of a Greek bank, a fast-food store and around 50 shop windows in central Athens were damaged.",
    "hypothesis": "The riots in Greece started on December 6.",
    "label": "1",
    "Rephrased premise": "During the anti-war protest of the 4th European Social Forum held in Athens, Greece, from May 4 to 7, 2006, tensions and confrontations erupted between Greek police and anarchist demonstrators in the city center. The march, involving about 1,000 anarchists, culminated in violent clashes with the police. Riot officers deployed tear gas, and significant damage occurred, including a Greek bank branch, a fast-food outlet, and approximately 50 shop windows in central Athens.",
    "eval": "same"
  },
  {
    "Original premise": "Libya's case against Britain and the US concerns the dispute over their demand for extradition of Libyans charged with blowing up a Pan Am jet over Lockerbie in 1988.",
    "hypothesis": "One case involved the extradition of Libyan suspects in the Pan Am Lockerbie bombing.",
    "label": "0",
    "Rephrased premise": "Libya's legal action against Britain and the US pertains to the conflict over their request for the extradition of Libyans accused of the 1988 Pan Am jet bombing over Lockerbie.",
    "eval": "same"
  },
  {
    "Original premise": "Argentina sought help from Britain on its privatization program and encouraged British investment.",
    "hypothesis": "Argentina sought UK expertise on privatization and agriculture.",
    "label": "1",
    "Rephrased premise": "Argentina requested assistance from Britain regarding its privatization efforts and promoted British investment.",
    "eval": "same"
  },
  {
    "Original premise": "Instead of action on crime, we got the federal long gun registry, which became a bloated bureaucratic nightmare to responsible hunters, farmers and rural Canadians. It cost taxpayers some CA$2 billion and it hasn't done a thing to reduce gun crime. said Harper. The Conservatives have provided amnesty for unregistered gun owners. At this time there is no legislation set before the House of Commons. Conservative Garry Breitkreuz from Saskatchewan tabled the bill killing the long-gun registry.",
    "hypothesis": "Garry Breitkreuz is a member of the Conservative Party.",
    "label": "0",
    "Rephrased premise": "The federal long gun registry, instead of addressing crime, turned into a cumbersome bureaucratic issue for responsible hunters, farmers, and rural Canadians, costing taxpayers around CA$2 billion without reducing gun crime, according to Harper. The Conservatives have granted amnesty to unregistered gun owners, and currently, no legislation is before the House of Commons. Conservative Garry Breitkreuz from Saskatchewan introduced the bill to dismantle the long-gun registry.",
    "eval": "same"
  },
  {
    "Original premise": "The researchers in the latest study fed one group of mice a diet in which 60 percent of calories came from fat. The diet started when the mice, all males, were 1 year old, which is middle-age in mouse longevity. As expected, the mice soon developed signs of impending diabetes, with grossly enlarged livers, and started to die much sooner than mice fed a standard diet.",
    "hypothesis": "At the age of one year, male mice were fed with a diet in which 60 percent of calories came from fat.",
    "label": "0",
    "Rephrased premise": "The latest study involved feeding a group of mice a diet where 60 percent of their caloric intake was derived from fat. This regimen began when the mice, all male, reached the age of one year, which is considered middle-aged for mice. Predictably, the mice soon exhibited early signs of diabetes, such as significantly enlarged livers, and began dying much earlier than those on a regular diet.",
    "eval": "same"
  },
  {
    "Original premise": "The soldiers, who were said to have been wearing Arab headdress, were accused of firing at Iraqi police when stopped at a road block.",
    "hypothesis": "The soldiers were driving a civilian car and were dressed in civilian clothes when a shooting took place between them and Iraqi patrols.",
    "label": "1",
    "Rephrased premise": "The soldiers, reportedly dressed in Arab attire, faced allegations of shooting at Iraqi police officers when halted at a checkpoint.",
    "eval": "same"
  },
  {
    "Original premise": "The United States would like to see U.N. weapons inspectors return to Iraq providing the Iraqis take concrete, affirmative and demonstrable actions \"to show full co-operation\" Clinton said.",
    "hypothesis": "U.N. weapons inspectors could stay in Iraq, Clinton said.",
    "label": "1",
    "Rephrased premise": "The United States desires the return of U.N. weapons inspectors to Iraq, contingent upon Iraqis taking clear, positive, and verifiable steps to demonstrate full cooperation, according to Clinton.",
    "eval": "same"
  },
  {
    "Original premise": "On 12 August, the San Carlos Battalion came across mines placed in their path and one soldier was killed while two were seriously injured. Meanwhile on 10 August, urban commandos took a patrol car by surprise and dropped a grenade inside the car, injuring four and partially destroying the vehicle.",
    "hypothesis": "A patrol car was attacked by the San Carlos Battalion.",
    "label": "1",
    "Rephrased premise": "On August 12, the San Carlos Battalion encountered mines on their route, resulting in one soldier's death and two others being severely wounded. In a separate incident on August 10, urban commandos ambushed a patrol car, tossing a grenade inside, which injured four individuals and caused partial damage to the vehicle.",
    "eval": "same"
  },
  {
    "Original premise": "Bomb-sniffing dogs were brought to Rodriguez's Mulberry St. apartment.",
    "hypothesis": "Bomb-sniffing dogs were estimated at Rodriguez's Mulberry St. apartment.",
    "label": "1",
    "Rephrased premise": "Bomb-sniffing dogs were deployed to the Mulberry St. apartment of Rodriguez.",
    "eval": "same"
  },
  {
    "Original premise": "Startling new research into mobile phones claims they may reduce a man's sperm count by up to 30%.",
    "hypothesis": "Male fertility may be affected by use of a mobile phones.",
    "label": "0",
    "Rephrased premise": "Research has revealed surprising findings that mobile phones could potentially lower a man's sperm count by as much as 30%.",
    "eval": "same"
  },
  {
    "Original premise": "Workers at the Lufthansa Technik Automotive plant in Co Dublin are due to vote today on changes to work practices being sought by the company's management. The aircraft engine maintenace firm has warned that rejection of the proposals would threaten a planned \u20ac17230m investment in the Rathcoole plant and put jobs at risk. LTA says it is not seeking any lay-offs or pay cuts and the planned investment would guarantee the future of the plant for the next 15 years. The work-practice changes are being sought following the collapse of Labour Relations Commission talks on the matter.",
    "hypothesis": "Lufthansa Technik Automotive fires 30 workers.",
    "label": "1",
    "Rephrased premise": "The employees at the Lufthansa Technik Automotive facility in Co Dublin are set to cast their votes today regarding modifications to work practices proposed by the management. The aircraft engine maintenance company has cautioned that if the proposals are rejected, it could jeopardize a planned \u20ac17230m investment in the Rathcoole plant and endanger jobs. LTA has stated that it is not aiming for any layoffs or salary reductions, and the proposed investment would secure the plant's future for the next 15 years. The request for changes in work practices comes after the breakdown of discussions with the Labour Relations Commission on the issue.",
    "eval": "same"
  },
  {
    "Original premise": "It rewrites the rules of global trade, established by the General Agreement on Tariffs and Trade, or GATT, in 1947, and modified in multiple rounds of negotiations since then.",
    "hypothesis": "GATT was formed in 1947.",
    "label": "1",
    "Rephrased premise": "The rules of global trade, initially set by the General Agreement on Tariffs and Trade (GATT) in 1947 and subsequently altered through various negotiation rounds, are being redefined.",
    "eval": "same"
  },
  {
    "Original premise": "Vice President Dick Cheney on Tuesday hurled an obscenity on the Senate floor to punctuate an angry exchange with Vermont Sen. Patrick Leahy as all senators gathered for their annual photo.",
    "hypothesis": "Cheney cursed at Sen. Patrick Leahy.",
    "label": "0",
    "Rephrased premise": "During a heated interaction on the Senate floor on Tuesday, Vice President Dick Cheney used a profanity directed at Vermont Senator Patrick Leahy while all senators were assembled for their yearly photograph.",
    "eval": "same"
  },
  {
    "Original premise": "In the May 2005 general election Michael Howard failed to unseat the Labour Government, although the Conservatives did gain 33 seats, playing the most significant role in reducing Labour's majority from 167 to 66.",
    "hypothesis": "In the May 2005 general election Conservatives got 33 seats.",
    "label": "1",
    "Rephrased premise": "During the general election in May 2005, Michael Howard was unsuccessful in displacing the Labour Government, despite the Conservatives securing an additional 33 seats, which was crucial in cutting Labour's majority from 167 to 66.",
    "eval": "same"
  },
  {
    "Original premise": "The Norwegian Newspaper Corpus is a large and self-expanding corpus of Norwegian newspaper texts. The collection of this dynamic and continually growing corpus began in 1998.",
    "hypothesis": "Dagbladet is a Norwegian newspaper.",
    "label": "1",
    "Rephrased premise": "The Norwegian Newspaper Corpus comprises an extensive and ever-growing collection of Norwegian newspaper articles, with its compilation starting in 1998.",
    "eval": "same"
  },
  {
    "Original premise": "The Dutch, who ruled Indonesia until 1949, called the city of Jakarta Batavia.",
    "hypothesis": "Formerly ( until 1949 ) Batavia, Jakarta is largest city and capital of Indonesia.",
    "label": "1",
    "Rephrased premise": "The city now known as Jakarta was referred to as Batavia by the Dutch, who governed Indonesia until 1949.",
    "eval": "same"
  },
  {
    "Original premise": "The name for the newest James Bond film has been announced today. The 22nd film, previously known only as \"Bond 22\", will be called \"Quantum of Solace\". EON Productions who are producing the film made the announcement today at Pinewood Studios, where production for the film has been under way since last year. The name of the film was inspired by a short story (of the same name) from For Your Eyes Only by Bond creator, Ian Fleming.",
    "hypothesis": "James Bond was created by Ian Fleming.",
    "label": "0",
    "Rephrased premise": "EON Productions revealed today that the latest James Bond movie, previously referred to as \"Bond 22,\" will be titled \"Quantum of Solace.\" This announcement was made at Pinewood Studios, where filming has been ongoing since last year. The film's title is drawn from a short story by Ian Fleming, the creator of James Bond, found in the collection For Your Eyes Only.",
    "eval": "same"
  },
  {
    "Original premise": "The gastric bypass operation, also known as stomach stapling, has become the most common surgical procedure for treating obesity.",
    "hypothesis": "Obesity is medically treated.",
    "label": "0",
    "Rephrased premise": "The gastric bypass surgery, often referred to as stomach stapling, has emerged as the predominant surgical method for addressing obesity.",
    "eval": "same"
  },
  {
    "Original premise": "The cost of the consumer of the United States fell in June.",
    "hypothesis": "U.S. consumer spending dived in June.",
    "label": "1",
    "Rephrased premise": "In June, the expenses of U.S. consumers decreased.",
    "eval": "same"
  },
  {
    "Original premise": "The former leader of Iraq was rushed to hospital last Sunday after refusing to eat for sixteen days. But according to news agencies, he has ended the hunger strike by eating lunch at the court in Baghdad. \"Saddam ate beef and rice and cola with bread which he brought from hospital,\" one source told Reuters news agency. He was fasting with three co-defendants, and they were demanding more security for their defence lawyers, three of whom have been murdered.",
    "hypothesis": "Some Australians have fasted to protest.",
    "label": "1",
    "Rephrased premise": "The former Iraqi leader was taken to the hospital last Sunday after a sixteen-day hunger strike. However, news agencies report that he has broken the fast by having lunch at the Baghdad court. \"Saddam consumed beef, rice, and cola with bread he brought from the hospital,\" a source informed Reuters. He was fasting alongside three co-defendants, who were calling for increased security for their defense lawyers, three of whom have been killed.",
    "eval": "same"
  },
  {
    "Original premise": "Sonia Gandhi can be defeated in the next elections in India by BJP.",
    "hypothesis": "Sonia Gandhi is defeated by BJP.",
    "label": "1",
    "Rephrased premise": "BJP has the potential to defeat Sonia Gandhi in the upcoming elections in India.",
    "eval": "same"
  },
  {
    "Original premise": "Israeli Prime Minister Ariel Sharon has said that Mahmoud Abbas is a man that Israel can do business with.",
    "hypothesis": "Palestinian leader, Mahmoud Abbas, may be someone Israel can talk with.",
    "label": "0",
    "Rephrased premise": "Israeli Prime Minister Ariel Sharon has expressed that Mahmoud Abbas is someone with whom Israel can engage in negotiations.",
    "eval": "same"
  },
  {
    "Original premise": "The Tempe Hometown Premiere will have all the trappings of a mega-Hollywood event including red carpet arrivals, klieg lights, screaming fans, paparazzi and media coverage of the film`s stars headed by Wolverine himself, Hugh Jackman and including Liev Schreiber, Ryan Reynolds, Taylor Kitsch, Lynn Collins, will.i.am and director Gavin Hood. Following a hard-fought contest stretching over thousands of cities and towns across the United States, Tempe emerged triumphant in a far-reaching, citywide bid to nab the gala event. Mayor Hugh Hallman himself rallied the community, which stepped up with dozens of videos that were posted on YouTube.",
    "hypothesis": "Hugh Jackman plays the role of Wolverine in the movie.",
    "label": "0",
    "Rephrased premise": "The Tempe Hometown Premiere is set to feature all the elements of a major Hollywood event, including red carpet arrivals, klieg lights, enthusiastic fans, paparazzi, and media coverage of the film's stars, led by Hugh Jackman as Wolverine, along with Liev Schreiber, Ryan Reynolds, Taylor Kitsch, Lynn Collins, will.i.am, and director Gavin Hood. After a competitive effort involving numerous cities and towns across the U.S., Tempe succeeded in securing the gala event, with Mayor Hugh Hallman rallying the community, which contributed numerous videos uploaded to YouTube.",
    "eval": "same"
  },
  {
    "Original premise": "In October, however, amid rising tensions between the government and opposition groups, a car bomb seriously injured an opposition politician and killed his driver, in Beirut.",
    "hypothesis": "A member of the opposition was injured in a car bomb attack in Beirut.",
    "label": "0",
    "Rephrased premise": "In October, as tensions escalated between the government and opposition factions, a car bomb in Beirut gravely wounded an opposition politician and resulted in the death of his driver.",
    "eval": "same"
  },
  {
    "Original premise": "Weinstock painstakingly reviewed dozens of studies for evidence of any link between sunscreen use and either an increase or decrease in melanoma.",
    "hypothesis": "skin cancer numbers increase.",
    "label": "1",
    "Rephrased premise": "Weinstock meticulously examined numerous studies to find any evidence connecting sunscreen usage with changes in melanoma rates, whether rising or falling.",
    "eval": "same"
  },
  {
    "Original premise": "Most people are familiar with the idea of St. Bernards or other dogs taking part in rescue and recovery efforts. Robots might also take part in search and rescue missions.",
    "hypothesis": "Robots are used to find missing victims.",
    "label": "0",
    "Rephrased premise": "Many individuals recognize the concept of St. Bernards or similar dogs being involved in rescue operations. Similarly, robots can also participate in search and rescue missions.",
    "eval": "same"
  },
  {
    "Original premise": "Illinois born Charlton Heston was a 27-year-old actor from Broadway and television when he arrived in Hollywood for a five-picture contract with Hal Wallis.",
    "hypothesis": "Charlton Heston was born in Illinois.",
    "label": "0",
    "Rephrased premise": "Charlton Heston, originally from Illinois, was a 27-year-old actor with experience in Broadway and television when he came to Hollywood under a five-film agreement with Hal Wallis.",
    "eval": "same"
  },
  {
    "Original premise": "Ruth's 1927 single season record of 60 home runs stood unsurpassed until Roger Maris hit 61 in 1961.",
    "hypothesis": "Babe Ruth hit 60 home runs in his lifetime.",
    "label": "1",
    "Rephrased premise": "Roger Maris broke Ruth's 1927 single-season record of 60 home runs by hitting 61 in 1961.",
    "eval": "same"
  },
  {
    "Original premise": "The German technology was employed to build Shanghai's existing maglev line, the first in the world to be used commercially.",
    "hypothesis": "Maglev is commercially used.",
    "label": "0",
    "Rephrased premise": "German engineering was utilized in constructing Shanghai's current maglev line, which is the first to be commercially operational worldwide.",
    "eval": "same"
  },
  {
    "Original premise": "SEATTLE \u2014 United Nations gang leader Clay Roueche should spend at least 30 years behind bars after his surprise guilty plea here Tuesday for conspiracy to smuggle cocaine, marijuana and illicit drug profits, U.S. Attorney Jeffrey Sullivan said. Sullivan said outside court that Roueche's leadership of the violent UN gang and the sophistication of the international drug conspiracy will all be factors used to argue for a longer sentence \u2014 up to life \u2014 for the 33-year-old Canadian. And he stressed that the plea agreement reached with Roueche did not include \"any kind of break with respect to sentencing.\".",
    "hypothesis": "Clay Roueche is 33 years old.",
    "label": "0",
    "Rephrased premise": "In Seattle, U.S. Attorney Jeffrey Sullivan stated that Clay Roueche, leader of the United Nations gang, should face a minimum of 30 years in prison following his unexpected guilty plea on Tuesday for conspiring to smuggle cocaine, marijuana, and illegal drug proceeds. Sullivan mentioned outside the courtroom that Roueche's role in the violent UN gang and the complexity of the international drug operation would be considered in advocating for a longer sentence, potentially up to life, for the 33-year-old Canadian. He emphasized that the plea deal with Roueche did not offer any leniency regarding sentencing.",
    "eval": "same"
  },
  {
    "Original premise": "Baker's voice will replace the earlier computer-synthesized voice that was previously used. BT said that his voice was chosen as it was instantly recognisable. It took him 11 days to record 11,593 phrases and sounds which could then be broken down and reassembled by a computer to make new words. It then took five months to process these recordings to make a workable service. BT have said that there will be no barriers as to what Tom Baker's voice can 'say', including rude words. \"What appeals to me most is the thought that I will be bringing good news to people whether it is a cheeky message, a birthday greeting, or just a quick hello,\" said Baker.",
    "hypothesis": "Tom Baker works for BT.",
    "label": "0",
    "Rephrased premise": "Tom Baker's voice is set to replace the previously used computer-generated voice. BT selected his voice for its immediate recognizability. Over 11 days, he recorded 11,593 phrases and sounds, which were then processed over five months to create a functional service that allows a computer to break down and reassemble these recordings into new words. BT stated that there are no restrictions on what Tom Baker's voice can articulate, including inappropriate words. Baker expressed his excitement about delivering messages, whether they are playful, celebratory, or simply a friendly greeting.",
    "eval": "same"
  },
  {
    "Original premise": "The Norwegian Nobel Committee is responsible for the selection of the candidates and the choice of Prize Winners for the Peace Prize. The Committee is composed of five members appointed by the Storting (Norwegian parliament). The Peace Prize is awarded in Oslo, Norway and not in Stockholm, Sweden like the other Nobel Prizes.",
    "hypothesis": "Nobel Peace Prize candidates have been chosen.",
    "label": "1",
    "Rephrased premise": "The Norwegian Nobel Committee, consisting of five members appointed by the Norwegian parliament (Storting), is tasked with selecting candidates and deciding the recipients of the Peace Prize, which is presented in Oslo, Norway, unlike the other Nobel Prizes awarded in Stockholm, Sweden.",
    "eval": "same"
  },
  {
    "Original premise": "More than 150 dolphins, marine turtles and beaked whales have been washed up dead on beaches in Africa.",
    "hypothesis": "Dead dolphins, turtles and whales have been found on African beaches.",
    "label": "0",
    "Rephrased premise": "Over 150 dolphins, marine turtles, and beaked whales have been discovered dead along the shores of Africa.",
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
