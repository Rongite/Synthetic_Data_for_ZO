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
        "question": "is the blues brothers based on a true story",
        "answer": False,
        "Original passage": "The Blues Brothers are an American blues and soul revivalist band which was founded in 1978 by comedy actors Dan Aykroyd and John Belushi as part of a musical sketch on Saturday Night Live. Belushi and Aykroyd fronted the band, respectively in character as lead vocalist ``Joliet'' Jake Blues and harmonica player\/vocalist Elwood Blues. The band was composed of well-known and respected musicians, and debuted as the musical guest on the April 22, 1978, episode of Saturday Night Live, opening the show performing ``Hey Bartender'', and later ``Soul Man''.",
        "Rephrased passage": "The Blues Brothers is a blues and soul band from the United States, established in 1978 by comedians Dan Aykroyd and John Belushi as part of a Saturday Night Live musical sketch. Aykroyd and Belushi led the band, playing the roles of Elwood Blues, a harmonica player and vocalist, and \"Joliet\" Jake Blues, the lead singer. The group consisted of recognized and accomplished musicians, making their first appearance as musical guests on the April 22, 1978 episode of Saturday Night Live, where they performed \"Hey Bartender\" and later \"Soul Man.\"",
        "eval": "same"
    },
    {
        "question": "is newcastle upon tyne the same as newcastle",
        "answer": True,
        "Original passage": "Newcastle upon Tyne (locally \/nju\u02d0\u02c8k\u00e6s\u0259l\/ ( listen)), commonly known as Newcastle, is a city in Tyne and Wear, North East England, 103 miles (166 km) south of Edinburgh and 277 miles (446 km) north of London on the northern bank of the River Tyne, 8.5 mi (13.7 km) from the North Sea. Newcastle is the most populous city in the North East, and forms the core of the Tyneside conurbation, the eighth most populous urban area in the United Kingdom. Newcastle is a member of the English Core Cities Group and is a member of the Eurocities network of European cities.",
        "Rephrased passage": "Newcastle upon Tyne, often simply referred to as Newcastle, is a city situated in Tyne and Wear in the North East of England, positioned 103 miles south of Edinburgh and 277 miles north of London along the northern bank of the River Tyne, 8.5 miles from the North Sea. As the most populous city in the North East, it is central to the Tyneside conurbation, which ranks eighth in population among urban areas in the United Kingdom. Newcastle is part of the English Core Cities Group and is associated with the Eurocities network of European cities.",
        "eval": "same"
    },
    {
        "question": "is post office and royal mail the same",
        "answer": False,
        "Original passage": "The Post Office provides information on services and accepts postal items and payment on behalf of the two collection and delivery divisions of Royal Mail Group, Royal Mail and Parcelforce. These include a variety of ordinary and guaranteed services both for delivery within the United Kingdom and to international destinations. Postage stamps (including commemorative stamps and other philatelic items) are sold, while applications for redirection of mail are accepted on behalf of Royal Mail.",
        "Rephrased passage": "\"Post Office functions involve offering information on services and handling postal items and payments for Royal Mail Group's two divisions, namely Royal Mail and Parcelforce. These divisions provide a range of general and guaranteed services for deliveries domestically across the United Kingdom and internationally. The Post Office also sells postage stamps, including commemorative ones and various philatelic items, and processes mail redirection applications on behalf of Royal Mail.\"",
        "eval": "same"
    },
    {
        "question": "has any supreme court nominee not been confirmed",
        "answer": True,
        "Original passage": "Justices are nominated by the president and then confirmed by the U.S. Senate. A nomination to the Court is considered to be official when the Senate receives a signed nomination letter from the president naming the nominee, which is then entered in the Senate's record. There have been 37 unsuccessful nominations to the Supreme Court of the United States. Of these, 11 nominees were rejected in Senate roll-call votes, 11 were withdrawn by the president, and 15 lapsed at the end of a session of Congress. Six of these unsuccessful nominees were subsequently nominated and confirmed to other seats on the Court. Additionally, although confirmed, seven nominees either declined office or (in one instance) died before assuming office.",
        "Rephrased passage": "Supreme Court justices are appointed by the President and must be confirmed by the U.S. Senate. An official nomination is recorded when the Senate receives a signed letter from the President, naming the nominee. Throughout history, there have been 37 failed nominations to the U.S. Supreme Court. Among these, 11 were rejected by roll-call votes in the Senate, another 11 were withdrawn by the President, and 15 nominations lapsed at the end of a Congressional session. Notably, six nominees who initially failed later received nominations and confirmations for different positions on the Court. Furthermore, seven nominees, despite being confirmed, either declined to take the position or, in one case, passed away before they could assume office.",
        "eval": "same"
    },
    {
        "question": "is thanksgiving always the last thursday of the month",
        "answer": True,
        "Original passage": "Thanksgiving, or Thanksgiving Day, is a public holiday celebrated on the fourth Thursday of November in the United States. It originated as a harvest festival. Thanksgiving has been celebrated nationally on and off since 1789, after Congress requested a proclamation by George Washington. It has been celebrated as a federal holiday every year since 1863, when, during the American Civil War, President Abraham Lincoln proclaimed a national day of ``Thanksgiving and Praise to our beneficent Father who dwelleth in the Heavens,'' to be celebrated on the last Thursday in November. Together with Christmas and the New Year, Thanksgiving is a part of the broader fall\/winter holiday season in the U.S.",
        "Rephrased passage": "Thanksgiving Day is a public holiday taking place on the fourth Thursday of November in the United States, originating from a harvest festival. Although it has been celebrated nationally since 1789 following Congress's request for George Washington's proclamation, it was officially declared a federal holiday in 1863 by President Abraham Lincoln amid the American Civil War. Lincoln proclaimed a national day of \"Thanksgiving and Praise to our beneficent Father who dwelleth in the Heavens,\" to be observed on the last Thursday of November. Alongside Christmas and New Year\u2019s, Thanksgiving is part of the wider fall and winter holiday season in the U.S.",
        "eval": "same"
    },
    {
        "question": "is lord of the rings after the hobbit",
        "answer": True,
        "Original passage": "The Lord of the Rings is an epic high fantasy novel written by English author and scholar J.R.R. Tolkien. The story began as a sequel to Tolkien's 1937 fantasy novel The Hobbit, but eventually developed into a much larger work. Written in stages between 1937 and 1949, The Lord of the Rings is one of the best-selling novels ever written, with over 150 million copies sold.",
        "Rephrased passage": "J.R.R. Tolkien, an English author and scholar, crafted The Lord of the Rings as a high fantasy epic. Originally intended as a follow-up to his 1937 novel, The Hobbit, it evolved into a significantly expanded narrative. The writing occurred in phases from 1937 to 1949, resulting in one of the best-selling books of all time, with sales exceeding 150 million copies.",
        "eval": "same"
    },
    {
        "question": "is the fallen movie going to have a sequel",
        "answer": False,
        "Original passage": "In December 2014, it was announced that Torment, the second installment in the Fallen book series, was in development. It is unknown whether the last two novels, Passion and Rapture, and the spin-off novel, Unforgiven, will be adapted as well. In 2017, producer Kevan Van Thompson asked the fans if they want an adaptation of ``Torment'', showing that the sequel still could be made.",
        "Rephrased passage": "In December 2014, the production of Torment, the sequel to the Fallen book series, was announced. However, the adaptation status of the remaining novels\u2014Passion, Rapture, and the spin-off Unforgiven\u2014remains uncertain. In 2017, producer Kevan Van Thompson questioned fans about their interest in a Torment adaptation, indicating that a sequel is still a possibility.",
        "eval": "same"
    },
    {
        "question": "does anyone on instagram have 1 billion followers",
        "answer": False,
        "Original passage": "This list contains the top 50 accounts with the most followers on the social photo-sharing platform Instagram. As of October 2018, the most followed user is Instagram's own account, with over 260 million followers. Cristiano Ronaldo is the most followed individual, with over 144 million followers. Twelve accounts have exceeded 100 million followers on the site.",
        "Rephrased passage": "The list enumerates the top 50 accounts with the highest follower counts on Instagram, the social photo-sharing app. As of October 2018, Instagram's official account tops the list with over 260 million followers. Cristiano Ronaldo is recognized as the most followed person, boasting over 144 million followers. On this platform, twelve accounts have surpassed the 100 million follower mark.",
        "eval": "same"
    },
    {
        "question": "in the phantom menace is padme the queen",
        "answer": True,
        "Original passage": "George Lucas, Rick McCallum, and casting director Robin Gurland auditioned over 200 actresses for the part of Padm\u00e9 Amidala. They chose 16-year-old actress Natalie Portman to play the role. According to The Phantom Menace production notes, ``The role required a young woman who could be believable as the ruler of that planet, but at the same time be vulnerable and open.'' Portman's performances in The Professional (1994) and Beautiful Girls (1996) impressed Lucas. He stated, ``I was looking for someone who was young, strong, along the lines of Leia. Natalie embodied all those traits and more.''",
        "Rephrased passage": "The casting for the role of Padm\u00e9 Amidala in The Phantom Menace involved George Lucas, Rick McCallum, and casting director Robin Gurland reviewing over 200 candidates. Ultimately, they selected Natalie Portman, then 16 years old, for the part. The production notes of The Phantom Menace highlight that the role demanded a character who was credible as the planet's ruler while also being approachable and susceptible. Lucas was particularly impressed by Portman's work in The Professional (1994) and Beautiful Girls (1996), stating he was searching for someone youthful and strong, akin to Leia, and Natalie embodied those qualities superbly.",
        "eval": "same"
    },
    {
        "question": "is lord of the rings considered an epic",
        "answer": True,
        "Original passage": "The Lord of the Rings is an epic high fantasy novel written by English author and scholar J.R.R. Tolkien. The story began as a sequel to Tolkien's 1937 fantasy novel The Hobbit, but eventually developed into a much larger work. Written in stages between 1937 and 1949, The Lord of the Rings is one of the best-selling novels ever written, with over 150 million copies sold.",
        "Rephrased passage": "J.R.R. Tolkien, an English author and scholar, wrote The Lord of the Rings as an epic high fantasy novel. Initially intended to be a follow-up to Tolkien's 1937 fantasy novel, The Hobbit, the story evolved into a significantly more expansive work. Completed in stages from 1937 to 1949, The Lord of the Rings ranks among the best-selling novels of all time, with over 150 million copies sold.",
        "eval": "same"
    },
    {
        "question": "does the euro sign go before the number",
        "answer": True,
        "Original passage": "The euro sign (\u20ac) is the currency sign used for the euro, the official currency of the Eurozone in the European Union (EU). The design was presented to the public by the European Commission on 12 December 1996. The international three-letter code (according to ISO standard ISO 4217) for the euro is EUR. In Unicode it is encoded at U+20AC \u20ac euro sign (HTML &#8364; &euro; ). In English, the sign precedes the value (for instance, \u20ac10, not 10 \u20ac, unlike most other European languages). In some style guides, but not others, the euro sign is unspaced.",
        "Rephrased passage": "The euro sign (\u20ac) represents the euro, the official currency of the Eurozone within the European Union (EU). Unveiled by the European Commission on December 12, 1996, the euro's international three-letter code is EUR, as per ISO standard ISO 4217. In Unicode, it is represented as U+20AC \u20ac euro sign (HTML &#8364; &euro; ). In English, the euro sign is placed before the amount (e.g., \u20ac10, rather than 10 \u20ac), which differs from the convention in most other European languages. Some style guides specify that the euro sign should be unspaced, while others do not.",
        "eval": "same"
    },
    {
        "question": "can stainless steel be used on induction cooktop",
        "answer": True,
        "Original passage": "For nearly all models of induction cooktops, a cooking vessel must be made of, or contain, a ferrous metal such as cast iron or some stainless steels. The iron in the pot concentrates the current to produce heat in the metal. If the metal is too thin, or does not provide enough resistance to current flow, heating will not be effective. Most induction tops will not heat copper or aluminum vessels because the magnetic field cannot produce a concentrated current; ``all metal'' induction tops use much higher frequencies to overcome that effect. Any vessel can be used if placed on a suitable metal disk which functions as a conventional hotplate.",
        "Rephrased passage": "Induction cooktops generally require cookware to be composed of or include ferrous metals, like cast iron or certain types of stainless steel. The iron in the cookware focuses the current, generating heat within the metal. If the metal is excessively thin or fails to provide sufficient resistance to the electrical flow, the heating will be ineffective. Copper and aluminum pieces typically won't heat up on most induction surfaces because the magnetic field is unable to create a concentrated current. Some induction models, termed \"all metal,\" utilize higher frequencies to counter this limitation. Any type of cookware can be used when placed atop an appropriate metal disk that acts like a traditional hotplate.",
        "eval": "same"
    },
    {
        "question": "is job termination the same as being fired",
        "answer": True,
        "Original passage": "``Firing'' is a common colloquial term in the English language (particularly used in the U.S.) for termination. The term ``firing'' may have been initiated in the 1910s at the National Cash Register Company. Other terms for dismissal are being ``sacked'', ``canned'', ``let go'', ``ran-off'', ``axed'', ``given walking papers'', ``given the pink slip'' or ``boned''. Other terms, more often used in Commonwealth countries, include ``to get the boot'' and ``to get the sack''.",
        "Rephrased passage": "\"Being 'fired' is a widely used informal term, especially in the U.S., for job termination. The expression 'firing' is thought to have originated around the 1910s at the National Cash Register Company. Other synonyms for being dismissed from a job include 'sacked,' 'canned,' 'let go,' 'axed,' 'given walking papers,' 'given the pink slip,' or 'boned.' In Commonwealth nations, common phrases include 'to get the boot' or 'to get the sack.'\"",
        "eval": "same"
    },
    {
        "question": "does age have any influence on attentional ability or inattentional blindness",
        "answer": True,
        "Original passage": "In a 2015 study, Cary Stothart, Walter Boot, and Daniel Simons attempted to replicate and extend the findings from both Graham and Burke's 2011 study and Steven Most and colleague's 2000 study on Amazon Mechanical Turk using a sample of 515 participants that varied in age. In this study, participants were tasked with counting the number of times a number of white moving objects crossed the vertical midpoint of a display while ignoring a number of black moving objects. The unexpected object in this case was a gray cross that moved horizontally across the display at various distances from the vertical midpoint (this was manipulated between participants). Overall, they found that inattentional blindness susceptibility increases with age, which replicates the finding from Graham and Burke. In fact, they found that every 10 years of age was associated with a 1.3 fold increase in the probability of displaying inattentional blindness. They also found that the probability of inattentional blindness increases as the distance between the observer's focus of attention and the unexpected object increases, which replicates the finding from Most and colleagues. However, they also found that the relationship that age has with inattentional blindness does not change as a function of the unexpected object's distance from the focus of attention, suggesting that useful field of view does not mediate the relationship between age and inattentional blindness.",
        "Rephrased passage": "A 2015 study conducted by Cary Stothart, Walter Boot, and Daniel Simons aimed to replicate and expand upon research by Graham and Burke in 2011 and by Steven Most and colleagues in 2000, using 515 diverse-in-age participants on Amazon Mechanical Turk. Participants were instructed to count the crossings of several white moving objects over the vertical midpoint of a display, while disregarding multiple black moving objects. The unexpected stimulus was a gray cross moving horizontally across the display at varying distances from the vertical midpoint, differing between participants. The researchers found that susceptibility to inattentional blindness rises with age, supporting Graham and Burke's findings, with an increase in age of ten years leading to a 1.3 times higher likelihood of experiencing inattentional blindness. Additionally, they confirmed Most and colleagues' observation that the probability of such blindness intensifies as the distance grows between the observer's attention focus and the unexpected object. However, they noted that age's relationship with inattentional blindness remains unchanged by the unexpected object's distance from the attention focus, indicating that the useful field of view doesn't mediate age-related inattentional blindness.",
        "eval": "same"
    },
    {
        "question": "is sanskrit the first language of the world",
        "answer": False,
        "Original passage": "Sanskrit belongs to the Indo-European family of languages. It is one of the three ancient documented languages that likely arose from a common root language now referred to as the Proto-Indo-European language:",
        "Rephrased passage": "Sanskrit is a member of the Indo-European language family. It is among the three ancient languages with documentation that are believed to have originated from a common proto-language known as Proto-Indo-European.",
        "eval": "same"
    },
    {
        "question": "is a jack russell considered a small breed",
        "answer": True,
        "Original passage": "The Jack Russell Terrier is a small terrier that has its origins in fox hunting. It is principally white-bodied and smooth, rough or broken-coated but can be any colour.",
        "Rephrased passage": "The Jack Russell Terrier, a small breed of terrier, originated from fox hunting. Typically, it has a predominantly white body and can have a smooth, rough, or broken coat, though it may also exhibit other colors.",
        "eval": "same"
    },
    {
        "question": "does air force one travel with fighter escort",
        "answer": False,
        "Original passage": "The Air Force usually does not have fighter aircraft escort the presidential aircraft over the United States but it has occurred, for example during the attack on the World Trade Center.",
        "Rephrased passage": "Air Force One typically travels without fighter jets accompanying it within the United States, although there have been exceptions, such as during the World Trade Center attacks.",
        "eval": "same"
    },
    {
        "question": "is it legal to own an ar15 in california",
        "answer": False,
        "Original passage": "The Roberti-Roos Assault Weapons Control Act of 1989 banned Colt AR-15 rifles by name in the State of California. California's 2000 Assault Weapons ban went further and banned AR-15s made by other manufacturers by name such as Bushmaster, PWA, and Olympic Arms.",
        "Rephrased passage": "The Roberti-Roos Assault Weapons Control Act of 1989 explicitly prohibited Colt AR-15 rifles within California. Additionally, the state's 2000 Assault Weapons ban extended this prohibition by specifically naming AR-15 models from various other manufacturers, including Bushmaster, PWA, and Olympic Arms.",
        "eval": "same"
    },
    {
        "question": "is the national anthem and star spangled banner the same",
        "answer": True,
        "Original passage": "``The Star-Spangled Banner'' is the national anthem of the United States. The lyrics come from ``Defence of Fort M'Henry'', a poem written on September 14, 1814, by the then 35-year-old lawyer and amateur poet Francis Scott Key after witnessing the bombardment of Fort McHenry by British ships of the Royal Navy in Baltimore Harbor during the Battle of Baltimore in the War of 1812. Key was inspired by the large U.S. flag, with 15 stars and 15 stripes, known as the Star-Spangled Banner, flying triumphantly above the fort during the U.S. victory.",
        "Rephrased passage": "\"The national anthem of the United States is \"The Star-Spangled Banner.\" Its lyrics originate from the poem \"Defence of Fort M'Henry,\" penned on September 14, 1814, by 35-year-old lawyer and amateur poet Francis Scott Key. Key wrote the poem after observing the British Royal Navy's bombardment of Fort McHenry in Baltimore Harbor during the Battle of Baltimore in the War of 1812. He was moved by the sight of the U.S. flag, featuring 15 stars and stripes, famously known as the Star-Spangled Banner, remaining aloft over the fort during America\u2019s triumph.\"",
        "eval": "same"
    },
    {
        "question": "is there an international airport in naples italy",
        "answer": True,
        "Original passage": "Naples International Airport (IATA: NAP, ICAO: LIRN) (Italian: Aeroporto Internazionale di Napoli) is the international airport serving Naples, Italy. It is located 3.2 NM (5.9 km; 3.7 mi) north-northeast of the city in the Capodichino district of Naples. The airport has two terminal buildings: Terminal 1 is for scheduled flights and Terminal 2, located away from the airfield, is used for charter operations.",
        "Rephrased passage": "Naples International Airport, identified by the codes IATA: NAP and ICAO: LIRN, serves as the international airport for Naples, Italy. Situated 3.2 nautical miles (equivalent to 5.9 kilometers or 3.7 miles) to the north-northeast of the city center in the Capodichino area, the airport features two terminal buildings. Terminal 1 handles scheduled flights, while Terminal 2, positioned away from the main airfield, is designated for charter flight operations.",
        "eval": "same"
    },
    {
        "question": "can you buy cadburys creme eggs all year round",
        "answer": False,
        "Original passage": "Creme eggs are available annually between 1 January and Easter Day. In the UK in the 1980s, Cadbury made Creme Eggs available year-round but sales dropped and they returned to seasonal availability.",
        "Rephrased passage": "Creme Eggs are sold each year from January 1st until Easter Day. During the 1980s in the UK, Cadbury offered Creme Eggs throughout the year, but due to declining sales, they reverted to a seasonal offering.",
        "eval": "same"
    },
    {
        "question": "do homologous chromosomes carry information for the same traits",
        "answer": True,
        "Original passage": "A couple of homologous chromosomes, or homologs, are a set of one maternal and one paternal chromosome that pair up with each other inside a cell during meiosis. Homologs have the same genes in the same loci where they provide points along each chromosome which enable a pair of chromosomes to align correctly with each other before separating during meiosis. This is the basis for Mendelian inheritance which characterizes inheritance patterns of genetic material from an organism to its offspring parent developmental cell at the given time and area.",
        "Rephrased passage": "Homologous chromosomes, or homologs, consist of one chromosome from the mother and one from the father, pairing together within a cell during meiosis. These homologs contain identical genes at corresponding loci, allowing them to align properly before they separate in the process of meiosis. This alignment is fundamental to Mendelian inheritance, which describes how genetic material is passed from parents to offspring through developmental cells.",
        "eval": "same"
    },
    {
        "question": "has michigan ever won a national championship in football",
        "answer": True,
        "Original passage": "Michigan began competing in intercollegiate football in 1879. The Wolverines joined the Big Ten Conference at its inception in 1896, and other than a hiatus from 1907 to 1916, have been members since. Michigan has won or shared 42 league titles, and, since the inception of the AP Poll in 1936, has finished in the top 10 a total of 38 times. The Wolverines claim 11 national championships, most recently that of the 1997 squad voted atop the final AP Poll.",
        "Rephrased passage": "Michigan commenced its intercollegiate football endeavors in 1879. They have been part of the Big Ten Conference since its establishment in 1896, with a break from 1907 to 1916, after which they've consistently been members. The Wolverines have secured or shared 42 conference titles and, since the start of the AP Poll in 1936, have finished in the top 10 a total of 38 times. They claim 11 national championships, with the most recent being the 1997 team, which was ranked first in the final AP Poll.",
        "eval": "same"
    },
    {
        "question": "you can tell a lot about a person by how they treat their waiter",
        "answer": True,
        "Original passage": "The Waiter Rule refers to a common belief that one's true character can be gleaned from how one treats staff or service workers, such as a ``waiter''. The rule was one of William H Swanson's 33 Unwritten Rules of Management, copied from Dave Barry's version ``If someone is nice to you but rude to the waiter, they are not a nice person.''",
        "Rephrased passage": "The Waiter Rule is based on the notion that a person's genuine character can be revealed through their behavior towards staff or service workers, like a \"waiter.\" Among William H. Swanson's 33 Unwritten Rules of Management, the idea was adapted from Dave Barry's saying, \"If someone is kind to you but disrespectful to the waiter, they are not truly kind.\"",
        "eval": "same"
    },
    {
        "question": "are there any crocodiles native to north america",
        "answer": True,
        "Original passage": "Within the United States, the American crocodile's distribution is limited to the southern tip of Florida, though at least two have been found as far north as the Tampa Bay area. The current US population, estimated at 2,000, represents a significant recovery from a few hundred in the 1970s.",
        "Rephrased passage": "The American crocodile can be found in the United States, mainly restricted to the southern tip of Florida. However, there have been at least two sightings as far north as the Tampa Bay area. The American crocodile population in the US, which now stands at around 2,000, shows a marked recovery from the mere hundreds in the 1970s.",
        "eval": "same"
    },
    {
        "question": "does lex luthor know who superman is in smallville",
        "answer": True,
        "Original passage": "Season seven displayed Lex's descent into darkness; he has a brother-like relationship with Grant Gabriel (Michael Cassidy), the new editor of the Daily Planet newest editor, until it's revealed Grant is actually a clone of Lex's late brother. After Lex buys the Daily Planet, Grant attempts to keep Lex from being controlling, thus Lex has his brother's clone murdered and staged as a failed mugging. Lex then discovers that the previous symbols are connected to the secret organization Veritas, which his father is a part of. The Veritas members learned that an alien visitor known as ``The Traveler'' would arrive in Smallville during the meteor shower of 1989. At this time, Lex realizes that Lionel has been covering up the Traveler's existence and subsequently kills his own father for it. He eventually discovers that the Veritas members knew of a means to control the Traveler, so Lex sets out to find the device. The device, an orb he finds in the mantle above a fireplace in the Luthor mansion, leads Lex to the Fortress of Solitude, where he is confronted by Clark. Having finally discovered Clark's secret, Lex uses the orb to bring down the Fortress around Clark and himself.",
        "Rephrased passage": "In the seventh season, Lex Luthor's path to darkness becomes evident. He forms a brotherly bond with Grant Gabriel (Michael Cassidy), the new editor at the Daily Planet, who is eventually revealed to be a clone of Lex's deceased brother. After purchasing the Daily Planet, Lex eliminates the clone by staging his murder as a botched robbery to retain control over the editorial decisions. Lex investigates connections to a secret society named Veritas, linked to his father. Members of Veritas anticipated the arrival of \"The Traveler,\" an alien, during the 1989 meteor shower in Smallville. Upon realizing Lionel Luthor was concealing the existence of the Traveler, Lex murders him. Discovering Veritas had a method to control the Traveler, Lex pursues a device to achieve this goal. He discovers an orb in the fireplace mantle at the Luthor mansion, which guides him to the Fortress of Solitude. There, Lex confronts Clark and uncovers Clark's secret, using the orb to collapse the Fortress around them both.",
        "eval": "same"
    },
    {
        "question": "do you have to have a college degree to take the bar exam",
        "answer": True,
        "Original passage": "In the canonical case, lawyers seeking admission must earn a Juris Doctor degree from a law school approved by the jurisdiction, and then pass a bar exam administered by it. Typically, there is also a character and fitness evaluation, which includes a background check. However, there are exceptions to each of these requirements. A lawyer who is admitted in one state is not automatically allowed to practice in any other. Some states have reciprocal agreements that allow attorneys from other states to practice without sitting for another full bar exam; such agreements differ significantly among the states.",
        "Rephrased passage": "Lawyers generally need to obtain a Juris Doctor degree from a law school accredited by the relevant jurisdiction and then pass its bar exam to gain admission. Additionally, a character and fitness assessment, including a background check, is usually required. Nonetheless, exceptions exist for these criteria. An attorney licensed in one state cannot automatically practice in another. Some states have reciprocal agreements permitting lawyers from other states to practice without retaking the entire bar exam, though these agreements vary widely between states.",
        "eval": "same"
    },
    {
        "question": "will there be a season 2 of marvel's iron fist",
        "answer": True,
        "Original passage": "The second season of the American web television series Iron Fist, which is based on the Marvel Comics character of the same name, follows Danny Rand \/ Iron Fist, a martial arts expert with the ability to call upon the power of the Iron Fist. It is set in the Marvel Cinematic Universe (MCU), sharing continuity with the films and other television series of the franchise. The season is produced by Marvel Television in association with ABC Studios, with Raven Metzner serving as showrunner.",
        "Rephrased passage": "The second season of Iron Fist, an American web television series inspired by the Marvel Comics character Danny Rand / Iron Fist, continues the journey of this martial arts expert who harnesses the power of the Iron Fist. This season is part of the Marvel Cinematic Universe (MCU), maintaining its continuity within the franchise's films and TV series. Produced by Marvel Television in collaboration with ABC Studios, Raven Metzner takes the role of showrunner.",
        "eval": "same"
    },
    {
        "question": "is 844 a toll free number in canada",
        "answer": True,
        "Original passage": "In the United States of America, Canada, and other countries participating in the North American Numbering Plan, a toll-free telephone number has one of the area codes 800, 833, 844, 855, 866, 877, and 888.",
        "Rephrased passage": "In Canada, as well as the United States and other countries involved in the North American Numbering Plan, toll-free phone numbers are assigned area codes such as 800, 833, 844, 855, 866, 877, and 888.",
        "eval": "same"
    },
    {
        "question": "does any other country have daylight savings time",
        "answer": True,
        "Original passage": "Most areas in North America and Europe, and some areas in the Middle East, observe daylight saving time (DST), while most areas of Africa and Asia do not. In South America, most countries in the north of the continent near the equator do not observe DST, while Paraguay and southern parts of Brazil do. The practice of observing daylight saving time in Oceania is also mixed, with New Zealand and parts of southeastern Australia observing DST, while most other areas do not.",
        "Rephrased passage": "Many regions in North America and Europe, as well as certain areas in the Middle East, implement daylight saving time (DST). Conversely, the majority of African and Asian regions do not follow this practice. In South America, countries close to the equator in the northern part generally do not observe DST, whereas Paraguay and some southern regions of Brazil do. Similarly, in Oceania, there is variation, with daylight saving time being followed in New Zealand and parts of southeastern Australia, while most other regions in Oceania do not participate.",
        "eval": "same"
    },
    {
        "question": "has anyone ever pitched back to back no hitters",
        "answer": True,
        "Original passage": "John Samuel Vander Meer (November 2, 1914 -- October 6, 1997) was an American professional baseball player. He played in Major League Baseball as a pitcher, most notably for the Cincinnati Reds. Vander Meer is best known for being the only pitcher in Major League Baseball history to throw two consecutive no-hitters. After the impressive start to his major league career, he experienced problems controlling the accuracy of his pitching, and his later career was marked by inconsistent performances.",
        "Rephrased passage": "John Samuel Vander Meer, born on November 2, 1914, and passing away on October 6, 1997, was an American baseball player who pitched in Major League Baseball, primarily for the Cincinnati Reds. Vander Meer holds the unique distinction of being the sole pitcher in MLB history to have thrown two back-to-back no-hitters. Despite a strong start to his major league journey, he later faced challenges with pitching accuracy, leading to variable performances throughout his career.",
        "eval": "same"
    },
    {
        "question": "is there a black bike week in myrtle beach",
        "answer": True,
        "Original passage": "Black Bike Week, also called Atlantic Beach Bikefest and Black Bikers Week, is an annual motorcycle rally in the Myrtle Beach, South Carolina area, held on Memorial Day weekend. It is also sometimes called Black Fill-in-the-Blank Week, because it has evolved to attract many non-motorcycling visitors who come for music, socializing and enjoying the beach. Events include motorcycle racing, concerts, parties, and street festivals. Called a ``one-of-a-kind event'' and ``an exhibitionist's paradise'' by Jeffrey Gettleman, Black Bike Week is ``all about riding, styling and profiling,'' in the words of Mayor Irene Armstrong of Atlantic Beach, South Carolina.",
        "Rephrased passage": "Black Bike Week, also known as Atlantic Beach Bikefest and Black Bikers Week, is an annual motorcycle rally in the vicinity of Myrtle Beach, South Carolina, scheduled for Memorial Day weekend. Occasionally referred to as Black Fill-in-the-Blank Week, it has evolved to draw numerous attendees who come not only for motorcycling but also for music, social interactions, and beach activities. The event features motorcycle races, concerts, parties, and street festivals. Jeffrey Gettleman describes Black Bike Week as a \"unique event\" and \"an exhibitionist's paradise,\" while Mayor Irene Armstrong of Atlantic Beach, South Carolina, emphasizes it is \"all about riding, styling and profiling.\"",
        "eval": "same"
    },
    {
        "question": "are static shock and black lightning the same person",
        "answer": False,
        "Original passage": "It was recently revealed that prior to his abduction, Static teamed with Justice League member Black Lightning in order to stop former Blood Syndicate member Holocaust, who had tried to kill the superhero while he was acting as the keynote speaker at Ernest Hemingway High's senior graduation.",
        "Rephrased passage": "Before his kidnapping, Static collaborated with Black Lightning, a member of the Justice League, to thwart Holocaust, an ex-member of the Blood Syndicate, who attempted to murder the hero during his keynote address at Ernest Hemingway High's senior graduation.",
        "eval": "same"
    },
    {
        "question": "is there such a thing as unbreakable glass",
        "answer": True,
        "Original passage": "Unbreakable glass is glass, or glass substitute, which does not display the normal fragility of glass - in general the term is not used to refer to something that is absolutely unbreakable.",
        "Rephrased passage": "\"Unbreakable glass refers to a type of glass or glass alternative that resists the typical brittleness associated with regular glass, although the term generally does not imply complete indestructibility.\"",
        "eval": "same"
    },
    {
        "question": "can liquid latex be used as an adhesive",
        "answer": True,
        "Original passage": "As the latex dries it becomes very sticky and will stick to itself if accidentally folded over. Most manufacturers offer a slick spray for latex once it is dry to take away the stickiness allowing the movement of the model's limbs. Alternatively, shimmer powders can be dusted over dried liquid latex to create metallic effects. One advantage to the tackiness of liquid latex is that it can act as an adhesive to attach things to the paint, such as zippers.",
        "Rephrased passage": "Once dry, liquid latex becomes highly adhesive, causing it to stick to itself if inadvertently bent. To eliminate this stickiness and enable ease of movement for models, many manufacturers provide a slick spray for post-drying application. Additionally, shimmer powders can be applied to dry liquid latex to produce metallic finishes. A benefit of liquid latex's stickiness is its capability to serve as glue, allowing items like zippers to be affixed to the latex surface.",
        "eval": "same"
    },
    {
        "question": "is it illegal to carry a pistol in your car in texas",
        "answer": False,
        "Original passage": "Gov. Perry also signed H.B. 1815 after passage by the 2007 Legislature, a bill that allows any Texas resident to carry a handgun in the resident's motor vehicle without a CHL or other permit. The bill revised Chapter 46, Section 2 of the Penal Code to state that it is in fact not ``Unlawful Carry of a Weapon'', as defined by the statute, for a person to carry a handgun while in a motor vehicle they own or control, or to carry while heading directly from the person's home to that car. However, lawful carry while in a vehicle requires these four critical qualifiers: (1) the weapon must not be in plain sight (in Texas law, ``plain sight'' and ``concealed'' are mutually exclusive opposing terms); (2) the carrier cannot be involved in criminal activities, other than Class C traffic misdemeanors; (3) the carrier cannot be prohibited by state or federal law from possessing a firearm; and (4) the carrier cannot be a member of a criminal gang.",
        "Rephrased passage": "Governor Perry signed House Bill 1815 into law after its approval by the 2007 Legislature. This legislation permits any Texan to carry a handgun in their vehicle without needing a CHL or additional licensing. Chapter 46, Section 2 of the Penal Code was updated, clarifying that it is not considered \"Unlawful Carry of a Weapon\" for someone to have a handgun in a vehicle they own or control, or when traveling directly from their home to that vehicle. However, lawful gun possession in a vehicle is contingent upon four specific conditions: (1) the weapon must be concealed and not in plain view, as defined in Texas law where \"plain sight\" and \"concealed\" are distinctly different; (2) the individual should not partake in criminal activities, aside from Class C traffic misdemeanors; (3) the person must not be prohibited under state or federal law from owning a firearm; and (4) the individual must not be affiliated with a criminal gang.",
        "eval": "same"
    },
    {
        "question": "is it legal to carry a sgian dubh",
        "answer": True,
        "Original passage": "When worn as part of the national dress of Scotland, the sgian-dubh is legal in Scotland, England and Wales. In Scotland under the Criminal Law (Consolidation) (Scotland) Act 1995 Sec. 49, Sub-sec. 5(c); in England and Wales, under the Criminal Justice Act 1988 (section 139) and the Offensive Weapons Act 1996 (section 4).",
        "Rephrased passage": "When included as a component of traditional Scottish attire, carrying a sgian-dubh is permitted in Scotland, as well as in England and Wales. This legality is upheld by the Criminal Law (Consolidation) (Scotland) Act 1995, Section 49, Subsection 5(c) in Scotland, and by the Criminal Justice Act 1988 (section 139) alongside the Offensive Weapons Act 1996 (section 4) in England and Wales.",
        "eval": "same"
    },
    {
        "question": "several cranial nerves innervate structures of the tongue",
        "answer": False,
        "Original passage": "The hypoglossal nerve is the twelfth cranial nerve, and innervates all the extrinsic and intrinsic muscles of the tongue, except for the palatoglossus which is innervated by the vagus nerve. It is a nerve with a solely motor function. The nerve arises from the hypoglossal nucleus in the brain stem as a number of small rootlets, passes through the hypoglossal canal and down through the neck, and eventually passes up again over the tongue muscles it supplies into the tongue. There are two hypoglossal nerves in the body: one on the left, and one on the right.",
        "Rephrased passage": "The hypoglossal nerve, known as the twelfth cranial nerve, is responsible for innervating all the tongue's extrinsic and intrinsic muscles, except for the palatoglossus, which is controlled by the vagus nerve. This nerve serves purely a motor function. Originating from the hypoglossal nucleus in the brain stem, it emerges as several small rootlets, travels through the hypoglossal canal, descends through the neck, and then ascends over the tongue muscles it innervates to reach the tongue. The body contains two hypoglossal nerves, one on each side.",
        "eval": "same"
    },
    {
        "question": "is porto rico a part of the united states",
        "answer": True,
        "Original passage": "Puerto Rico (Spanish for ``Rich Port''), officially the Commonwealth of Puerto Rico (Spanish: Estado Libre Asociado de Puerto Rico, lit. ``Free Associated State of Puerto Rico'') and briefly called Porto Rico, is an unincorporated territory of the United States located in the northeast Caribbean Sea.",
        "Rephrased passage": "\"Puerto Rico, known in Spanish as 'Rich Port,' is formally designated as the Commonwealth of Puerto Rico (Spanish: Estado Libre Asociado de Puerto Rico, meaning 'Free Associated State of Puerto Rico') and is sometimes referred to as Porto Rico. It is an unincorporated territory belonging to the United States, situated in the northeast Caribbean Sea.\"",
        "eval": "same"
    },
    {
        "question": "is there sales tax in the state of washington",
        "answer": True,
        "Original passage": "Washington has a 6.50% statewide sales tax. Local rates vary based on an individual's location at the point of purchase and can total up to 3.10% for a combined rate of 9.60%. In addition, due to the large number of Native American sovereign nations located within the state, sales-tax rates, if any, can vary based on state treaties with each nation.",
        "Rephrased passage": "\"Washington imposes a statewide sales tax of 6.50%. Depending on where a purchase is made, local taxes can add up to an additional 3.10%, resulting in a maximum combined rate of 9.60%. Furthermore, the presence of numerous Native American sovereign nations within the state means that sales tax rates may differ according to treaties between the state and each nation.\"",
        "eval": "same"
    }
]


def few_shot(i, example):
    return example[i]

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

correct_answer = 0
total_answer = 0

for i in tqdm(range(len(test_set))):
    output_data = {}
    prompt = generate_prompt(test_set[i]["question"], test_set[i]["answer"], test_set[i]["Original passage"], test_set[i]["Rephrased passage"]) #
    response = client.chat.completions.create( # change
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
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
