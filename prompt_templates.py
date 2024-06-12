#################### DECOMPOSE PROMPT SECTION #############################
######################################################################
######################################################################
######################################################################

decompose_9shot_instruct = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, your goal is to decompose it down into simpler, standalone claims.
Each decomposed claim should:

1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable without reference to other claims, containing all necessary information for retrieval.
4. NOT ADD ANY INFORMATION, WHICH IS NOT PRESENT IN THE multi-hop claim, ONLY USE INFORMATION GIVEN IN THE multi-hop claim.
5. Be separated with a newline.
[/INST]
###
MULTI-HOP CLAIM: Ian Hunter from Mott the Hoople was born before the band member that released the album Dixie Chicken. That band member had the song 'Ride Like the Wind' dedicated to him.
DECOMPOSED CLAIMS: Ian Hunter was a member of Mott the Hoople.
Ian Hunter was born before the band member who released the album "Dixie Chicken."
The song "Ride Like the Wind" was dedicated to the band member who released the album "Dixie Chicken."

MULTI-HOP CLAIM: The defending champion of the 1979 Romika Cup - Singles and another tennis player are both tennis players. The other tennis player was defending champion at the 1994 Waldbaum's Hamlet Cup.
DECOMPOSED CLAIMS: The 1979 Romika Cup - Singles had a defending champion who is a tennis player.
The 1994 Waldbaum's Hamlet Cup had a defending champion who is a tennis player.

MULTI-HOP CLAIM: The movie '100 Million BC' continues the urban myth of an experiment allegedly carried out at the Philadelphia Naval Shipyard.
DECOMPOSED CLAIMS: The movie '100 Million BC' is based on or references an urban myth experiment.
An urban myth experiment was allegedly carried out at the Philadelphia Naval Shipyard

MULTI-HOP CLAIM: T the band Comeback Kid took their name form a headline about the current NHL team owner who played for the team that won the Stanley cup in 1992.
DECOMPOSED CLAIMS: The band Comeback Kid took their name from a headline.
The headline related to the name "Comeback Kid" involves an NHL team owner.
The NHL team owner associated with "Comeback Kid" was a former player for the NHL team that won the Stanley cup in 1992.

MULTI-HOP CLAIM: The TV show starring Will Smith was written by the creator of 'The Borowitz Report'. This show also starred Joseph Marcell.",
DECOMPOSED CLAIMS: The TV show starring Will Smith was written by the creator of 'The Borowitz Report'.
Joseph Marcell co-starred in the TV show with Will Smith.
Will Smith starred in a TV show.
    
MULTI-HOP CLAIM: The actor who played Colonel Sherman T. Potter in 'M*A*S*H' played Seldom Jackson in 'Kentucky Jones'.
DECOMPOSED CLAIMS: An actor played Colonel Sherman T. Potter in 'M*A*S*H'.
An actor played Seldom Jackson in 'Kentucky Jones.'

MULTI-HOP CLAIM: The producer of Future Profits is a male. So is Park Jimin's and Jae's co-host of After School Club.
DECOMPOSED CLAIMS: The producer of Future Profits is a male.
Park Jimin and Jae co-host After School Club.
The co-host of After School Club, alongside Park Jimin and Jae, is a male.

MULTI-HOP CLAIM: Joe Seiders from Taylor Locke and the Roughs was also in another band. This band and Kings of Leon are not both American rock bands.
DECOMPOSED CLAIMS: Joe Seiders was a member of Taylor Locke and the Roughs.
The Taylor Locke and the Roughs band member Joe Seiders was also a member of another band.
One of the bands Joe Seiders was a member of is not an American rock band.

MULTI-HOP CLAIM: Gerald B. Lefcourt was a featured personality in a documentary. The character portrayed in in that documerntary is a real-life informal adviser to the White House in cybersecurity.
DECOMPOSED CLAIMS: Gerald B. Lefcourt was featured in a documentary.
The documentary featuring Gerald B. Lefcourt portrays a character who is a real-life informal adviser to the White House in cybersecurity.
###

MULTI-HOP CLAIM: {claim}
DECOMPOSED CLAIMS:"""

decompose_9shot_noinstruct = """<s>
MULTI-HOP CLAIM: Ian Hunter from Mott the Hoople was born before the band member that released the album Dixie Chicken. That band member had the song 'Ride Like the Wind' dedicated to him.
DECOMPOSED CLAIMS: Ian Hunter was a member of Mott the Hoople.
Ian Hunter was born before the band member who released the album "Dixie Chicken."
The song "Ride Like the Wind" was dedicated to the band member who released the album "Dixie Chicken."

MULTI-HOP CLAIM: The defending champion of the 1979 Romika Cup - Singles and another tennis player are both tennis players. The other tennis player was defending champion at the 1994 Waldbaum's Hamlet Cup.
DECOMPOSED CLAIMS: The 1979 Romika Cup - Singles had a defending champion who is a tennis player.
The 1994 Waldbaum's Hamlet Cup had a defending champion who is a tennis player.

MULTI-HOP CLAIM: The movie '100 Million BC' continues the urban myth of an experiment allegedly carried out at the Philadelphia Naval Shipyard.
DECOMPOSED CLAIMS: The movie '100 Million BC' is based on or references an urban myth experiment.
An urban myth experiment was allegedly carried out at the Philadelphia Naval Shipyard

MULTI-HOP CLAIM: T the band Comeback Kid took their name form a headline about the current NHL team owner who played for the team that won the Stanley cup in 1992.
DECOMPOSED CLAIMS: The band Comeback Kid took their name from a headline.
The headline related to the name "Comeback Kid" involves an NHL team owner.
The NHL team owner associated with "Comeback Kid" was a former player for the NHL team that won the Stanley cup in 1992.

MULTI-HOP CLAIM: The TV show starring Will Smith was written by the creator of 'The Borowitz Report'. This show also starred Joseph Marcell.",
DECOMPOSED CLAIMS: The TV show starring Will Smith was written by the creator of 'The Borowitz Report'.
Joseph Marcell co-starred in the TV show with Will Smith.
Will Smith starred in a TV show.
    
MULTI-HOP CLAIM: The actor who played Colonel Sherman T. Potter in 'M*A*S*H' played Seldom Jackson in 'Kentucky Jones'.
DECOMPOSED CLAIMS: An actor played Colonel Sherman T. Potter in 'M*A*S*H'.
An actor played Seldom Jackson in 'Kentucky Jones.'

MULTI-HOP CLAIM: The producer of Future Profits is a male. So is Park Jimin's and Jae's co-host of After School Club.
DECOMPOSED CLAIMS: The producer of Future Profits is a male.
Park Jimin and Jae co-host After School Club.
The co-host of After School Club, alongside Park Jimin and Jae, is a male.

MULTI-HOP CLAIM: Joe Seiders from Taylor Locke and the Roughs was also in another band. This band and Kings of Leon are not both American rock bands.
DECOMPOSED CLAIMS: Joe Seiders was a member of Taylor Locke and the Roughs.
The Taylor Locke and the Roughs band member Joe Seiders was also a member of another band.
One of the bands Joe Seiders was a member of is not an American rock band.

MULTI-HOP CLAIM: Gerald B. Lefcourt was a featured personality in a documentary. The character portrayed in in that documerntary is a real-life informal adviser to the White House in cybersecurity.
DECOMPOSED CLAIMS: Gerald B. Lefcourt was featured in a documentary.
The documentary featuring Gerald B. Lefcourt portrays a character who is a real-life informal adviser to the White House in cybersecurity.

MULTI-HOP CLAIM: {claim}
DECOMPOSED CLAIMS:"""

decompose_add_context = """<s> [INST] Given a multi-hop claim and context information comprised of ten title and sentece combinations, your goal is to decompose the claim down into simpler, standalone claims by making use of the context.
Some claims are wrong, if there is information in the context that can be used to correct the claim, please do so.
Each decomposed claim should:
1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable without reference to other claims, containing all necessary information for retrieval.
4. Incorporate Key Entities from Context: If the multi-hop claim refers to entities in a generic or indirect manner (e.g., "the guy that worked with Robert Richards"), and the context provides specific names or details (e.g., the guy is named John Doe), include these specific entities in your decomposed claims.
5. NOT ADD OR MAKE UP ANY INFORMATION, WHICH IS NOT PRESENT IN THE given multi-hop claims and context.
6. Be separated with a newline. 
[/INST]
###
CONTEXT: The Second Coming (Adina Howard album) The Second Coming is the second studio album by American singer Adina Howard.\nScott Stapp Anthony Scott Flippen (born August 8, 1973), also known as Scott Stapp, is an American singer, songwriter, and musician, known as the lead vocalist and lyricist of rock bands Creed and Art of Anarchy.\nKevin McMahon (musician) Kevin Michael McMahon (born September 21, 1953) is a musician, singer, and songwriter for the long-standing bands Lucky Pierre and Prick.\nMina Caputo Mina Caputo (born Keith Caputo on December 4, 1973) is an American singer best known as the lead singer and a founding member of the New York heavy metal band Life of Agony.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nCorey Taylor Corey Todd Taylor (born December 8, 1973) is an American musician, singer, songwriter, actor, and author, best known as the lead singer and lyricist of the American heavy metal band Slipknot and the American alternative metal band Stone Sour.\nAmerican Caesar (album) American Caesar is the eleventh studio album by American rock singer Iggy Pop, released in September 1993 by record label Virgin.\nMark Oliver Everett Mark Oliver Everett (born April 10, 1963) is the American lead singer, songwriter, guitarist, keyboardist and sometime drummer of the rock band Eels.\nMichael Gira Michael Rolfe Gira ( ; born February 19, 1954) is an American singer-songwriter, musician, author and artist.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
DECOMPOSED CLAIMS: The play Almost, Maine, created by an award-winning American actor and an accomplished playwright, broke box office records and garnered critical acclaim at its premiere.\nAlmost, Maine was performed by the Portland Stage Company.\n

CONTEXT: 'The Second Coming (Adina Howard album) The Second Coming is the second studio album by American singer Adina Howard.\nScott Stapp Anthony Scott Flippen (born August 8, 1973), also known as Scott Stapp, is an American singer, songwriter, and musician, known as the lead vocalist and lyricist of rock bands Creed and Art of Anarchy.\nKevin McMahon (musician) Kevin Michael McMahon (born September 21, 1953) is a musician, singer, and songwriter for the long-standing bands Lucky Pierre and Prick.\nMina Caputo Mina Caputo (born Keith Caputo on December 4, 1973) is an American singer best known as the lead singer and a founding member of the New York heavy metal band Life of Agony.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nCorey Taylor Corey Todd Taylor (born December 8, 1973) is an American musician, singer, songwriter, actor, and author, best known as the lead singer and lyricist of the American heavy metal band Slipknot and the American alternative metal band Stone Sour.\nAmerican Caesar (album) American Caesar is the eleventh studio album by American rock singer Iggy Pop, released in September 1993 by record label Virgin.\nMark Oliver Everett Mark Oliver Everett (born April 10, 1963) is the American lead singer, songwriter, guitarist, keyboardist and sometime drummer of the rock band Eels.\nMichael Gira Michael Rolfe Gira ( ; born February 19, 1954) is an American singer-songwriter, musician, author and artist.\nAlice Cooper Alice Cooper (born Vincent Damon Furnier on February 4, 1948) is an American singer, songwriter, and actor whose career spans over five decades.\nMeat Loaf Michael Lee Aday (born Marvin Lee Aday; September 27, 1947), better known by his stage name Meat Loaf, is an American musician, singer, songwriter, record producer, and actor.\nIggy Pop James Newell Osterberg Jr. (born April 21, 1947), known professionally by his stage name Iggy Pop ( ), and designated the "Godfather of Punk," is an American singer, songwriter, musician, producer and actor.\nJimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTrent Reznor Michael Trent Reznor (born May 17, 1965) is an American singer, songwriter, musician, record producer, and film score composer.\n
CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
DECOMPOSED CLAIMS: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\n

CONTEXT: Mary and the Witch's Flower Mary and the Witch's Flower (Japanese: メアリと魔女の花 , Hepburn: Meari to Majo no Hana ) is a 2017 Japanese anime fantasy film directed by Hiromasa Yonebayashi and produced by Studio Ponoc, based on "The Little Broomstick" by Mary Stewart.\nGamblers' Ballet "Cardinal Knowledge" was used as the outro music for the Cartoon Saloon, Oscar nominated animation "The Secret of Kells", and "Dúisigí" and "Cabhraigí Léi" were used in the Japanese film "Kadokawa", which was directed by Ryuichi Hiroki.\nTomm Moore His first two feature films, "The Secret of Kells" (2009) and "Song of the Sea" (2014), have received critical acclaim and were both nominated for the Academy Award for Best Animated Feature.\nThe Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\nArrietty Arrietty, titled The Borrower Arrietty (Japanese: 借りぐらしのアリエッティ , Hepburn: Kari-gurashi no Arietti ) in Japan and The Secret World of Arrietty in North America, is a 2010 Japanese animated fantasy film produced by Studio Ghibli and directed by Hiromasa Yonebayashi as his feature film debut as a director.\nFrom Up on Poppy Hill From Up on Poppy Hill (Japanese: , Hepburn: Kokuriko-zaka Kara , "From Coquelicot Hill") is a 2011 Japanese animated drama film directed by Gorō Miyazaki, scripted by Hayao Miyazaki and Keiko Niwa, and produced by Studio Ghibli.\nKunio Katō Kunio Katō (加藤久仁生 ) (born 1977) is a Japanese animator known for the short film "La Maison en Petits Cubes" which won the 2008 Academy Award for Best Animated Short Film.\nYukihiro Miyamoto Yukihiro Miyamoto (宮本 幸裕 , Miyamoto Yukihiro , born November 21, 1977) , is a Japanese animator and director best known for directing the anime film "", which was a box-office success in Japan, with a total gross of more than ¥2 billion.\nCartoon Saloon On February 2, 2010 "The Secret of Kells" was nominated in the category of best animated film at the 82nd Academy Awards.\nMai Mai Miracle Mai Mai Shinko and the Millennium-Old Magic) is a Japanese animated film based on Nobuko Takagi's novelization of her autobiography, "Maimai Shinko".
CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
DECOMPOSED CLAIMS: The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film.\nMike McFarland voiced in the 2009 stop-motion film Summer Wars from Japan.
###
CONTEXT: {context}
CLAIM: {claim}
DECOMPOSED CLAIMS:"""

decompose_add_context_noinstruct = """<s>
###
CONTEXT: The Second Coming (Adina Howard album) The Second Coming is the second studio album by American singer Adina Howard.\nScott Stapp Anthony Scott Flippen (born August 8, 1973), also known as Scott Stapp, is an American singer, songwriter, and musician, known as the lead vocalist and lyricist of rock bands Creed and Art of Anarchy.\nKevin McMahon (musician) Kevin Michael McMahon (born September 21, 1953) is a musician, singer, and songwriter for the long-standing bands Lucky Pierre and Prick.\nMina Caputo Mina Caputo (born Keith Caputo on December 4, 1973) is an American singer best known as the lead singer and a founding member of the New York heavy metal band Life of Agony.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nCorey Taylor Corey Todd Taylor (born December 8, 1973) is an American musician, singer, songwriter, actor, and author, best known as the lead singer and lyricist of the American heavy metal band Slipknot and the American alternative metal band Stone Sour.\nAmerican Caesar (album) American Caesar is the eleventh studio album by American rock singer Iggy Pop, released in September 1993 by record label Virgin.\nMark Oliver Everett Mark Oliver Everett (born April 10, 1963) is the American lead singer, songwriter, guitarist, keyboardist and sometime drummer of the rock band Eels.\nMichael Gira Michael Rolfe Gira ( ; born February 19, 1954) is an American singer-songwriter, musician, author and artist.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
DECOMPOSED CLAIMS: The play Almost, Maine, created by an award-winning American actor and an accomplished playwright, broke box office records and garnered critical acclaim at its premiere.\nAlmost, Maine was performed by the Portland Stage Company.\n

CONTEXT: 'The Second Coming (Adina Howard album) The Second Coming is the second studio album by American singer Adina Howard.\nScott Stapp Anthony Scott Flippen (born August 8, 1973), also known as Scott Stapp, is an American singer, songwriter, and musician, known as the lead vocalist and lyricist of rock bands Creed and Art of Anarchy.\nKevin McMahon (musician) Kevin Michael McMahon (born September 21, 1953) is a musician, singer, and songwriter for the long-standing bands Lucky Pierre and Prick.\nMina Caputo Mina Caputo (born Keith Caputo on December 4, 1973) is an American singer best known as the lead singer and a founding member of the New York heavy metal band Life of Agony.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nCorey Taylor Corey Todd Taylor (born December 8, 1973) is an American musician, singer, songwriter, actor, and author, best known as the lead singer and lyricist of the American heavy metal band Slipknot and the American alternative metal band Stone Sour.\nAmerican Caesar (album) American Caesar is the eleventh studio album by American rock singer Iggy Pop, released in September 1993 by record label Virgin.\nMark Oliver Everett Mark Oliver Everett (born April 10, 1963) is the American lead singer, songwriter, guitarist, keyboardist and sometime drummer of the rock band Eels.\nMichael Gira Michael Rolfe Gira ( ; born February 19, 1954) is an American singer-songwriter, musician, author and artist.\nAlice Cooper Alice Cooper (born Vincent Damon Furnier on February 4, 1948) is an American singer, songwriter, and actor whose career spans over five decades.\nMeat Loaf Michael Lee Aday (born Marvin Lee Aday; September 27, 1947), better known by his stage name Meat Loaf, is an American musician, singer, songwriter, record producer, and actor.\nIggy Pop James Newell Osterberg Jr. (born April 21, 1947), known professionally by his stage name Iggy Pop ( ), and designated the "Godfather of Punk," is an American singer, songwriter, musician, producer and actor.\nJimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTrent Reznor Michael Trent Reznor (born May 17, 1965) is an American singer, songwriter, musician, record producer, and film score composer.\n
CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
DECOMPOSED CLAIMS: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\n

CONTEXT: Mary and the Witch's Flower Mary and the Witch's Flower (Japanese: メアリと魔女の花 , Hepburn: Meari to Majo no Hana ) is a 2017 Japanese anime fantasy film directed by Hiromasa Yonebayashi and produced by Studio Ponoc, based on "The Little Broomstick" by Mary Stewart.\nGamblers' Ballet "Cardinal Knowledge" was used as the outro music for the Cartoon Saloon, Oscar nominated animation "The Secret of Kells", and "Dúisigí" and "Cabhraigí Léi" were used in the Japanese film "Kadokawa", which was directed by Ryuichi Hiroki.\nTomm Moore His first two feature films, "The Secret of Kells" (2009) and "Song of the Sea" (2014), have received critical acclaim and were both nominated for the Academy Award for Best Animated Feature.\nThe Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\nArrietty Arrietty, titled The Borrower Arrietty (Japanese: 借りぐらしのアリエッティ , Hepburn: Kari-gurashi no Arietti ) in Japan and The Secret World of Arrietty in North America, is a 2010 Japanese animated fantasy film produced by Studio Ghibli and directed by Hiromasa Yonebayashi as his feature film debut as a director.\nFrom Up on Poppy Hill From Up on Poppy Hill (Japanese: , Hepburn: Kokuriko-zaka Kara , "From Coquelicot Hill") is a 2011 Japanese animated drama film directed by Gorō Miyazaki, scripted by Hayao Miyazaki and Keiko Niwa, and produced by Studio Ghibli.\nKunio Katō Kunio Katō (加藤久仁生 ) (born 1977) is a Japanese animator known for the short film "La Maison en Petits Cubes" which won the 2008 Academy Award for Best Animated Short Film.\nYukihiro Miyamoto Yukihiro Miyamoto (宮本 幸裕 , Miyamoto Yukihiro , born November 21, 1977) , is a Japanese animator and director best known for directing the anime film "", which was a box-office success in Japan, with a total gross of more than ¥2 billion.\nCartoon Saloon On February 2, 2010 "The Secret of Kells" was nominated in the category of best animated film at the 82nd Academy Awards.\nMai Mai Miracle Mai Mai Shinko and the Millennium-Old Magic) is a Japanese animated film based on Nobuko Takagi's novelization of her autobiography, "Maimai Shinko".
CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
DECOMPOSED CLAIMS: The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film.\nMike McFarland voiced in the 2009 stop-motion film Summer Wars from Japan.
###
CONTEXT: {context}
CLAIM: {claim}
DECOMPOSED CLAIMS:"""

add_context_prompt = """<s> [INST] Given a multi-hop claim and context information comprising sentences and titles, your task is to enhance the claim by embedding pertinent details from the context. The goal is to add depth to the claim without altering its original meaning, ensuring it becomes more informative yet remains concise.
Each enhanced claim should:
1. Be Precise and Contextually Enriched: Infuse the claim with key details from the context, such as names, dates, and locations, to make it more informative but avoid overloading with unnecessary information.
2. Maintain Accuracy and Coherence: Ensure the enhanced claim is factually accurate and logically consistent with the context provided, correcting any discrepancies.
3. Preserve Original Meaning: The enhancement should not deviate from the original claim's intent, maintaining the core message while expanding on its specifics.
4. Ensure Clarity and Brevity: The claim should be succinct, avoiding verbose or complex structures, making it easily understandable and retrievable in searches.
5. NOT ADD OR MAKE UP ANY INFORMATION, WHICH IS NOT PRESENT IN THE given multi-hop claim and context.
[/INST]
###
CONTEXT: 'Almost, Maine It premiered at the Portland Stage Company in Portland, Maine in 2004 where it broke box office records and garnered critical acclaim.\nThe Gin Game The Gin Game is a two-person, two-act play by Donald L. Coburn that premiered at American Theater Arts in Hollywood in September 1976, directed by Kip Niven.\nThe Patriots (play) The Patriots is an award-winning play written in a prologue and three acts by Sidney Kingsley in 1943.\nThe Last Days of Judas Iscariot The Last Days of Judas Iscariot is a play by American playwright Stephen Adly Guirgis first staged Off-Broadway at The Public Theater on March 2, 2005 directed by Philip Seymour Hoffman.\nTopdog/Underdog Topdog/Underdog is a play by American playwright Suzan-Lori Parks which premiered in 2001 off-Broadway in New York City.\nCollected Stories (play) Collected Stories is a play by Donald Margulies which premiered at South Coast Repertory in 1996, and was presented on Broadway in 2010.\nThe Iceman Cometh First published in 1946, the play premiered on Broadway at the Martin Beck Theatre on October 9, 1946, directed by Eddie Dowling, where it ran for 136 performances before closing on March 15, 1947.\nJason Wells (playwright) Jason Wells (born 1960) is an American actor and award-winning playwright.\nHal Holbrook He first received critical acclaim in 1954 for a one-man stage show he developed while studying at Denison University, performing as Mark Twain.\nWonder of the World (play) Wonder of the World is a play by American playwright David Lindsay-Abaire.'
CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
ENHANCED CLAIM: The play 'Almost, Maine,' premiered by the Portland Stage Company in 2004, was a collaboration between an award-winning American actor and an accomplished playwright, breaking box office records and garnering critical acclaim.

CONTEXT: Jimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nOzzy Osbourne John Michael "Ozzy" Osbourne (born 3 December 1948) is an English singer, songwriter, and actor.\nYou Gotta Sin to Get Saved You Gotta Sin to Get Saved is the second album by American singer-songwriter Maria McKee, released in 1993 (see 1993 in music).\nScott Stapp Anthony Scott Flippen (born August 8, 1973), also known as Scott Stapp, is an American singer, songwriter, and musician, known as the lead vocalist and lyricist of rock bands Creed and Art of Anarchy.\nBret Michaels Bret Michael Sychak (born March 15, 1963), professionally known as Bret Michaels, is an American singer-songwriter and musician.\nMichael Gira Michael Rolfe Gira ( ; born February 19, 1954) is an American singer-songwriter, musician, author and artist.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\nMarilyn Manson Brian Hugh Warner (born January 5, 1969), known professionally as Marilyn Manson, is an American singer, songwriter, musician, composer, actor, painter, author and former music journalist.\nCorey Taylor Corey Todd Taylor (born December 8, 1973) is an American musician, singer, songwriter, actor, and author, best known as the lead singer and lyricist of the American heavy metal band Slipknot and the American alternative metal band Stone Sour.
CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
ENHANCED CLAIM: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters. Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CONTEXT: 'Gamblers' Ballet "Cardinal Knowledge" was used as the outro music for the Cartoon Saloon, Oscar nominated animation "The Secret of Kells", and "Dúisigí" and "Cabhraigí Léi" were used in the Japanese film "Kadokawa", which was directed by Ryuichi Hiroki.\nFabrice Ziolkowski Fabrice Ziolkowski (born January 28, 1954) is a French-American screenwriter, director, producer, and voice director, best known for scripting the Oscar-nominated feature animation film "The Secret of Kells", writing the animated television series "Gawayn", and directing and producing the avant-garde documentary film "L.A.X.".\nThe Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\nArrietty Arrietty, titled The Borrower Arrietty (Japanese: 借りぐらしのアリエッティ , Hepburn: Kari-gurashi no Arietti ) in Japan and The Secret World of Arrietty in North America, is a 2010 Japanese animated fantasy film produced by Studio Ghibli and directed by Hiromasa Yonebayashi as his feature film debut as a director.\nMadame Tutli-Putli Madame Tutli-Putli is a 2007 stop motion-animated short film by Montreal filmmakers Chris Lavis and Maciek Szczerbowski, collectively known as Clyde Henry Productions, and produced by the National Film Board of Canada (NFB).\nSteamboy Steamboy (Japanese: スチームボーイ , Hepburn: Suchīmubōi ) is a 2004 Japanese steampunk animated action film produced by Sunrise, directed and co-written by Katsuhiro Otomo, his second major anime release, following "Akira".\nCartoon Saloon The company has also developed the animated film "The Secret of Kells".\nNasu: Summer in Andalusia Nasu: Summer in Andalusia (茄子 アンダルシアの夏 , Nasu: Andarushia no Natsu ) is a 2003 Japanese anime film by Madhouse, directed by Kitarō Kōsaka, the famed animation supervisor of the Oscar-winning anime film "Spirited Away" and "Princess Mononoke" and long-time collaborator of Studio Ghibli, and adapted from a short 3-tankōbon manga by Iō Kuroda, entitled "Nasu", which was serialized in the Afternoon manga magazine.\nEleanor's Secret Eleanor's Secret (original French title Kérity, la maison des contes) is a 2009 Franco-Italian animated feature film directed by Dominique Monféry.\n 'Summer Wars Summer Wars (Japanese: サマーウォーズ , Hepburn: Samā Wōzu ) is a 2009 Japanese animated science fiction film directed by Mamoru Hosoda, animated by Madhouse and distributed by Warner Bros.
CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
ENHANCED CLAIM: Summer Wars,' directed by Mamoru Hosoda and animated by Madhouse, is a 2009 Japanese animated science fiction film. 'The Secret of Kells,' a French-Belgian-Irish animated fantasy film, was produced by Cartoon Saloon and premiered on 8 February 2009. The 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
###
CONTEXT: {context}
CLAIM: {claim}
ENHANCED CLAIM:"""

decompose_entity_based = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, decompose it into simpler, standalone claims that adhere to the following criteria:
1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight the main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable on its own, containing all necessary information for retrieval.
4. Avoid Adding Information: Do not include any information that is not present in the original multi-hop claim. Rely solely on the provided content.
5. Manage Redundancy: If multiple potential decomposed claims involve the same entity or fact, consolidate them into a single claim unless different aspects are critically relevant for separate retrieval paths.
6. Separate Clearly: Do not use listing and separate each decomposed claim with a newline.
[/INST]
###
MULTI-HOP CLAIM: The Lusty Men is a western film directed by Nicholas Ray. He and Elia Kazan, the writer of America America, are both actors and directors.
DECOMPOSED CLAIMS: The Lusty Men is a western film directed by Nicholas Ray.
Nicholas Ray is both an actor and a director.
Elia Kazan wrote the film America America.
Elia Kazan is both an actor and a director.

MULTI-HOP CLAIM: Sia heavily sampled the song 'Montagues and Capulets' by Sergei O.Prokofieff's grandfather, a Russian composer, for her song 'Taken for Granted.' The song is part of Prokofiev's ballet 'Romeo and Juliet,' which was first performed in 1940. The song 'Party Like a Russian' by Robbie Williams also samples 'Montagues and Capulets.'
DECOMPOSED CLAIMS: Sia heavily sampled the song 'Montagues and Capulets' for her song 'Taken for Granted.'
'Montagues and Capulets' was composed by Sergei O. Prokofieff's grandfather, a Russian composer.
The song 'Montagues and Capulets' is also sampled in Robbie Williams' song 'Party Like a Russian.

MULTI-HOP CLAIM: The musician that signed Godhead (band) to their label, Marilyn Manson, and Joe Lynn Turner are both American singer, songwriters. Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
DECOMPOSED CLAIMS: Marilyn Manson signed the band Godhead to their label.
Marilyn Manson is an American singer and songwriter.
Joe Lynn Turner is an American singer and songwriter.
'Second Hand Life' is the tenth solo studio album by Joe Lynn Turner.

MULTI-HOP CLAIM: The Melville suburb in southern Hamilton, New Zealand contains streets named after war heroes including Douglas Melville Charles Guest. Douglas Melville Charles Guest, also known as Douglas Bader, was the subject of the 1956 British biographical film 'Reach for the Sky'. He was a Royal Air Force flying ace during the Second World War.
DECOMPOSED CLAIMS: The Melville suburb in southern Hamilton, New Zealand, features streets named after war heroes.
Douglas Melville Charles Guest, also known as Douglas Bader, was a Royal Air Force flying ace during the Second World War.
Douglas Bader was the subject of the 1956 British biographical film 'Reach for the Sky'.
One of the streets in Melville suburb is named after Douglas Melville Charles Guest.

MULTI-HOP CLAIM: Channel 8, which has Ong-Art Singlumpong for an executive director, has broadcast the American sports HBO World Championship Boxing that premiered in the early 1970s.
DECOMPOSED CLAIMS: Channel 8 is a television channel that has Ong-Art Singlumpong as an executive director.
Channel 8 has broadcast the American sports program HBO World Championship Boxing.
HBO World Championship Boxing premiered in the early 1970s.

MULTI-HOP CLAIM: The Landsman Kill, a body of water, is near the Dutchess County Fairgrounds and several historical landmarks on New York State Route 308, which intersects State Route 199 at Rock City in the state where billionaire Robert F. Kennedy was a junior senator.
DECOMPOSED CLAIMS: The Landsman Kill is a body of water near the Dutchess County Fairgrounds.
The Dutchess County Fairgrounds and several historical landmarks are located on New York State Route 308.
New York State Route 308 intersects with State Route 199 at Rock City.
Robert F. Kennedy, a billionaire, was a junior senator in New York state.

MULTI-HOP CLAIM: Kelly Preston, the actress that starred as a parent in Sky High (2005 film), is married to John Travolta, one of the subjects of film-critic Nigel Andrews' books. She also starred in Cheyenne Warrior, a 1994 American film about a pregnant woman who is stranded at a trading post during the American Civil War.
DECOMPOSED CLAIMS: Kelly Preston starred as a parent in the film Sky High (2005).
John Travolta is one of the subjects of film-critic Nigel Andrews' books.
Kelly Preston starred in Cheyenne Warrior, a 1994 American film about a pregnant woman stranded at a trading post during the American Civil War.

MULTI-HOP CLAIM: The founder of the International Grenfell Association, Sir Wilfred Thomason Grenfell, was a medical missionary to Newfoundland and Labrador. The city of Corner Brook, serviced by the Corner Brook Transit, is located in the province of Newfoundland and Labrador, Canada. The International Grenfell Association, founded by Grenfell, is headquartered in Mary's Harbour, which is also in Newfoundland and Labrador.
DECOMPOSED CLAIMS: Sir Wilfred Thomason Grenfell was the founder of the International Grenfell Association.
Corner Brook is serviced by the Corner Brook Transit.
The International Grenfell Association is headquartered in Mary's Harbour.

MULTI-HOP CLAIM: The 2016 romantic drama "Me Before You," directed by Thea Sharrock, features Sam Claflin as Will Traynor. Sam Claflin is known for his roles as Finnick Odair in "The Hunger Games" film series, Philip Swift in "Pirates of the Caribbean: On Stranger Tides," and Woody Harrelson's co-star in "The Lost Future."
DECOMPOSED CLAIMS: The 2016 romantic drama "Me Before You," is directed by Thea Sharrock.
"Me Before You" features Sam Claflin as Will Traynor.
Sam Claflin co-starred with Woody Harrelson in "The Lost Future."

###
MULTI-HOP CLAIM: {claim}
DECOMPOSED CLAIMS:"""

decompose_without_redundancy = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, decompose it into simpler, standalone claims that adhere to the following criteria:
1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight the main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable on its own, containing all necessary information for retrieval.
4. Avoid Adding Information: Do not include any information that is not present in the original multi-hop claim. Rely solely on the provided content.
5. Manage Redundancy: If multiple potential decomposed claims involve the same entity or fact, consolidate them into a single claim unless different aspects are critically relevant for separate retrieval paths.
6. Separate Clearly: Do not use listing and separate each decomposed claim with a newline.
[/INST]
###
MULTI-HOP CLAIM: The Lusty Men is a western film directed by Nicholas Ray. He and Elia Kazan, the writer of America America, are both actors and directors.
DECOMPOSED CLAIMS: The Lusty Men is a western film directed by Nicholas Ray who is both an actor and a director.
Elia Kazan is both an actor and a director and wrote the film America America.

MULTI-HOP CLAIM: Sia heavily sampled the song 'Montagues and Capulets' by Sergei O.Prokofieff's grandfather, a Russian composer, for her song 'Taken for Granted.' The song is part of Prokofiev's ballet 'Romeo and Juliet,' which was first performed in 1940. The song 'Party Like a Russian' by Robbie Williams also samples 'Montagues and Capulets.'
DECOMPOSED CLAIMS: Sia heavily sampled the song 'Montagues and Capulets' for her song 'Taken for Granted.'
'Montagues and Capulets' was composed by Sergei O. Prokofieff's grandfather, a Russian composer.
The song 'Montagues and Capulets' is also sampled in Robbie Williams' song 'Party Like a Russian.

MULTI-HOP CLAIM: The musician that signed Godhead (band) to their label, Marilyn Manson, and Joe Lynn Turner are both American singer, songwriters. Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
DECOMPOSED CLAIMS: Marilyn Manson who is an American singer and songwriter, signed the band Godhead to their label.
'Second Hand Life' is the tenth solo studio album by Joe Lynn Turner who is an American singer and songwriter.

MULTI-HOP CLAIM: The Melville suburb in southern Hamilton, New Zealand contains streets named after war heroes including Douglas Melville Charles Guest. Douglas Melville Charles Guest, also known as Douglas Bader, was the subject of the 1956 British biographical film 'Reach for the Sky'. He was a Royal Air Force flying ace during the Second World War.
DECOMPOSED CLAIMS: The Melville suburb in southern Hamilton, New Zealand, features streets named after war heroes, including one named after Douglas Melville Charles Guest, also known as Douglas Bader.
Douglas Bader, a Royal Air Force flying ace during the Second World War, was the subject of the 1956 British biographical film 'Reach for the Sky'.

MULTI-HOP CLAIM: Channel 8, which has Ong-Art Singlumpong for an executive director, has broadcast the American sports HBO World Championship Boxing that premiered in the early 1970s.
DECOMPOSED CLAIMS: Channel 8 is a television channel that has Ong-Art Singlumpong as an executive director.
Channel 8 has broadcast the American sports program HBO World Championship Boxing which premiered in the early 1970s.

MULTI-HOP CLAIM: The Landsman Kill, a body of water, is near the Dutchess County Fairgrounds and several historical landmarks on New York State Route 308, which intersects State Route 199 at Rock City in the state where billionaire Robert F. Kennedy was a junior senator.
DECOMPOSED CLAIMS: The Landsman Kill, a body of water, is located near the Dutchess County Fairgrounds along New York State Route 308.
New York State Route 308, home to several historical landmarks, intersects with State Route 199 at Rock City.
Robert F. Kennedy, a billionaire and former junior senator, served in New York state.

MULTI-HOP CLAIM: Kelly Preston, the actress that starred as a parent in Sky High (2005 film), is married to John Travolta, one of the subjects of film-critic Nigel Andrews' books. She also starred in Cheyenne Warrior, a 1994 American film about a pregnant woman who is stranded at a trading post during the American Civil War.
DECOMPOSED CLAIMS: Kelly Preston starred as a parent in the film Sky High (2005) and in Cheyenne Warrior, a 1994 American film about a pregnant woman stranded at a trading post during the American Civil War.
John Travolta, Kelly Preston's husband, is one of the subjects of film-critic Nigel Andrews' books.

MULTI-HOP CLAIM: The founder of the International Grenfell Association, Sir Wilfred Thomason Grenfell, was a medical missionary to Newfoundland and Labrador. The city of Corner Brook, serviced by the Corner Brook Transit, is located in the province of Newfoundland and Labrador, Canada. The International Grenfell Association, founded by Grenfell, is headquartered in Mary's Harbour, which is also in Newfoundland and Labrador.
DECOMPOSED CLAIMS: Sir Wilfred Thomason Grenfell was the founder of the International Grenfell Association, which is headquartered in Mary's Harbour.
Corner Brook is serviced by the Corner Brook Transit.

MULTI-HOP CLAIM: The 2016 romantic drama "Me Before You," directed by Thea Sharrock, features Sam Claflin as Will Traynor. Sam Claflin is known for his roles as Finnick Odair in "The Hunger Games" film series, Philip Swift in "Pirates of the Caribbean: On Stranger Tides," and Woody Harrelson's co-star in "The Lost Future."
DECOMPOSED CLAIMS: The 2016 romantic drama "Me Before You," is directed by Thea Sharrock.
"Me Before You" features Sam Claflin who co-starred with Woody Harrelson in "The Lost Future."

###
MULTI-HOP CLAIM: {claim}
DECOMPOSED CLAIMS:"""

decompose_atomar = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, please transform it into atomic claims. Each atomic claim should adhere to the following guidelines:
1. Single Entity Focus: Each atomic claim must contain information pertaining to only one entity. This ensures clarity and precision in data retrieval from a vector database, where each data point (or vector) represents a single, distinct entity or concept.
2. Avoid Relationships: Do not include relationships or connections between entities in the atomic claims. This maintains the independence of data points for more straightforward and efficient retrieval.
3. Non-Redundant: Ensure that each entity is mentioned only once across all claims. This avoids duplication and maintains database efficiency.
4. Separate Clearly: Do not use listing and separate each decomposed claim with a newline.
[/INST]
###
MULTI-HOP CLAIM: The Lusty Men is a western film directed by Nicholas Ray. He and Elia Kazan, the writer of America America, are both actors and directors.
ATOMIC CLAIMS: The Lusty Men is a western film.
Nicholas Ray is a director and actor.
Elia Kazan is a writer, director, and actor.
America America is a film.

MULTI-HOP CLAIM: Kelly Preston, the actress that starred as a parent in Sky High (2005 film), is married to John Travolta, one of the subjects of film-critic Nigel Andrews' books. She also starred in Cheyenne Warrior, a 1994 American film about a pregnant woman who is stranded at a trading post during the American Civil War.
ATOMIC CLAIMS: Kelly Preston is an actress.
Sky High is a 2005 film.
Cheyenne Warrior is a 1994 American film.
Nigel Andrews wrote a book.
John Travolta is a subject of a book.

MULTI-HOP CLAIM: Sia heavily sampled the song 'Montagues and Capulets' by Sergei O. Prokofieff's grandfather, a Russian composer, for her song 'Taken for Granted.' The song is part of Prokofiev's ballet 'Romeo and Juliet,' which was first performed in 1940. The song 'Party Like a Russian' by Robbie Williams also samples 'Montagues and Capulets.'
ATOMIC CLAIMS: Sia is an artist.
'Taken for Granted' is a song.
Sergei O. Prokofiev's grandfather is a Russian composer.
'Montagues and Capulets' is part of a ballet.
'Romeo and Juliet' premiered in 1940.
Robbie Williams is an artist.
'Party Like a Russian' is a song.

MULTI-HOP CLAIM: The 2016 romantic drama "Me Before You," directed by Thea Sharrock, features Sam Claflin as Will Traynor. Sam Claflin is known for his roles as Finnick Odair in "The Hunger Games" film series, Philip Swift in "Pirates of the Caribbean: On Stranger Tides," and Woody Harrelson's co-star in "The Lost Future."
ATOMIC CLAIMS: "Me Before You" is a 2016 romantic drama.
Thea Sharrock is a director.
Sam Claflin is an actor.
Woody Harrelson is an actor.
"The Lost Future" is a film.

MULTI-HOP CLAIM: Channel 8, which has Ong-Art Singlumpong for an executive director, has broadcast the American sports HBO World Championship Boxing that premiered in the early 1970s.
ATOMIC CLAIMS: Channel 8 is a television channel.
Ong-Art Singlumpong is an executive director.
HBO World Championship Boxing is an American sports program.

MULTI-HOP CLAIM: The musician that signed Godhead (band) to their label, Marilyn Manson, and Joe Lynn Turner are both American singer, songwriters. Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
ATOMIC CLAIMS: Marilyn Manson is a musician.
Godhead is a band.
Joe Lynn Turner is an American singer and songwriter.
"Second Hand Life" is a tenth solo studio album.

MULTI-HOP CLAIM: Neither the vocalist of The Halo Method, Lukas Rossi, nor Terry Kath are New Yorkers.
ATOMIC CLAIMS: Lukas Rossi is a vocalist.
Terry Kath is not a New Yorker.
The Halo Method is a band.

MULTI-HOP CLAIM: The film "Ghost Team" is based on The Quiet Family, a 1998 Korean comedy horror film directed by Kim Jee-woon. Oviya, an Indian film actress, is part of the ensemble cast in this horror comedy.
ATOMIC CLAIMS: "Ghost Team" is a film.
"The Quiet Family" is a 1998 Korean comedy horror film.
Kim Jee-woon is a director.
Oviya is an Indian film actress.

MULTI-HOP CLAIM: Euptelea and Muehlenbeckia are both genuses.
ATOMIC CLAIMS: Euptelea is a genus.
Muehlenbeckia is a genus.

###
MULTI-HOP CLAIM: {claim}
ATOMIC CLAIMS:
"""


decision_prompt_9shot = """<s> [INST] You are given a base claim and an refined claim. Your task is to determine whether the refined claim explicitly mentions all entities from the base claim. Provide a simple True or False evaluation based on whether the refined claim explicitly covers all entities from the base claim.

Guidelines for Evaluation:
1. Explicit Entity Mention: Check that every entity mentioned in the base claim is explicitly named in the refined claim.
2. Clear Decision: Provide a 'True' if all entities are explicitly mentioned, or 'False' if any are missing.
[/INST]
###
Base Claim: The language spoken by some residents of the commune that Tamellaht is part of is traditionally written with the ancient Libyco-Berber script. It is closely related to the Tumzabt and Teggargrent languages.
Refined Claim: The Tugurt language, spoken by some residents of the commune Tamacine that Tamellaht is part of, is traditionally written with the ancient Libyco-Berber script, which now exists in the form of Tifinagh. It is closely related to the nearby Tumzabt (Mozabite) and Teggargrent (Ouargli) languages.
Evaluation: True

Base Claim: Erwin S. Barrie managed a company from its founding in 1922 until he retired in 1975. The works of the artist closely associated with that company are more likely to be seen in an art gallery than the works of Armen Chakmakian.
Refined Claim: Erwin S. Barrie managed the Grand Central Art Galleries from its founding in 1922 until he retired in 1975. The works of the artist closely associated with that company are more likely to be seen in an art gallery than the works of Armen Chakmakian.
Evaluation: False

Base Claim: Joey Ramone and Miles Hunt do not have the same nationality.
Refined Claim: Joey Ramone and Miles Hunt do not have the same nationality.
Evaluation: True

Base Claim: The British actor that starred in Howard the Duck, alongside Lea Thompson, Jeffrey Jones, and Tim Robbins, is best known for the Stephen Sondheim Broadway production which British composer and lyricist received an Academy Award and Tony Awards.
Refined Claim: The British actor that starred in Howard the Duck, alongside Lea Thompson, Jeffrey Jones, and Tim Robbins, is best known for his performance in the Stephen Sondheim Broadway production "Into the Woods." The composer and lyricist of "Into the Woods," Stephen Sondheim, received an Academy Award and Tony Awards for his work.
Evaluation: False

Base Claim: The President that Jefferson County, Oklahoma was named in honor of, was an American Founding Father, a proponent of individual rights, who examined Ice Age fossils of Kentucky's Big Bone Lick.
Refined Claim: Jefferson County, Oklahoma was named in honor of Thomas Jefferson, an American Founding Father and the third President of the United States. He was known for his proponent of individual rights and his examination of Ice Age fossils of Kentucky's Big Bone Lick.
Evaluation: True

Base Claim: The director of Jackass Mail and Sam Wanamaker are directors.
Refined Claim: The director of Jackass Mail, Norman Z. McLeod, and Jack Wanamaker, a pseudonym used by American film director William Witney, are both directors.
Evaluation: True

Base Claim: The 432 Park Avenue building is taller than the building that is the site of the former Loew's State Theatre (New York City).
Refined Claim: The 432 Park Avenue building is taller than the building that is the site of the former Loew's State Theatre (New York City).
Evaluation: False

Base Claim: One of the 1990's films Ian Paul Cassidy appeared in was directed by an American actress born on March 26, 1960. This actress starred in Red Dawn.
Refined Claim: One of the 1990's films Ian Paul Cassidy appeared in was directed by an American actress born on March 26, 1960. This actress starred in Red Dawn.
Evaluation: False

Base Claim: Lubabalo Kondlo won a silver medal in the 2012  SportAccord World Mind Games inaugurated in 2011 in Beijing.
Refined Claim: Lubabalo Kondlo won a silver medal in the 2012  SportAccord World Mind Games inaugurated in 2011 in Beijing.
Evaluation: True
###

Base Claim: {base_claim}
Refined Claim: {refined_claim}
Evaluation:"""


#################### CLAIM REFINEMENT SECTION #############################
######################################################################
######################################################################
######################################################################
add_key_entities_not_supported_prompt = """<s> [INST] Refine a multi-hop claim using provided context information. Focus on clarifying and expanding the claim by integrating detailed information about inferred or implied entities.
Outcome: The refined claim should maintain the original structure while providing a complete, accurate expansion, ensuring all essential entities from the claim are explicitly mentioned.
Guidelines:
1. Relevance Check: Carefully assess the context to ensure relevance. Only use parts that directly clarify or correct the original claim.
2. Detail Integration: Seamlessly incorporate relevant details such as names, dates, and locations to specify previously implied or inferred entities, thereby enriching the claim without altering its fundamental structure.
3. Clarity and Precision: Keep your refinement concise and focused without adding unnecessary details and elaborations.
4. Accuracy: Ensure all information added is accurate and verifiable within the provided context. Do not introduce assumptions or unverified details.
[/INST]
###
CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
CONTEXT: 'Almost, Maine It premiered at the Portland Stage Company in Portland, Maine in 2004 where it broke box office records and garnered critical acclaim.\nThe Gin Game The Gin Game is a two-person, two-act play by Donald L. Coburn that premiered at American Theater Arts in Hollywood in September 1976, directed by Kip Niven.\nThe Patriots (play) The Patriots is an award-winning play written in a prologue and three acts by Sidney Kingsley in 1943.\nThe Last Days of Judas Iscariot The Last Days of Judas Iscariot is a play by American playwright Stephen Adly Guirgis first staged Off-Broadway at The Public Theater on March 2, 2005 directed by Philip Seymour Hoffman.\nTopdog/Underdog Topdog/Underdog is a play by American playwright Suzan-Lori Parks which premiered in 2001 off-Broadway in New York City.\nCollected Stories (play) Collected Stories is a play by Donald Margulies which premiered at South Coast Repertory in 1996, and was presented on Broadway in 2010.\nThe Iceman Cometh First published in 1946, the play premiered on Broadway at the Martin Beck Theatre on October 9, 1946, directed by Eddie Dowling, where it ran for 136 performances before closing on March 15, 1947.\nJason Wells (playwright) Jason Wells (born 1960) is an American actor and award-winning playwright.\nHal Holbrook He first received critical acclaim in 1954 for a one-man stage show he developed while studying at Denison University, performing as Mark Twain.\nWonder of the World (play) Wonder of the World is a play by American playwright David Lindsay-Abaire.'
REFINED CLAIM: The play 'Almost, Maine,' premiered by the Portland Stage Company in 2004, was a collaboration between an award-winning American actor and an accomplished playwright, breaking box office records and garnering critical acclaim.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
CONTEXT: Jimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nOzzy Osbourne John Michael "Ozzy" Osbourne (born 3 December 1948) is an English singer, songwriter, and actor.\nYou Gotta Sin to Get Saved You Gotta Sin to Get Saved is the second album by American singer-songwriter Maria McKee, released in 1993 (see 1993 in music).\nScott Stapp Anthony Scott Flippen (born August 8, 1973), also known as Scott Stapp, is an American singer, songwriter, and musician, known as the lead vocalist and lyricist of rock bands Creed and Art of Anarchy.\nBret Michaels Bret Michael Sychak (born March 15, 1963), professionally known as Bret Michaels, is an American singer-songwriter and musician.\nMichael Gira Michael Rolfe Gira ( ; born February 19, 1954) is an American singer-songwriter, musician, author and artist.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\nMarilyn Manson Brian Hugh Warner (born January 5, 1969), known professionally as Marilyn Manson, is an American singer, songwriter, musician, composer, actor, painter, author and former music journalist.\nCorey Taylor Corey Todd Taylor (born December 8, 1973) is an American musician, singer, songwriter, actor, and author, best known as the lead singer and lyricist of the American heavy metal band Slipknot and the American alternative metal band Stone Sour.
REFINED CLAIM: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters. Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
CONTEXT: 'Gamblers' Ballet "Cardinal Knowledge" was used as the outro music for the Cartoon Saloon, Oscar nominated animation "The Secret of Kells", and "Dúisigí" and "Cabhraigí Léi" were used in the Japanese film "Kadokawa", which was directed by Ryuichi Hiroki.\nFabrice Ziolkowski Fabrice Ziolkowski (born January 28, 1954) is a French-American screenwriter, director, producer, and voice director, best known for scripting the Oscar-nominated feature animation film "The Secret of Kells", writing the animated television series "Gawayn", and directing and producing the avant-garde documentary film "L.A.X.".\nThe Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\nArrietty Arrietty, titled The Borrower Arrietty (Japanese: 借りぐらしのアリエッティ , Hepburn: Kari-gurashi no Arietti ) in Japan and The Secret World of Arrietty in North America, is a 2010 Japanese animated fantasy film produced by Studio Ghibli and directed by Hiromasa Yonebayashi as his feature film debut as a director.\nMadame Tutli-Putli Madame Tutli-Putli is a 2007 stop motion-animated short film by Montreal filmmakers Chris Lavis and Maciek Szczerbowski, collectively known as Clyde Henry Productions, and produced by the National Film Board of Canada (NFB).\nSteamboy Steamboy (Japanese: スチームボーイ , Hepburn: Suchīmubōi ) is a 2004 Japanese steampunk animated action film produced by Sunrise, directed and co-written by Katsuhiro Otomo, his second major anime release, following "Akira".\nCartoon Saloon The company has also developed the animated film "The Secret of Kells".\nNasu: Summer in Andalusia Nasu: Summer in Andalusia (茄子 アンダルシアの夏 , Nasu: Andarushia no Natsu ) is a 2003 Japanese anime film by Madhouse, directed by Kitarō Kōsaka, the famed animation supervisor of the Oscar-winning anime film "Spirited Away" and "Princess Mononoke" and long-time collaborator of Studio Ghibli, and adapted from a short 3-tankōbon manga by Iō Kuroda, entitled "Nasu", which was serialized in the Afternoon manga magazine.\nEleanor's Secret Eleanor's Secret (original French title Kérity, la maison des contes) is a 2009 Franco-Italian animated feature film directed by Dominique Monféry.\n 'Summer Wars Summer Wars (Japanese: サマーウォーズ , Hepburn: Samā Wōzu ) is a 2009 Japanese animated science fiction film directed by Mamoru Hosoda, animated by Madhouse and distributed by Warner Bros.
REFINED CLAIM: Summer Wars is a 2009 Japanese animated science fiction film. 'The Secret of Kells,' a French-Belgian-Irish animated fantasy film, premiered on 8 February 2009. The 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
###
CLAIM: {claim}
CONTEXT: {context}
REFINED CLAIM:"""

add_key_entities_refined_prompt = """<s> [INST] Refine a multi-hop claim using provided context information. Focus on clarifying and expanding the claim by integrating detailed information about inferred or implied entities. The context not always provides information on all entities, so you should only add information that is directly relevant to the claim.
Outcome: The refined claim should maintain the original structure while providing a complete, accurate expansion, ensuring all essential entities from the claim are explicitly mentioned.
Guidelines:
1. Relevance Check: Carefully assess the context to ensure relevance. Only use parts that directly clarify the original claim.
2. Detail Integration: Seamlessly incorporate relevant details such as names, dates, and locations to specify previously implied or inferred entities, thereby enriching the claim without altering its fundamental structure.
3. Clarity and Precision: Keep your refinement concise and focused without adding unnecessary details and elaborations.
4. Accuracy: Ensure all information added is accurate and verifiable within the provided context. Do not introduce assumptions or unverified details.
[/INST]
###
CLAIM: The campaign, during which the Battle of Harlem was fought, was a series of engagements for control of New York City and the state of New Jersey, fought in the 1700s near White Plains, New York.
CONTEXT: Battle of Harlem Heights The Battle of Harlem Heights was fought during the New York and New Jersey campaign of the American Revolutionary War.\nBattle of White Plains The Battle of White Plains was a battle in the New York and New Jersey campaign of the American Revolutionary War fought on October 28, 1776, near White Plains, New York.\nOriskany Battlefield State Historic Site Oriskany Battlefield State Historic Site is a historic site in Oneida County, New York, United States that marks the Battle of Oriskany, fought in 1777 during the American Revolution, one of the bloodiest engagements of the war.\nNew York and New Jersey campaign The New York and New Jersey campaign was a series of battles for control of New York City and the state of New Jersey in the American Revolutionary War between British forces under General Sir William Howe and the Continental Army under General George Washington in 1776 and the winter months of 1777.\nBattle of Fort Washington The Battle of Fort Washington was a battle fought in New York on November 16, 1776 during the American Revolutionary War between the United States and Great Britain.
REFINED CLAIM: The New York and New Jersey campaign, during which the Battle of Harlem was fought, was a series of engagements for control of New York City and the state of New Jersey, fought in the 1700s near White Plains, New York, where the Battle of White Plains took place.

CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
CONTEXT: 'Almost, Maine' It premiered at the Portland Stage Company in Portland, Maine in 2004 where it broke box office records and garnered critical acclaim.\nThe Gin Game The Gin Game is a two-person, two-act play by Donald L. Coburn that premiered at American Theater Arts in Hollywood in September 1976, directed by Kip Niven.\nThe Patriots (play) The Patriots is an award-winning play written in a prologue and three acts by Sidney Kingsley in 1943.\nThe Last Days of Judas Iscariot The Last Days of Judas Iscariot is a play by American playwright Stephen Adly Guirgis first staged Off-Broadway at The Public Theater on March 2, 2005 directed by Philip Seymour Hoffman.\nTopdog/Underdog Topdog/Underdog is a play by American playwright Suzan-Lori Parks which premiered in 2001 off-Broadway in New York City.
REFINED CLAIM: The play 'Almost, Maine,' premiered by the Portland Stage Company in 2004, was a collaboration between an award-winning American actor and an accomplished playwright, breaking box office records and garnering critical acclaim.

CLAIM: Mark Dayton is the current governor of the state where the former wrestler, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.
CONTEXT: Elephant Parts "Elephant Parts" is one hour long with parody commercials and comedy sketches, and features five full-length music videos, including the popular songs "Rio", and "Cruisin'", which featured wrestler Steve Strong and Monterey-based comic "Chicago" Steve Barkley.\nGovernor of Minnesota The current governor of Minnesota is Mark Dayton of the Democratic-Farmer-Labor Party (DFL).\nStephen Cepello Stephen Cepello (born June 29, 1949) is an American artist and former professional wrestler. As a wrestler, he was best known by his ring name, "The California Terminator" Steve Strong.\nStephen Cepello After retiring from wrestling to focus on his art career, he was selected to paint the official Governor's Mansion and Minnesota State Capitol portraits of former wrestler and Governor of Minnesota Jesse Ventura.\nIndependence Party of Minnesota It was the party of former Minnesota Governor Jesse Ventura (1999–2003).
REFINED CLAIM: Mark Dayton is the current governor of Minnesota, the state where the former wrestler Stephen Cepello, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
CONTEXT: Jimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nOzzy Osbourne John Michael "Ozzy" Osbourne (born 3 December 1948) is an English singer, songwriter, and actor.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\nMarilyn Manson Brian Hugh Warner (born January 5, 1969), known professionally as Marilyn Manson, is an American singer, songwriter, musician, composer, actor, painter, author and former music journalist.
REFINED CLAIM: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters. Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: Guillermo del Toro directed the film which had the man, who starred in Bone Dry alongside Lance Henriksen, play the role of Jared Nomak.
CONTEXT: Luke Goss He has appeared in numerous films including "Blade II" (2002) as Jared Nomak, "One Night with the King" (2006) as King Xerxes, "Tekken" (2010) as Steve Fox and "Interview with a Hitman" (2012) as Viktor.\nBone Dry The film starred Luke Goss and Lance Henriksen.\nHellboy (2004 film) Hellboy is a 2004 American supernatural superhero film directed by Guillermo del Toro and starring Ron Perlman, loosely based on the Dark Horse Comics graphic novel "" by Mike Mignola.\nCronos (film) Cronos is a 1993 Mexican horror drama film written and directed by Guillermo del Toro, starring veteran Argentinean actor Federico Luppi and American actor Ron Perlman.\nBlade II It was written by David S. Goyer, who also wrote the previous film, directed by Guillermo del Toro, and had Wesley Snipes returning as the lead character and producer.
REFINED CLAIM: Guillermo del Toro directed "Blade II", the film which had Luke Goss starred as Jared Nomak. Luke Goss also appeared alongside Lance Henriksen in the film "Bone Dry."

CLAIM: The subdivisions of the city that Pervalka is part of, are called elderships in English.
CONTEXT: Neringa Municipality Neringa ( ) or Neringa Municipality (Lithuanian: "Neringos savivaldybė" ) is a municipality of Klaipėda County in westernmost Lithuania, comprising several villages in the Curonian Spit.\nPervalka Administratively, it is part of the city of Neringa.\nMunicipalities of Lithuania These are sub-divided into 60 municipalities (Lithuanian: plural – "savivaldybės", singular – "savivaldybė"), which in turn are further sub-divided into over 500 smaller groups, known as elderships (Lithuanian: plural – "seniūnijos", singular – "seniūnija").\nPervalka It is situated on the northern part of the Žirgai Cape and south of Pervalka Cape.\nGarkalne Municipality Garkalne Municipality (Latvian: "Garkalnes novads" ) is a municipality in Vidzeme, Latvia.
REFINED CLAIM: Pervalka is part of Neringa Municipality, which is one of Lithuania's municipalities subdivided into elderships.

CLAIM: The 2010 population of a town in Waldo County which was covered by the district that contains Mount View High School (Maine) was 2,099.
CONTEXT: Unity, Maine The population was 2,099 at the 2010 census.\nWaldo County, Maine As of the 2010 census, the population was 38,786.\nMaine School Administrative District 3 Maine School Administrative District 3 (MSAD 3) is an operating school district within Maine, covering the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity.\nMount View High School (Maine) The school is a part of Maine School Administrative District 3, and serves students from the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity and Waldo.\nProspect, Maine The population was 709 at the 2010 census.
REFINED CLAIM: The 2010 population of Unity, Maine, a town in Waldo County covered by Maine School Administrative District 3 which includes Mount View High School, was 2,099.

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
CONTEXT: 'Gamblers' Ballet "Cardinal Knowledge" was used as the outro music for the Cartoon Saloon, Oscar nominated animation "The Secret of Kells", and "Dúisigí" and "Cabhraigí Léi" were used in the Japanese film "Kadokawa", which was directed by Ryuichi Hiroki.\nFabrice Ziolkowski Fabrice Ziolkowski (born January 28, 1954) is a French-American screenwriter, director, producer, and voice director, best known for scripting the Oscar-nominated feature animation film "The Secret of Kells", writing the animated television series "Gawayn", and directing and producing the avant-garde documentary film "L.A.X.".\nThe Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\nEleanor's Secret Eleanor's Secret (original French title Kérity, la maison des contes) is a 2009 Franco-Italian animated feature film directed by Dominique Monféry.\n 'Summer Wars Summer Wars (Japanese: サマーウォーズ , Hepburn: Samā Wōzu ) is a 2009 Japanese animated science fiction film directed by Mamoru Hosoda, animated by Madhouse and distributed by Warner Bros.
REFINED CLAIM: Summer Wars is a 2009 Japanese animated science fiction film. 'The Secret of Kells,' a French-Belgian-Irish animated fantasy film, premiered on 8 February 2009. The 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.

CLAIM: The kind of company that this group, AIA Central, whose headquarters are in the former Furama Hong Kong Hotel, is a pan-Asian life insurance group.
CONTEXT: AIA Tower It was named after American International Assurance, a former member of American International Group, which occupies 12 floors of the building.\nAIA Central The AIA Central (), formerly called AIG Tower (), in Hong Kong is a 185-metre (607 ft.), 37-storey skyscraper that was completed in 2005 and serves as the headquarters of AIA Group.\nInsurance International Holdings Company Limited, is a Chinese insurance conglomerate incorporated and headquartered in Hong Kong.\nAIA Group known as AIA () is the largest independent public listed pan-Asian life insurance group.\nFurama Hong Kong Hotel The site is now occupied by the AIA Central office building.
REFINED CLAIM: AIA Central, located in the building formerly known as the Furama Hong Kong Hotel, serves as the headquarters for AIA Group, which is the largest independent public listed pan-Asian life insurance group.
###
CLAIM: {claim}
CONTEXT: {context}
REFINED CLAIM:"""

add_key_entities_refined_prompt_4shot= """<s> [INST] Refine a multi-hop claim using provided context information. Focus on clarifying and expanding the claim by integrating detailed information about inferred or implied entities. The context not always provides information on all entities, so you should only add information that is directly relevant to the claim.
Outcome: The refined claim should maintain the original structure while providing a complete, accurate expansion, ensuring all essential entities from the claim are explicitly mentioned.
Guidelines:
1. Relevance Check: Carefully assess the context to ensure relevance. Only use parts that directly clarify the original claim.
2. Detail Integration: Seamlessly incorporate relevant details such as names, dates, and locations to specify previously implied or inferred entities, thereby enriching the claim without altering its fundamental structure.
3. Clarity and Precision: Keep your refinement concise and focused without adding unnecessary details and elaborations.
4. Accuracy: Ensure all information added is accurate and verifiable within the provided context. Do not introduce assumptions or unverified details.
[/INST]
###
CLAIM: Guillermo del Toro directed the film which had the man, who starred in Bone Dry alongside Lance Henriksen, play the role of Jared Nomak.
CONTEXT: Luke Goss He has appeared in numerous films including "Blade II" (2002) as Jared Nomak, "One Night with the King" (2006) as King Xerxes, "Tekken" (2010) as Steve Fox and "Interview with a Hitman" (2012) as Viktor.\nBone Dry The film starred Luke Goss and Lance Henriksen.\nHellboy (2004 film) Hellboy is a 2004 American supernatural superhero film directed by Guillermo del Toro and starring Ron Perlman, loosely based on the Dark Horse Comics graphic novel "" by Mike Mignola.\nBlade II It was written by David S. Goyer, who also wrote the previous film, directed by Guillermo del Toro, and had Wesley Snipes returning as the lead character and producer.
REFINED: Guillermo del Toro directed "Blade II", the film which had Luke Goss starred as Jared Nomak. Luke Goss also appeared alongside Lance Henriksen in the film "Bone Dry".

CLAIM: Mark Dayton is the current governor of the state where the former wrestler, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.
CONTEXT: Elephant Parts "Elephant Parts" is one hour long with parody commercials and comedy sketches, and features five full-length music videos, including the popular songs "Rio", and "Cruisin'", which featured wrestler Steve Strong and Monterey-based comic "Chicago" Steve Barkley.\nGovernor of Minnesota The current governor of Minnesota is Mark Dayton of the Democratic-Farmer-Labor Party (DFL).\nStephen Cepello Stephen Cepello (born June 29, 1949) is an American artist and former professional wrestler. As a wrestler, he was best known by his ring name, "The California Terminator" Steve Strong.\nStephen Cepello After retiring from wrestling to focus on his art career, he was selected to paint the official Governor's Mansion and Minnesota State Capitol portraits of former wrestler and Governor of Minnesota Jesse Ventura.
REFINED: Mark Dayton is the current governor of Minnesota, the state where the former wrestler Stephen Cepello, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
CONTEXT: Jimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nOzzy Osbourne John Michael "Ozzy" Osbourne (born 3 December 1948) is an English singer, songwriter, and actor.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
REFINED: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters. Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: The 2010 population of a town in Waldo County which was covered by the district that contains Mount View High School (Maine) was 2,099.
CONTEXT: Unity, Maine The population was 2,099 at the 2010 census.\nWaldo County, Maine As of the 2010 census, the population was 38,786.\nMaine School Administrative District 3 Maine School Administrative District 3 (MSAD 3) is an operating school district within Maine, covering the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity.\nMount View High School (Maine) The school is a part of Maine School Administrative District 3, and serves students from the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity and Waldo.
REFINED: The 2010 population of Unity, Maine, a town in Waldo County covered by Maine School Administrative District 3 which includes Mount View High School, was 2,099.
###
CLAIM: {claim}
CONTEXT: {context}
REFINED:"""

add_key_entities_refined_prompt2 = """<s> [INST] Refine a multi-hop claim using provided context information. Focus on clarifying and expanding the claim by integrating detailed information about inferred or implied entities. The context not always provides information on all entities, so you should only add information that is directly relevant to the claim.
Outcome: The refined claim should maintain the original structure while providing a complete, accurate expansion, ensuring all essential entities from the claim are explicitly mentioned.
Guidelines:
1. Relevance Check: Carefully assess the context to ensure relevance. Only use parts that directly clarify the original claim.
2. Detail Integration: Seamlessly incorporate relevant details such as names, dates, and locations to specify previously implied or inferred entities, thereby enriching the claim without altering its fundamental structure.
3. Clarity and Precision: Keep your refinement concise and focused without adding unnecessary details and elaborations.
4. Accuracy: Ensure all information added is accurate and verifiable within the provided context. Do not introduce assumptions or unverified details.
[/INST]
###
CLAIM: The campaign, during which the Battle of Harlem was fought, was a series of engagements for control of New York City and the state of New Jersey, fought in the 1700s near White Plains, New York.
CONTEXT: Battle of Harlem Heights The Battle of Harlem Heights was fought during the New York and New Jersey campaign of the American Revolutionary War.\nBattle of White Plains The Battle of White Plains was a battle in the New York and New Jersey campaign of the American Revolutionary War fought on October 28, 1776, near White Plains, New York.\nNew York and New Jersey campaign The New York and New Jersey campaign was a series of battles for control of New York City and the state of New Jersey in the American Revolutionary War between British forces under General Sir William Howe and the Continental Army under General George Washington in 1776 and the winter months of 1777.
REFINED CLAIM: The New York and New Jersey campaign, during which the Battle of Harlem was fought, was a series of engagements for control of New York City and the state of New Jersey, fought in the 1700s near White Plains, New York, where the Battle of White Plains took place.

CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
CONTEXT: 'Almost, Maine' It premiered at the Portland Stage Company in Portland, Maine in 2004 where it broke box office records and garnered critical acclaim.\nThe Patriots (play) The Patriots is an award-winning play written in a prologue and three acts by Sidney Kingsley in 1943.
REFINED CLAIM: The play 'Almost, Maine,' premiered by the Portland Stage Company in 2004, was a collaboration between an award-winning American actor and an accomplished playwright, breaking box office records and garnering critical acclaim.

CLAIM: Mark Dayton is the current governor of the state where the former wrestler, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.
CONTEXT: Elephant Parts "Elephant Parts" is one hour long with parody commercials and comedy sketches, and features five full-length music videos, including the popular songs "Rio", and "Cruisin'", which featured wrestler Steve Strong and Monterey-based comic "Chicago" Steve Barkley.\nGovernor of Minnesota The current governor of Minnesota is Mark Dayton of the Democratic-Farmer-Labor Party (DFL).\nStephen Cepello Stephen Cepello (born June 29, 1949) is an American artist and former professional wrestler. As a wrestler, he was best known by his ring name, "The California Terminator" Steve Strong.\nStephen Cepello After retiring from wrestling to focus on his art career, he was selected to paint the official Governor's Mansion and Minnesota State Capitol portraits of former wrestler and Governor of Minnesota Jesse Ventura.
REFINED CLAIM: Mark Dayton is the current governor of Minnesota, the state where the former wrestler Stephen Cepello, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
CONTEXT: Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\nMarilyn Manson Brian Hugh Warner (born January 5, 1969), known professionally as Marilyn Manson, is an American singer, songwriter, musician, composer, actor, painter, author and former music journalist.
REFINED CLAIM: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters. Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: Guillermo del Toro directed the film which had the man, who starred in Bone Dry alongside Lance Henriksen, play the role of Jared Nomak.
CONTEXT: Luke Goss He has appeared in numerous films including "Blade II" (2002) as Jared Nomak, "One Night with the King" (2006) as King Xerxes, "Tekken" (2010) as Steve Fox and "Interview with a Hitman" (2012) as Viktor.\nBone Dry The film starred Luke Goss and Lance Henriksen.\nHellboy (2004 film) Hellboy is a 2004 American supernatural superhero film directed by Guillermo del Toro and starring Ron Perlman, loosely based on the Dark Horse Comics graphic novel "" by Mike Mignola.\nBlade II It was written by David S. Goyer, who also wrote the previous film, directed by Guillermo del Toro, and had Wesley Snipes returning as the lead character and producer.
REFINED CLAIM: Guillermo del Toro directed "Blade II", the film which had Luke Goss starred as Jared Nomak. Luke Goss also appeared alongside Lance Henriksen in the film "Bone Dry."

CLAIM: The subdivisions of the city that Pervalka is part of, are called elderships in English.
CONTEXT: Neringa Municipality Neringa ( ) or Neringa Municipality (Lithuanian: "Neringos savivaldybė" ) is a municipality of Klaipėda County in westernmost Lithuania, comprising several villages in the Curonian Spit.\nPervalka Administratively, it is part of the city of Neringa.\nMunicipalities of Lithuania These are sub-divided into 60 municipalities (Lithuanian: plural – "savivaldybės", singular – "savivaldybė"), which in turn are further sub-divided into over 500 smaller groups, known as elderships (Lithuanian: plural – "seniūnijos", singular – "seniūnija").\nPervalka It is situated on the northern part of the Žirgai Cape and south of Pervalka Cape.
REFINED CLAIM: Pervalka is part of Neringa Municipality, which is one of Lithuania's municipalities subdivided into elderships.

CLAIM: The 2010 population of a town in Waldo County which was covered by the district that contains Mount View High School (Maine) was 2,099.
CONTEXT: Unity, Maine The population was 2,099 at the 2010 census.\nWaldo County, Maine As of the 2010 census, the population was 38,786.\nMaine School Administrative District 3 Maine School Administrative District 3 (MSAD 3) is an operating school district within Maine, covering the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity.\nMount View High School (Maine) The school is a part of Maine School Administrative District 3, and serves students from the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity and Waldo.
REFINED CLAIM: The 2010 population of Unity, Maine, a town in Waldo County covered by Maine School Administrative District 3 which includes Mount View High School, was 2,099.

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
CONTEXT: The Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\n 'Summer Wars Summer Wars (Japanese: サマーウォーズ , Hepburn: Samā Wōzu ) is a 2009 Japanese animated science fiction film directed by Mamoru Hosoda, animated by Madhouse and distributed by Warner Bros.
REFINED CLAIM: Summer Wars is a 2009 Japanese animated science fiction film. 'The Secret of Kells,' a French-Belgian-Irish animated fantasy film, premiered on 8 February 2009. The 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.

CLAIM: The kind of company that this group, AIA Central, whose headquarters are in the former Furama Hong Kong Hotel, is a pan-Asian life insurance group.
CONTEXT: AIA Tower It was named after American International Assurance, a former member of American International Group, which occupies 12 floors of the building.\nAIA Central The AIA Central (), formerly called AIG Tower (), in Hong Kong is a 185-metre (607 ft.), 37-storey skyscraper that was completed in 2005 and serves as the headquarters of AIA Group.\nAIA Group known as AIA () is the largest independent public listed pan-Asian life insurance group.\nFurama Hong Kong Hotel The site is now occupied by the AIA Central office building.
REFINED CLAIM: AIA Central, located in the building formerly known as the Furama Hong Kong Hotel, serves as the headquarters for AIA Group, which is the largest independent public listed pan-Asian life insurance group.
###
CLAIM: {claim}
CONTEXT: {context}
REFINED CLAIM:"""

add_key_entities_refined_prompt3 = """<s> [INST] Refine a multi-hop claim using provided context information. Focus on clarifying and expanding the claim by integrating detailed information about inferred or implied entities. The context not always provides information on all entities, so you should only add information that is directly relevant to the claim.
Outcome: The refined claim should maintain the original structure while providing a complete, accurate expansion, ensuring all essential entities from the claim are explicitly mentioned.
Guidelines:
1. Relevance Check: Carefully assess the context to ensure relevance. Only use parts that directly clarify the original claim.
2. Detail Integration: Seamlessly incorporate relevant details such as names, dates, and locations to specify previously implied or inferred entities, thereby enriching the claim without altering its fundamental structure.
3. Clarity and Precision: Keep your refinement concise and focused without adding unnecessary details and elaborations.
4. Accuracy: Ensure all information added is accurate and verifiable within the provided context. Do not introduce assumptions or unverified details.
[/INST]
###
CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
CONTEXT: 'Almost, Maine' It premiered at the Portland Stage Company in Portland, Maine in 2004 where it broke box office records and garnered critical acclaim.\nThe Patriots (play) The Patriots is an award-winning play written in a prologue and three acts by Sidney Kingsley in 1943.
REFINED CLAIM: The play 'Almost, Maine,' premiered by the Portland Stage Company in 2004, was a collaboration between an award-winning American actor and an accomplished playwright, breaking box office records and garnering critical acclaim.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
CONTEXT: Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\nMarilyn Manson Brian Hugh Warner (born January 5, 1969), known professionally as Marilyn Manson, is an American singer, songwriter, musician, composer, actor, painter, author and former music journalist.
REFINED CLAIM: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters. Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: Guillermo del Toro directed the film which had the man, who starred in Bone Dry alongside Lance Henriksen, play the role of Jared Nomak.
CONTEXT: Luke Goss He has appeared in numerous films including "Blade II" (2002) as Jared Nomak, "One Night with the King" (2006) as King Xerxes, "Tekken" (2010) as Steve Fox and "Interview with a Hitman" (2012) as Viktor.\nBone Dry The film starred Luke Goss and Lance Henriksen.\nHellboy (2004 film) Hellboy is a 2004 American supernatural superhero film directed by Guillermo del Toro and starring Ron Perlman, loosely based on the Dark Horse Comics graphic novel "" by Mike Mignola.\nBlade II It was written by David S. Goyer, who also wrote the previous film, directed by Guillermo del Toro, and had Wesley Snipes returning as the lead character and producer.
REFINED CLAIM: Guillermo del Toro directed "Blade II", the film which had Luke Goss starred as Jared Nomak. Luke Goss also appeared alongside Lance Henriksen in the film "Bone Dry."

CLAIM: The kind of company that this group, AIA Central, whose headquarters are in the former Furama Hong Kong Hotel, is a pan-Asian life insurance group.
CONTEXT: AIA Tower It was named after American International Assurance, a former member of American International Group, which occupies 12 floors of the building.\nAIA Central The AIA Central (), formerly called AIG Tower (), in Hong Kong is a 185-metre (607 ft.), 37-storey skyscraper that was completed in 2005 and serves as the headquarters of AIA Group.\nAIA Group known as AIA () is the largest independent public listed pan-Asian life insurance group.\nFurama Hong Kong Hotel The site is now occupied by the AIA Central office building.
REFINED CLAIM: AIA Central, located in the building formerly known as the Furama Hong Kong Hotel, serves as the headquarters for AIA Group, which is the largest independent public listed pan-Asian life insurance group.
###
CLAIM: {claim}
CONTEXT: {context}
REFINED CLAIM:"""


add_key_entities_and_change_prompt = """<s> [INST] Refine a multi-hop claim using provided context information. Focus on clarifying and expanding the claim by integrating detailed information about inferred or implied entities. The context not always provides information on all entities, so you should only add information that is directly relevant to the claim. Some claims contain factual wrong information, the context might provide the correct information which should be changed accordingly.
Outcome: The refined claim should maintain the original structure while providing a complete, accurate expansion, ensuring all essential entities from the claim are explicitly mentioned, and correcting any factual inaccuracies.
Guidelines:
1. Relevance Check: Carefully assess the context to ensure relevance. Only use parts that directly clarify the original claim.
2. Detail Integration: Seamlessly incorporate relevant details such as names, dates, and locations to specify previously implied or inferred entities, thereby enriching the claim without altering its fundamental structure.
3. Clarity and Precision: Keep your refinement concise and focused without adding unnecessary details and elaborations.
4. Accuracy: Ensure all information added is accurate and verifiable within the provided context. Do not introduce assumptions or unverified details.
[/INST]
###
CLAIM: The campaign, during which the Battle of Harlem was fought, was a series of engagements for control of New York City and the state of New Jersey, fought in the 1700s near White Plains, New York.
CONTEXT: Battle of Harlem Heights The Battle of Harlem Heights was fought during the New York and New Jersey campaign of the American Revolutionary War.\nBattle of White Plains The Battle of White Plains was a battle in the New York and New Jersey campaign of the American Revolutionary War fought on October 28, 1776, near White Plains, New York.\nOriskany Battlefield State Historic Site Oriskany Battlefield State Historic Site is a historic site in Oneida County, New York, United States that marks the Battle of Oriskany, fought in 1777 during the American Revolution, one of the bloodiest engagements of the war.\nNew York and New Jersey campaign The New York and New Jersey campaign was a series of battles for control of New York City and the state of New Jersey in the American Revolutionary War between British forces under General Sir William Howe and the Continental Army under General George Washington in 1776 and the winter months of 1777.\nBattle of Fort Washington The Battle of Fort Washington was a battle fought in New York on November 16, 1776 during the American Revolutionary War between the United States and Great Britain.
REFINED CLAIM: The New York and New Jersey campaign, during which the Battle of Harlem was fought, was a series of engagements for control of New York City and the state of New Jersey, fought in the 1700s near White Plains, New York, where the Battle of White Plains took place.

CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company from 1986 to 1993.
CONTEXT: 'Almost, Maine' It premiered at the Portland Stage Company in Portland, Maine in 2004 where it broke box office records and garnered critical acclaim.\nThe Gin Game The Gin Game is a two-person, two-act play by Donald L. Coburn that premiered at American Theater Arts in Hollywood in September 1976, directed by Kip Niven.\nThe Patriots (play) The Patriots is an award-winning play written in a prologue and three acts by Sidney Kingsley in 1943.\nThe Last Days of Judas Iscariot The Last Days of Judas Iscariot is a play by American playwright Stephen Adly Guirgis first staged Off-Broadway at The Public Theater on March 2, 2005 directed by Philip Seymour Hoffman.\nTopdog/Underdog Topdog/Underdog is a play by American playwright Suzan-Lori Parks which premiered in 2001 off-Broadway in New York City.
REFINED CLAIM: The play 'Almost, Maine,' premiered by the Portland Stage Company in 2004, was a collaboration between an award-winning American actor and an accomplished playwright, breaking box office records and garnering critical acclaim in 2004.

CLAIM: Mark Dayton is the current governor of the state where the former wrestler, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.
CONTEXT: Elephant Parts "Elephant Parts" is one hour long with parody commercials and comedy sketches, and features five full-length music videos, including the popular songs "Rio", and "Cruisin'", which featured wrestler Steve Strong and Monterey-based comic "Chicago" Steve Barkley.\nGovernor of Minnesota The current governor of Minnesota is Mark Dayton of the Democratic-Farmer-Labor Party (DFL).\nStephen Cepello Stephen Cepello (born June 29, 1949) is an American artist and former professional wrestler. As a wrestler, he was best known by his ring name, "The California Terminator" Steve Strong.\nStephen Cepello After retiring from wrestling to focus on his art career, he was selected to paint the official Governor's Mansion and Minnesota State Capitol portraits of former wrestler and Governor of Minnesota Jesse Ventura.\nIndependence Party of Minnesota It was the party of former Minnesota Governor Jesse Ventura (1999–2003).
REFINED CLAIM: Mark Dayton is the current governor of Minnesota, the state where the former wrestler Stephen Cepello, who appeared in a music video in "Elephant Parts", painted the Governor's Mansion.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
The musician that signed Godhead (band) to their label and Joe Lynn Turner are both AOR singer, songwriters.
CONTEXT: Jimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nOzzy Osbourne John Michael "Ozzy" Osbourne (born 3 December 1948) is an English singer, songwriter, and actor.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.\nMarilyn Manson Brian Hugh Warner (born January 5, 1969), known professionally as Marilyn Manson, is an American singer, songwriter, musician, composer, actor, painter, author and former music journalist.
REFINED CLAIM: The musician that signed Godhead (band) to their label and Joe Lynn Turner are both American singer, songwriters. Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: Chasty Ballesteros directed the film which had the man, who starred in Bone Dry alongside Lance Henriksen, play the role of Jared Nomak.
CONTEXT: Luke Goss He has appeared in numerous films including "Blade II" (2002) as Jared Nomak, "One Night with the King" (2006) as King Xerxes, "Tekken" (2010) as Steve Fox and "Interview with a Hitman" (2012) as Viktor.\nBone Dry The film starred Luke Goss and Lance Henriksen.\nHellboy (2004 film) Hellboy is a 2004 American supernatural superhero film directed by Guillermo del Toro and starring Ron Perlman, loosely based on the Dark Horse Comics graphic novel "" by Mike Mignola.\nCronos (film) Cronos is a 1993 Mexican horror drama film written and directed by Guillermo del Toro, starring veteran Argentinean actor Federico Luppi and American actor Ron Perlman.\nBlade II It was written by David S. Goyer, who also wrote the previous film, directed by Guillermo del Toro, and had Wesley Snipes returning as the lead character and producer.
REFINED CLAIM: Guillermo del Toro directed "Blade II", the film which had Luke Goss starred as Jared Nomak. Luke Goss also appeared alongside Lance Henriksen in the film "Bone Dry."

CLAIM: The subdivisions of the city that Pervalka is part of, are called elderships in English.
CONTEXT: Neringa Municipality Neringa ( ) or Neringa Municipality (Lithuanian: "Neringos savivaldybė" ) is a municipality of Klaipėda County in westernmost Lithuania, comprising several villages in the Curonian Spit.\nPervalka Administratively, it is part of the city of Neringa.\nMunicipalities of Lithuania These are sub-divided into 60 municipalities (Lithuanian: plural – "savivaldybės", singular – "savivaldybė"), which in turn are further sub-divided into over 500 smaller groups, known as elderships (Lithuanian: plural – "seniūnijos", singular – "seniūnija").\nPervalka It is situated on the northern part of the Žirgai Cape and south of Pervalka Cape.\nGarkalne Municipality Garkalne Municipality (Latvian: "Garkalnes novads" ) is a municipality in Vidzeme, Latvia.
REFINED CLAIM: Pervalka is part of Neringa Municipality, which is one of Lithuania's municipalities subdivided into elderships.

CLAIM: The 2010 population of a town in Waldo County which was covered by the district that contains Mount View High School (Maine) was 2,099.
CONTEXT: Unity, Maine The population was 2,099 at the 2010 census.\nWaldo County, Maine As of the 2010 census, the population was 38,786.\nMaine School Administrative District 3 Maine School Administrative District 3 (MSAD 3) is an operating school district within Maine, covering the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity.\nMount View High School (Maine) The school is a part of Maine School Administrative District 3, and serves students from the towns of Brooks, Freedom, Jackson, Knox, Liberty, Monroe, Montville, Thorndike, Troy, Unity and Waldo.\nProspect, Maine The population was 709 at the 2010 census.
REFINED CLAIM: The 2010 population of Unity, Maine, a town in Waldo County covered by Maine School Administrative District 3 which includes Mount View High School, was 2,099.

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
CONTEXT: 'Gamblers' Ballet "Cardinal Knowledge" was used as the outro music for the Cartoon Saloon, Oscar nominated animation "The Secret of Kells", and "Dúisigí" and "Cabhraigí Léi" were used in the Japanese film "Kadokawa", which was directed by Ryuichi Hiroki.\nFabrice Ziolkowski Fabrice Ziolkowski (born January 28, 1954) is a French-American screenwriter, director, producer, and voice director, best known for scripting the Oscar-nominated feature animation film "The Secret of Kells", writing the animated television series "Gawayn", and directing and producing the avant-garde documentary film "L.A.X.".\nThe Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\nEleanor's Secret Eleanor's Secret (original French title Kérity, la maison des contes) is a 2009 Franco-Italian animated feature film directed by Dominique Monféry.\n 'Summer Wars Summer Wars (Japanese: サマーウォーズ , Hepburn: Samā Wōzu ) is a 2009 Japanese animated science fiction film directed by Mamoru Hosoda, animated by Madhouse and distributed by Warner Bros.
REFINED CLAIM: Summer Wars is a 2009 Japanese animated science fiction film. 'The Secret of Kells,' a French-Belgian-Irish animated fantasy film, premiered on 8 February 2009. The 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.

CLAIM: The kind of company that this group, whose headquarters are in the former Furama Hong Kong Hotel, is a Tiense life insurance group.
CONTEXT: AIA Tower It was named after American International Assurance, a former member of American International Group, which occupies 12 floors of the building.\nAIA Central The AIA Central (), formerly called AIG Tower (), in Hong Kong is a 185-metre (607 ft.), 37-storey skyscraper that was completed in 2005 and serves as the headquarters of AIA Group.\nInsurance International Holdings Company Limited, is a Chinese insurance conglomerate incorporated and headquartered in Hong Kong.\nAIA Group known as AIA () is the largest independent public listed pan-Asian life insurance group.\nFurama Hong Kong Hotel The site is now occupied by the AIA Central office building.
REFINED CLAIM: AIA Central, located in the building formerly known as the Furama Hong Kong Hotel, serves as the headquarters for AIA Group, which is the largest independent public listed pan-Asian life insurance group.
###
CLAIM: {claim}
CONTEXT: {context}
REFINED CLAIM:"""















































add_key_entities_based_on_question_answering_prompt = """<s> [INST] Your task is to refine a multi-hop claim using provided context information, which consists of question and answers. 
Aim to enrich the claim with specific details that are crucial for understanding the claim's full meaning and implications.
Important: Not all context is relevant and it might happen that none of the context sentences are relevant, in which case you should not add any information and just return the original multi-hop claim and no other text.
Each refined claim should:
1. Identify and Integrate Key Information: Look for names, dates, and locations in the context that directly relate to the claim. Focus on embedding this information to make the claim more detailed and informative.
2. Preserve the Essence of the Claim: Ensure that the refined claim remains true to the original message. It should provide a fuller picture without changing the fundamental intent of the claim.
3. Be Concise and Focused: While adding depth, keep the refinement clear and to the point. Avoid inserting extraneous or speculative details.
4. Identify Inferred Entities: Focus on entities that are indirectly referenced or can be deduced from the claim's context to fully understand its narrative and implications.
5. Accuracy is Key: Ensure all added details are factually correct and supported by the context provided. Do not fabricate or assume information not present in the context.
[/INST]
###
CLAIM: Legend of the Boneknapper Dragon is a 2010 animated short film by DreamWorks Animation. It is contained in the franchise based on the books written by a writer. The writer also wrote "That Rabbit Belongs to Emily Brown".
CONTEXT: Which 2010 animated short film by DreamWorks Animation is the 'Legend of the Boneknapper Dragon'?
Yes, "Legend of the Boneknapper Dragon" is a 2010 animated short film by DreamWorks Animation.
What franchise is the 'Legend of the Boneknapper Dragon' a part of, based on books written by a specific writer?
The 'Legend of the Boneknapper Dragon' is a part of the 'How to Train Your Dragon' franchise, which is loosely based on the British book series by Cressida Cowell.
Who is the writer that wrote the 'Legend of the Boneknapper Dragon' and 'That Rabbit Belongs to Emily Brown'?
Cressida Cowell wrote 'That Rabbit Belongs to Emily Brown' and 'How to Train Your Dragon'. The 'Legend of the Boneknapper Dragon' is one of the books in the 'How to Train Your Dragon' series.
REFINED CLAIM: Legend of the Boneknapper Dragon is a 2010 animated short film by DreamWorks Animation, part of the How to Train Your Dragon franchise, which is based on the How to Train Your Dragon book series by British author Cressida Cowell. Cressida Cowell also wrote 'That Rabbit Belongs to Emily Brown'.

CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
CONTEXT: What is the name of the play performed by the Portland Stage Company that broke box office records and received critical acclaim?
Almost, Maine.
Who is the award-winning American actor involved in the play that premiered at the Portland Stage Company?
John Cariani
REFINED CLAIM: The play 'Almost, Maine,' by John Cariani, an award-winning American actor and accomplished playwright, was performed by the Portland Stage Company where it broke box office records and garnered critical acclaim.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
CONTEXT: Who is the musician that signed the band Godhead to their label?
Marilyn Manson.
Which American singer and songwriter released 'Second Hand Life' as their tenth solo studio album?
Joe Lynn Turner.
REFINED CLAIM: Marilyn Manson, the musician that signed Godhead (band) to their label Posthuman Records, and Joe Lynn Turner are both American singer, songwriters. Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
CONTEXT: Which film did Mike McFarland voice in, Summer Wars or The Secret of Kells?
Summer Wars.
Is 'The Secret of Kells' a stop-motion animated film, and what is its country of origin?
'The Secret of Kells' is not a stop-motion animated film; it is a traditional animated fantasy film. It is a co-production among France, Belgium, and Ireland.
In what year was the film Summer Wars released?
2009.
What kind of film is 'Summer Wars'?
'Summer Wars' is a 2009 Japanese animated science fiction film.
REFINED CLAIM: Summer Wars is a 2009 Japanese animated science fiction film, which Mike McFarland voiced in. 'The Secret of Kells', a French-Belgian-Irish animated fantasy film, premiered on 8 February 2009.

CLAIM: Grigori Vasilyevich Aleksandrov was a prominent director who worked frequently with the composer of Moya Moskva.
CONTEXT: Who was Grigori Vasilyevich Aleksandrov?
Grigori Vasilyevich Aleksandrov, originally named Mormonenko, was a prominent Soviet film director.
Which composer worked frequently with Grigori Vasilyevich Aleksandrov and composed 'Moya Moskva'?
Isaak Osipovich Dunayevsky.
REFINED CLAIM: Grigori Vasilyevich Aleksandrov was a prominent director who worked frequently with Isaak Osipovich Dunayevsky, the composer of Moya Moskva.

CLAIM: The Lusty Men is a western directed by this man. He and the writer of America America are both actors and directors.
CONTEXT: Who directed the western film 'The Lusty Men'?
Nicholas Ray.
Who is the writer and director of 'America America'?
Elia Kazan.
REFINED CLAIM: The Lusty Men is a western directed by Nicholas Ray. He and Elia Kazan, the writer of America America, are both actors and directors.

CLAIM: The actress that starred as a parent in Sky High (2005 film) is married to one of the subjects of film-critic Nigel Andrews' books. She also starred in a 1994 film about a pregnant woman who is stranded at a trading post during the American Civil War.
CONTEXT: Who is the actress that played a parent in the 2005 film Sky High?
Kelly Preston.
Which film critic Nigel Andrews wrote a book about, and who is the subject of that book married to the actress from Sky Hig'?
John Travolta.
What is the name of the 1994 film in which the Sky High actress starred as a pregnant woman stranded at a trading post during the American Civil War?
Cheyenne Warrior.
REFINED CLAIM: Kelly Preston, the actress that starred as a parent in Sky High (2005 film), is married to John Travolta, one of the subjects of film-critic Nigel Andrews' books. She also starred in Cheyenne Warrior, a 1994 film about a pregnant woman who is stranded at a trading post during the American Civil War.

CLAIM: Dana White is the president of the organization that Forrest Griffin vs. Stephan Bonnar was thought to be one of the greatest bar fights in history.
CONTEXT: What organization is Dana White the president of, where Forrest Griffin and Stephan Bonnar had their famous fight?
Ultimate Fighting Championship (UFC).
Which fight between Forrest Griffin and Stephan Bonnar is often referred to as one of the greatest bar fights in history?
The Ultimate Fighter 1 contest of the Ultimate Fighting Championship.
REFINED CLAIM: Dana White is the president of the Ultimate Fighting Championship (UFC), where Forrest Griffin and Stephan Bonnar had their famous fight in the finals of The Ultimate Fighter 1 contest, often referred to as one of the greatest bar fights in history.

CLAIM: A species related to Xanthosoma and Coronilla are both flowering plants.
QUESTIONS: Which species is related to Xanthosoma and is a flowering plant?
Colocasia.
Is Coronilla a flowering plant?
Yes.
REFINED CLAIM: Colocasia, a species related to Xanthosoma, and Coronilla are both flowering plants.

###
CLAIM: {claim}
CONTEXT: {context}
REFINED CLAIM:"""


add_key_entities_based_on_question_answering_prompt2 = """<s> [INST] Your task is to refine a multi-hop claim using provided context information, which consists of question and answers. 
Aim to enrich the claim with specific details that are crucial for understanding the claim's full meaning and implications.
Important: Not all context is relevant and it might happen that none of the context sentences are relevant, in which case you should not add any information and just return the original multi-hop claim and no other text.
Each refined claim should:
1. Identify and Integrate Key Information: Look for names, dates, and locations in the context that directly relate to the claim. Focus on embedding this information to make the claim more detailed and informative.
2. Preserve the Essence of the Claim: Ensure that the refined claim remains true to the original message. It should provide a fuller picture without changing the fundamental intent of the claim.
3. Be Concise and Focused: While adding depth, keep the refinement clear and to the point. Avoid inserting extraneous or speculative details.
4. Identify Inferred Entities: Focus on entities that are indirectly referenced or can be deduced from the claim's context to fully understand its narrative and implications.
5. Accuracy is Key: Ensure all added details are factually correct and supported by the context provided. Do not fabricate or assume information not present in the context.
[/INST]
###
CLAIM: The President that Jefferson County, Oklahoma was named in honor of, was an American Founding Father, a proponent of individual rights, who examined Ice Age fossils of Kentucky's Big Bone Lick.
CONTEXT: Which President was Jefferson County, Oklahoma named in honor of?
Jefferson County, Oklahoma was named in honor of President Thomas Jefferson.
Who was an American Founding Father, a proponent of individual rights, and examined Ice Age fossils of Kentucky's Big Bone Lick?
Thomas Jefferson, an American Founding Father and the third President of the United States, was a proponent of individual rights. He examined Ice Age fossils of Kentucky's Big Bone Lick.
REFINED CLAIM: Thomas Jefferson, the President that Jefferson County, Oklahoma was named in honor of, was an American Founding Father, a proponent of individual rights, and examined Ice Age fossils of Kentucky's Big Bone Lick. 

CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
CONTEXT: What is the name of the play performed by the Portland Stage Company that broke box office records and received critical acclaim?
The play performed by the Portland Stage Company that broke box office records and received critical acclaim is "Almost, Maine."
Who is the award-winning American actor involved in the play that premiered at the Portland Stage Company?
The award-winning American actor involved in the play that premiered at the Portland Stage Company is John Cariani.
REFINED CLAIM: The play 'Almost, Maine,' by John Cariani, an award-winning American actor and accomplished playwright, was performed by the Portland Stage Company where it broke box office records and garnered critical acclaim.

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
CONTEXT: Who is the musician that signed the band Godhead to their label?
Marilyn Manson signed the band Godhead to his label, Posthuman Records.
Which American singer and songwriter released 'Second Hand Life' as their tenth solo studio album?
Joe Lynn Turner released 'Second Hand Life' as his tenth solo studio album.
REFINED CLAIM: Marilyn Manson, the musician that signed Godhead (band) to their label Posthuman Records, and Joe Lynn Turner are both American singer, songwriters. Second Hand Life is the tenth solo studio album by Joe Lynn Turner.

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
CONTEXT: Which film did Mike McFarland voice in, Summer Wars or The Secret of Kells?
Mike McFarland voiced in the film Summer Wars.
Is 'The Secret of Kells' a stop-motion animated film, and what is its country of origin?
'The Secret of Kells' is not a stop-motion animated film; it is a traditional animated fantasy film. It is a co-production among France, Belgium, and Ireland.
In what year was the film Summer Wars released?
The film Summer Wars was released in the year 2009.
What kind of film is 'Summer Wars'?
'Summer Wars' is a 2009 Japanese animated science fiction film.
REFINED CLAIM: Summer Wars is a 2009 Japanese animated science fiction film, which Mike McFarland voiced in. 'The Secret of Kells', a French-Belgian-Irish animated fantasy film, premiered on 8 February 2009.

CLAIM: Grigori Vasilyevich Aleksandrov was a prominent director who worked frequently with the composer of Moya Moskva.
CONTEXT: Who was Grigori Vasilyevich Aleksandrov?
Grigori Vasilyevich Aleksandrov, originally named Mormonenko, was a prominent Soviet film director, recognized as a People's Artist of the USSR in 1947 and a Hero of Socialist Labor in 1973. He won the Stalin Prizes in 1941 and 1950.
Which composer worked frequently with Grigori Vasilyevich Aleksandrov and composed 'Moya Moskva'?
Isaak Osipovich Dunayevsky, a celebrated Soviet film composer and conductor, worked frequently with Grigori Vasilyevich Aleksandrov and composed the music for 'Moya Moskva'.
REFINED CLAIM: Grigori Vasilyevich Aleksandrov was a prominent director who worked frequently with Isaak Osipovich Dunayevsky, the composer of Moya Moskva.

CLAIM: The Lusty Men is a western directed by this man. He and the writer of America America are both actors and directors.
CONTEXT: Who directed the western film 'The Lusty Men'?
Nicholas Ray directed the western film 'The Lusty Men'.
Who is the writer and director of 'America America'?
Elia Kazan is the writer and director of 'America America'.
REFINED CLAIM: The Lusty Men is a western directed by Nicholas Ray. He and Elia Kazan, the writer of America America, are both actors and directors.

CLAIM: The actress that starred as a parent in Sky High (2005 film) is married to one of the subjects of film-critic Nigel Andrews' books. She also starred in a 1994 film about a pregnant woman who is stranded at a trading post during the American Civil War.
CONTEXT: Who is the actress that played a parent in the 2005 film Sky High?
Kelly Preston played a parent in the 2005 film Sky High.
Which film critic Nigel Andrews wrote a book about, and who is the subject of that book married to the actress from Sky Hig'?
Nigel Andrews wrote a book about John Travolta, who is married to Kelly Preston, the actress from Sky High.
What is the name of the 1994 film in which the Sky High actress starred as a pregnant woman stranded at a trading post during the American Civil War?
The name of the 1994 film is 'Cheyenne Warrior', in which Kelly Preston starred as a pregnant woman stranded at a trading post during the American Civil War.
REFINED CLAIM: Kelly Preston, the actress that starred as a parent in Sky High (2005 film), is married to John Travolta, one of the subjects of film-critic Nigel Andrews' books. She also starred in Cheyenne Warrior, a 1994 film about a pregnant woman who is stranded at a trading post during the American Civil War.

CLAIM: Dana White is the president of the organization that Forrest Griffin vs. Stephan Bonnar was thought to be one of the greatest bar fights in history.
CONTEXT: What organization is Dana White the president of, where Forrest Griffin and Stephan Bonnar had their famous fight?
Dana White is the president of the Ultimate Fighting Championship (UFC), where Forrest Griffin and Stephan Bonnar had their famous fight.
Which fight between Forrest Griffin and Stephan Bonnar is often referred to as one of the greatest bar fights in history?
The fight between Forrest Griffin and Stephan Bonnar that is often referred to as one of the greatest bar fights in history is the finals of The Ultimate Fighter 1 contest of the Ultimate Fighting Championship.
REFINED CLAIM: Dana White is the president of the Ultimate Fighting Championship (UFC), where Forrest Griffin and Stephan Bonnar had their famous fight in the finals of The Ultimate Fighter 1 contest, often referred to as one of the greatest bar fights in history.

CLAIM: A species related to Xanthosoma and Coronilla are both flowering plants.
QUESTIONS: Which species is related to Xanthosoma and is a flowering plant?
Colocasia is related to Xanthosoma and is a flowering plant.
Is Coronilla a flowering plant?
Yes, Coronilla is a genus of flowering plants.
REFINED CLAIM: Colocasia, a species related to Xanthosoma, and Coronilla are both flowering plants.
###
CLAIM: {claim}
CONTEXT: {context}
REFINED CLAIM:"""



"""
CLAIM: Hey Monday was featured on the cover page of the Alternative Press music magazine for the '100 Bands You Need to Know in 2010' alongside The Cab discography and Never Shout Never.
QUESTIONS: How are The Cab discography and Never Shout Never related to Hey Monday in the context of music and publications?\nWhat is the significance of the '100 Bands You Need to Know in 2010' list where Hey Monday, The Cab, and Never Shout Never were mentioned?
"""



#################### SUBQUESTION SECTION #############################
######################################################################
######################################################################
######################################################################
question_answering_prompt_simple = """<s> [INST] Your task is to answer the given question based on the given context.
Do not make up any information, only use information mentioned in the context.
Please be as concise as possible and provide a clear answer with only the most relevant information to the question.
[/INST]
QUESTION: {question}
CONTEXT: {context}
ANSWER:"""

question_answering_prompt = """<s> [INST] Your task is to answer the given question based on the given context.
Do not make up any information, only use information mentioned in the context.
Please be as concise as possible and provide a clear answer with only the most relevant information to the question.
[/INST]
###
QUESTION: What is the name of the play performed by the Portland Stage Company that broke box office records and received critical acclaim?
CONTEXT: 'Almost, Maine' It premiered at the Portland Stage Company in Portland, Maine in 2004 where it broke box office records and garnered critical acclaim.\nThe Patriots (play) The Patriots is an award-winning play written in a prologue and three acts by Sidney Kingsley in 1943.\nTopdog/Underdog Topdog/Underdog is a play by American playwright Suzan-Lori Parks which premiered in 2001 off-Broadway in New York City.
ANSWER: Almost, Maine.

QUESTION: Which American singer and songwriter released 'Second Hand Life' as their tenth solo studio album?
CONTEXT: Jimmy Urine James Euringer (born September 7, 1969), known professionally as Jimmy Urine, is an American singer, songwriter, and musician.\nTommy Henriksen Tommy Henriksen (born February 21, 1964) is an American musician from Port Jefferson, New York best known for his work as a guitarist, bassist and songwriter with Alice Cooper, the Hollywood Vampires and German metal band Warlock.\nSecond Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
ANSWER: Joe Lynn Turner.

QUESTION: Is 'The Secret of Kells' a stop-motion animated film, and what is its country of origin?
CONTEXT: 'Gamblers' Ballet "Cardinal Knowledge" was used as the outro music for the Cartoon Saloon, Oscar nominated animation "The Secret of Kells", and "Dúisigí" and "Cabhraigí Léi" were used in the Japanese film "Kadokawa", which was directed by Ryuichi Hiroki.\nThe Secret of Kells The Secret of Kells is a 2009 French-Belgian-Irish animated fantasy film animated by Cartoon Saloon that premiered on 8 February 2009 at the 59th Berlin International Film Festival.\n 'Summer Wars Summer Wars (Japanese: サマーウォーズ , Hepburn: Samā Wōzu ) is a 2009 Japanese animated science fiction film directed by Mamoru Hosoda, animated by Madhouse and distributed by Warner Bros.
ANSWER: 'The Secret of Kells' is not a stop-motion animated film; it is a traditional animated fantasy film. It is a co-production among France, Belgium, and Ireland.

QUESTION: Who was Grigori Vasilyevich Aleksandrov?
CONTEXT: Ozzy Osbourne John Michael "Ozzy" Osbourne (born 3 December 1948) is an English singer, songwriter, and actor.\nFabrice Ziolkowski Fabrice Ziolkowski (born January 28, 1954) is a French-American screenwriter, director, producer, and voice director, best known for scripting the Oscar-nominated feature animation film "The Secret of Kells", writing the animated television series "Gawayn", and directing and producing the avant-garde documentary film "L.A.X.".\nFurama Hong Kong Hotel The site is now occupied by the AIA Central office building.
ANSWER: There is no information given on Grigori Vasilyevich Aleksandrov in the context.

QUESTION: Who starred in Bone Dry, alongside Lance Henriksen, in the role of Jared Nomak?
CONTEXT: Luke Goss He has appeared in numerous films including "Blade II" (2002) as Jared Nomak, "One Night with the King" (2006) as King Xerxes, "Tekken" (2010) as Steve Fox and "Interview with a Hitman" (2012) as Viktor.\nBlade II It was written by David S. Goyer, who also wrote the previous film, directed by Guillermo del Toro, and had Wesley Snipes returning as the lead character and producer.\nBone Dry The film starred Luke Goss and Lance Henriksen.
ANSWER: Luke Goss.

QUESTION: What campaign was the Battle of Harlem Heights fought during?
CONTEXT: Battle of Harlem Heights The Battle of Harlem Heights was fought during the New York and New Jersey campaign of the American Revolutionary War.\nBattle of White Plains The Battle of White Plains was a battle in the New York and New Jersey campaign of the American Revolutionary War fought on October 28, 1776, near White Plains, New York.\nOriskany Battlefield State Historic Site Oriskany Battlefield State Historic Site is a historic site in Oneida County, New York, United States that marks the Battle of Oriskany, fought in 1777 during the American Revolution, one of the bloodiest engagements of the war.
ANSWER:New York and New Jersey campaign of the American Revolutionary War.

QUESTION: Who is the actress that played a parent in the 2005 film Sky High?
CONTEXT: Hellboy (2004 film) Hellboy is a 2004 American supernatural superhero film directed by Guillermo del Toro and starring Ron Perlman, loosely based on the Dark Horse Comics graphic novel "" by Mike Mignola.\nCronos (film) Cronos is a 1993 Mexican horror drama film written and directed by Guillermo del Toro, starring veteran Argentinean actor Federico Luppi and American actor Ron Perlman.\nBlade II It was written by David S. Goyer, who also wrote the previous film, directed by Guillermo del Toro, and had Wesley Snipes returning as the lead character and producer.
ANSWER: There is no information given on the actress that played a parent in the 2005 film Sky High in the context.

QUESTION: What organization is Dana White the president of, where Forrest Griffin and Stephan Bonnar had their famous fight?
CONTEXT: Forrest Griffin vs. Stephan Bonnar Forrest Griffin vs. Stephan Bonnar is a duo of fights starting in the finals of The Ultimate Fighter 1 contest which received national acclaim, and was highly regarded among fans as one of the most exciting, greatest and memorable fights in the history of not only the Ultimate Fighting Championship (UFC) but of mixed martial arts (MMA) in general.\nUltimate Fighting Championship Dana White serves as the president of the UFC.\nUltimate Fighting Championship The Ultimate Fighting Championship (UFC) is an American mixed martial arts organization based in Las Vegas, Nevada, that is owned and operated by parent company WME–IMG.
ANSWER: Ultimate Fighting Championship (UFC).

QUESTION: Who directed the western film 'The Lusty Men'?
CONTEXT: The Lusty Men It was directed by Nicholas Ray and produced by Jerry Wald and Norman Krasna from a screenplay by David Dortort, Horace McCoy, Alfred Hayes, Andrew Solt, and Jerry Wald based on the novel by Claude Stanush.\nNicholas Ray Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director best known for the movie "Rebel Without a Cause."\nElia Kazan Elia Kazan (born Elias Kazantzoglou September 7, 1909 – September 28, 2003) was a Greek-American director, producer, writer and actor, described by "The New York Times" as "one of the most honored and influential directors in Broadway and Hollywood history".
ANSWER: Nicholas Ray.
###
QUESTION: {question}
CONTEXT: {context}
ANSWER:"""


sub_question_prompt = """<s> [INST] In a multi-hop information retrieval scenario, generate questions from a claim to identify and explore every key entity, particularly focusing on those that are not directly mentioned but implied. These questions should help in finding respective documents for each entity discussed in the claim. Ensure questions are direct and standalone, without referencing 'the claim' in their phrasing.
1. Direct Identification: Formulate questions that directly identify each entity mentioned in the claim to gather specific information about them.
2. Infer Hidden Entities: Create questions aimed at uncovering entities that are implied or indirectly referenced in the claim. These are often crucial for understanding the full context and connections.
3. Cover All Aspects: Ensure your questions collectively address all parts of the claim, enabling a comprehensive retrieval of information across all related entities.
4. Prioritize Clarity: Questions should be clear and to the point, facilitating the retrieval of relevant documents without ambiguity.[/INST]
###
CLAIM: The play by an award-winning American actor with an accomplished playwright premiere where it broke box office records and garnered critical acclaim was performed by the Portland Stage Company.
QUESTIONS: What is the name of the play performed by the Portland Stage Company that broke box office records and received critical acclaim?\nWho is the award-winning American actor involved in the play that premiered at the Portland Stage Company?

CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
QUESTIONS: Who is the musician that signed the band Godhead to their label?\nWhich American singer and songwriter released 'Second Hand Life' as their tenth solo studio album?

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
QUESTIONS: Which film did Mike McFarland voice in, Summer Wars or The Secret of Kells?\nIs 'The Secret of Kells' a stop-motion animated film, and what is its country of origin?\nIn what year was the film Summer Wars released?

CLAIM: The Lusty Men is a western directed by this man. He and the writer of America America are both actors and directors.
QUESTIONS: Who directed the western film 'The Lusty Men'?\nWho is the writer and director of 'America America'?

CLAIM: The actress that starred as a parent in Sky High (2005 film) is married to one of the subjects of film-critic Nigel Andrews' books. She also starred in a 1994 film about a pregnant woman who is stranded at a trading post during the American Civil War.
QUESTIONS: Who is the actress that played a parent in the 2005 film 'Sky High'?\nWhich film critic Nigel Andrews wrote a book about, and who is the subject of that book married to the actress from 'Sky High'?\nWhat is the name of the 1994 film in which the 'Sky High' actress starred as a pregnant woman stranded at a trading post during the American Civil War?

CLAIM: Dana White is the president of the organization that Forrest Griffin vs. Stephan Bonnar was thought to be one of the greatest bar fights in history.
QUESTIONS: What organization is Dana White the president of, where Forrest Griffin and Stephan Bonnar had their famous fight?\nWhich fight between Forrest Griffin and Stephan Bonnar is often referred to as one of the greatest bar fights in history?

CLAIM: Hey Monday was featured on the cover page of the Alternative Press music magazine for the '100 Bands You Need to Know in 2010' alongside The Cab discography and Never Shout Never.
QUESTIONS: How are The Cab discography and Never Shout Never related to Hey Monday in the context of music and publications?\nWhat is the significance of the '100 Bands You Need to Know in 2010' list where Hey Monday, The Cab, and Never Shout Never were mentioned?

CLAIM: A species related to Xanthosoma and Coronilla are both flowering plants.
QUESTIONS: Which species is related to Xanthosoma and is a flowering plant?\nIs Coronilla a flowering plant?

CLAIM: Grigori Vasilyevich Aleksandrov was a prominent director who worked frequently with the composer of Moya Moskva.
QUESTIONS: Who was Grigori Vasilyevich Aleksandrov?\nWhich composer worked frequently with Grigori Vasilyevich Aleksandrov and composed 'Moya Moskva'?
###
CLAIM: {claim}
QUESTIONS:"""


