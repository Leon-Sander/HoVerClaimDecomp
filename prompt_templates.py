decompose_6shot = """[INST] Given the multi-hop claim, decompose it into simpler, claims aimed at improving retrieval results.
DO NOT ADD ANY INFORMATION, ONLY USE INFORMATION GIVEN IN THE multi-hop claim.
Separate each decomposed claim with a newline. 
Do not use list formatting or numbers to enumerate the claims.

EXAMPLES:'''MULTI-HOP CLAIM: Ian Hunter from Mott the Hoople was born before the band member that released the album Dixie Chicken. That band member had the song 'Ride Like the Wind' dedicated to him.
DECOMPOSED CLAIMS: Ian Hunter was a member of Mott the Hoople.
Ian Hunter was born before the band member who released the album "Dixie Chicken."
The album "Dixie Chicken" was released by a band member of Mott the Hoople.
The song "Ride Like the Wind" was dedicated to the band member who released the album "Dixie Chicken."

MULTI-HOP CLAIM: The defending champion of the 1979 Romika Cup - Singles and another tennis player are both tennis players. The other tennis player was defending champion at the 1994 Waldbaum's Hamlet Cup.
DECOMPOSED CLAIMS: The 1979 Romika Cup - Singles was a tennis event with a defending champion.
The 1994 Waldbaum's Hamlet Cup was a tennis event with a defending champion.

MULTI-HOP CLAIM: The movie '100 Million BC' continues the urban myth of an experiment allegedly carried out at the Philadelphia Naval Shipyard.
DECOMPOSED CLAIMS: The movie '100 Million BC' involves the urban myth of an experiment.
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
'''

MULTI-HOP CLAIM: {claim}
DECOMPOSED CLAIMS: [/INST]"""


enrich_template ="""[INST] Given the multi-hop claim and context information, enrich the claim with key facts, entities, or relationships mentioned
in the context. BUT BE AWARE THAT THE CONTEXT MAX CONTAIN DOCUMENTS WHICH HAVE ZERO ASSOCIATION WITH THE CLAIM AND SHOULD BE IGNORED.
ONLY USE RELEVANT INFORMATION. 

DO NOT MAKE UP ANY INFORMATION OTHER THAN WHAT IS GIVEN TO YOU BY THE CONTEXT AND MULTI-HOP CLAIM.

EXAMPLES:'''MULTI-HOP CLAIM: Ian Hunter from Mott the Hoople was born before the band member that released the album Dixie Chicken. 
That band member had the song 'Ride Like the Wind' dedicated to him.
CONTEXT: TITLE: 'Ian Hunter (singer)', DOCUMENT: 'Ian Hunter Patterson (born 3 June 1939), known as Ian Hunter, is a British singer-songwriter who is best known as the lead singer of the English rock band Mott the Hoople, from its inception in 1969 to its dissolution in 1974, and at the time of its 2009 and 2013 reunions. Hunter was a musician and songwriter before joining Mott the Hoople, and continued in this vein after he left the band. He embarked on a solo career despite ill health and disillusionment with commercial success, and often worked in collaboration with Mick Ronson, David Bowie's sideman and arranger from the "Ziggy Stardust and the Spiders from Mars" period.',
TITLE: 'Ride Like the Wind', DOCUMENT: 'Ride Like the Wind" is a song written and recorded by American singer-songwriter Christopher Cross. It was released in February 1980 as the lead single from his Grammy-winning self-titled debut album. It reached number 2 on the US charts for four consecutive weeks, behind Blondie's "Call Me". On the album's inner sleeve, Christopher Cross dedicated this song to Lowell George, formerly of the band Little Feat, who had died in 1979. It features backing vocals by Michael McDonald and a guitar solo by Cross.'
TITLE: 'Steve Hunter', DOCUMENT: 'Stephen John Hunter (born June 14, 1948) is an American guitarist, primarily a session player best known for his collaborations with Lou Reed and Alice Cooper. He has been often called "The Deacon". Hunter first played with Mitch Ryder's Detroit, beginning a long association with record producer Bob Ezrin who has said Steve Hunter has contributed so much to rock music in general that he truly deserves the designation of "Guitar Hero". Steve Hunter has played some of the greatest riffs in rock history - that first slamming solo that rings in Aerosmith's "Train Kept A Rollin'", the acoustic intro on Peter Gabriel's "Solsbury Hill" and he wrote the legendary intro interlude that made Lou Reed's live version of "Sweet Jane" a first gold record (the "Rock 'N' Roll Animal" live set).'

ENRICHED MULTI-HOP CLAIM: Ian Hunter, born 3 June 1939, known as the lead singer of Mott the Hoople, was born before Lowell George. 
The song 'Ride Like the Wind,' was dedicated to Lowell George, who was a member of the band Little Feat that released the album 'Dixie Chicken.'
Lowell George passed away in 1979.
'''

The enriched multi-hop claim is more specific, and has clear statements that maintain the original claim's essence while ensuring that each component is directly tied to verifiable facts.

CONTEXT: {context}
MULTI-HOP CLAIM: {claim}
ENRICHED MULTI-HOP CLAIM:[/INST]"""

enrich_template2 ="""[INST] Given the multi-hop claim and context information, enrich the claim with key facts, entities, or relationships mentioned
in the context. The context consists of documents retrieved based on the multi-hop claim, which may or may not contain all the relevant information. 
To aid in assessing the relevance of each document to the claim, document TITLES are provided alongside their content. Carefully consider:

- FOCUS ONLY on information that directly supports or refutes the claim.
- IGNORE DOCUMENTS THAT ARE UNRELATED to the claim.
- DO NOT MAKE UP ANY INFORMATION. Use ONLY the details provided by the context and the claim.
- SYNTHESIZE key information from MULTIPLE RELEVANT DOCUMENTS carefully, if applicable.

EXAMPLE:
MULTI-HOP CLAIM: Ian Hunter from Mott the Hoople was born before the band member that released the album Dixie Chicken. That band member had the song 'Ride Like the Wind' dedicated to him.
CONTEXT: TITLE: 'Ian Hunter (singer)', DOCUMENT: 'Ian Hunter Patterson (born 3 June 1939), known as Ian Hunter, is a British singer-songwriter who is best known as the lead singer of the English rock band Mott the Hoople, from its inception in 1969 to its dissolution in 1974, and at the time of its 2009 and 2013 reunions. Hunter was a musician and songwriter before joining Mott the Hoople, and continued in this vein after he left the band. He embarked on a solo career despite ill health and disillusionment with commercial success, and often worked in collaboration with Mick Ronson, David Bowie's sideman and arranger from the "Ziggy Stardust and the Spiders from Mars" period.',
TITLE: 'Ride Like the Wind', DOCUMENT: 'Ride Like the Wind" is a song written and recorded by American singer-songwriter Christopher Cross. It was released in February 1980 as the lead single from his Grammy-winning self-titled debut album. It reached number 2 on the US charts for four consecutive weeks, behind Blondie's "Call Me". On the album's inner sleeve, Christopher Cross dedicated this song to Lowell George, formerly of the band Little Feat, who had died in 1979. It features backing vocals by Michael McDonald and a guitar solo by Cross.'
TITLE: 'Steve Hunter', DOCUMENT: 'Stephen John Hunter (born June 14, 1948) is an American guitarist, primarily a session player best known for his collaborations with Lou Reed and Alice Cooper. He has been often called "The Deacon". Hunter first played with Mitch Ryder's Detroit, beginning a long association with record producer Bob Ezrin who has said Steve Hunter has contributed so much to rock music in general that he truly deserves the designation of "Guitar Hero". Steve Hunter has played some of the greatest riffs in rock history - that first slamming solo that rings in Aerosmith's "Train Kept A Rollin'", the acoustic intro on Peter Gabriel's "Solsbury Hill" and he wrote the legendary intro interlude that made Lou Reed's live version of "Sweet Jane" a first gold record (the "Rock 'N' Roll Animal" live set).'
ENRICHED MULTI-HOP CLAIM: Ian Hunter, born 3 June 1939 and lead singer of Mott the Hoople, was born before Lowell George, member of Little Feat that released 'Dixie Chicken'. 'Ride Like the Wind' was dedicated to Lowell George, who passed away in 1979.

Your task is to process the given CONTEXT and MULTI-HOP CLAIM to produce an ENRICHED MULTI-HOP CLAIM that incorporates only relevant and verifiable facts from the context, adhering to the instructions above.

CONTEXT: {context}
MULTI-HOP CLAIM: {claim}
ENRICHED MULTI-HOP CLAIM:[/INST]"""


decompose_enriched = """[INST] Given a multi-hop claim and the same multi-hop claim with enriched information for retrieval from a vector database, your goal is to decompose the enriched claim down into simpler, standalone claims that are optimized to retrieve documents to verify the multi-hop claim.
Each decomposed claim should:

1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable without reference to other claims, containing all necessary information for retrieval.
4. Optimize for Vector Similarity: Use language and terms that are expected to generate vectors similar to those of the target documents, enhancing the likelihood of a match.
5. NOT ADD OR MAKE UP ANY INFORMATION, WHICH IS NOT PRESENT IN THE given multi-hop claims.
6. Be separated with a newline. 

Example:'''CLAIM: The director of comedy "Flirting with Disaster" and Tony Gatlif are not from the same country.     
ENRICHED CLAIM: The director of the 1996 American black comedy film "Flirting with Disaster," David O. Russell, is not from the same country as Tony Gatlif. Tony Gatlif is a French film director of Romani ethnicity, while David O. Russell is an American film director.
DECOMPOSED CLAIMS: David O. Russell directed the 1996 American black comedy film "Flirting with Disaster."
David O. Russell is an American film director.
Tony Gatlif is a French film director of Romani ethnicity.'''

CLAIM: {claim}
ENRICHED CLAIM: {enriched_claim}
DECOMPOSED CLAIMS:[/INST]"""


decompose_9shot = """[INST] Given a multi-hop claim for retrieval from a vector database, your goal is to decompose it down into simpler, standalone claims.
These claims should be formulated to optimize for cosine similarity calculations,
focusing on capturing the essence of the information in a way that aligns with likely vectors representing similar concepts. Each decomposed claim should:

1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable without reference to other claims, containing all necessary information for retrieval.
4. Optimize for Vector Similarity: Use language and terms that are expected to generate vectors similar to those of the target documents, enhancing the likelihood of a match.
5. NOT ADD ANY INFORMATION, WHICH IS NOT PRESENT IN THE multi-hop claim, ONLY USE INFORMATION GIVEN IN THE multi-hop claim.
6. Be separated with a newline.

EXAMPLES:'''MULTI-HOP CLAIM: Ian Hunter from Mott the Hoople was born before the band member that released the album Dixie Chicken. That band member had the song 'Ride Like the Wind' dedicated to him.
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
'''

MULTI-HOP CLAIM: {claim}
DECOMPOSED CLAIMS:[/INST]"""

decompose_9shot_instruct = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, your goal is to decompose it down into simpler, standalone claims.
Each decomposed claim should:

1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable without reference to other claims, containing all necessary information for retrieval.
4. Optimize for Vector Similarity: Use language and terms that are expected to generate vectors similar to those of the target documents, enhancing the likelihood of a match.
5. NOT ADD ANY INFORMATION, WHICH IS NOT PRESENT IN THE multi-hop claim, ONLY USE INFORMATION GIVEN IN THE multi-hop claim.
6. Be separated with a newline.

Here are nine examples, followed by the MULTI-HOP CLAIM you have to decompose.
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


decompose_6_2 = """<s>
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

MULTI-HOP CLAIM: The actor who played Colonel Sherman T. Potter in 'M*A*S*H' played Seldom Jackson in 'Kentucky Jones'.
DECOMPOSED CLAIMS: An actor played Colonel Sherman T. Potter in 'M*A*S*H'.
An actor played Seldom Jackson in 'Kentucky Jones.'

MULTI-HOP CLAIM: The producer of Future Profits is a male. So is Park Jimin's and Jae's co-host of After School Club.
DECOMPOSED CLAIMS: The producer of Future Profits is a male.
Park Jimin and Jae co-host After School Club.
The co-host of After School Club is a male.

MULTI-HOP CLAIM: {claim}
DECOMPOSED CLAIMS:"""


decompose_6_3 = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, your goal is to decompose it down into simpler, standalone claims.
Each decomposed claim should:

1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable without reference to other claims, containing all necessary information for retrieval.
4. Optimize for Vector Similarity: Use language and terms that are expected to generate vectors similar to those of the target documents, enhancing the likelihood of a match.
5. NOT ADD ANY INFORMATION, WHICH IS NOT PRESENT IN THE multi-hop claim, ONLY USE INFORMATION GIVEN IN THE multi-hop claim.
6. Be separated with a newline.

Here are nine examples, followed by the MULTI-HOP CLAIM you have to decompose.
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

MULTI-HOP CLAIM: The actor who played Colonel Sherman T. Potter in 'M*A*S*H' played Seldom Jackson in 'Kentucky Jones'.
DECOMPOSED CLAIMS: An actor played Colonel Sherman T. Potter in 'M*A*S*H'.
An actor played Seldom Jackson in 'Kentucky Jones.'

MULTI-HOP CLAIM: The producer of Future Profits is a male. So is Park Jimin's and Jae's co-host of After School Club.
DECOMPOSED CLAIMS: The producer of Future Profits is a male.
Park Jimin and Jae co-host After School Club.
The co-host of After School Club is a male.
###

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