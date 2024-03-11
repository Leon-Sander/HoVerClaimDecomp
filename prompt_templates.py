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

decompose_add_context = """





"""