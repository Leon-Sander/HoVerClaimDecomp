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
CLAIM: The musician that signed Godhead (band) to their label and another singer are both American singer, songwriters. Second Hand Life is the tenth solo studio album by this other singer.
DECOMPOSED: An American singer and songwriter signed the band Godhead to their label.
"Second Hand Life" is the tenth solo studio album released by an American singer and songwriter.

CLAIM: Of the two films, Summer Wars and The Secret of Kells, the 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.
DECOMPOSED: The film Summer Wars might be a 2009 stop-motion animated film from japan.
The film The Secret of Kells might be a 2009 stop-motion animated film from japan.
The 2009 stop-motion animated film that Mike McFarland voiced in, is from Japan.

CLAIM: The actress that starred as a parent in Sky High (2005 film) is married to one of the subjects of film-critic Nigel Andrews' books. She also starred in a 1994 film about a pregnant woman who is stranded at a trading post during the American Civil War.
DECOMPOSED: An actress starred as a parent in the 2005 film Sky High.
The actress who starred as a parent in the 2005 film Sky High is married to one of the subjects of film-critic Nigel Andrews' books.
The actress who starred as a parent in the 2005 film Sky High also starred in a 1994 film about a pregnant woman who is stranded at a trading post during the American Civil War.

CLAIM: Dana White is the president of the organization that Forrest Griffin vs. Stephan Bonnar was thought to be one of the greatest bar fights in history.
DECOMPOSED: Dana White is the president of the organization where Forrest Griffin vs. Stephan Bonnar fought.
Forrest Griffin vs. Stephan Bonnar was thought to be one of the greatest bar fights in history.

CLAIM: The home of Sir John Cutts who died in 1615 is a house at a parish 18.7 miles from central London.
DECOMPOSED: The home of Sir John Cutts is a house at a parish 18.7 miles from central London.
Sir John Cutts died in 1615.

CLAIM: A species related to Xanthosoma and Coronilla are both flowering plants.
DECOMPOSED: A species related to Xanthosoma is a flowering plant.
Coronilla is a flowering plant.

CLAIM: Hey Monday was featured on the cover page of the Alternative Press music magazine for the '100 Bands You Need to Know in 2010' alongside The Cab discography and Never Shout Never.
DECOMPOSED: Hey Monday was featured on the cover page of the Alternative Press music magazine for the '100 Bands You Need to Know in 2010'.
The Cab discography was featured on the cover page of the Alternative Press music magazine for the '100 Bands You Need to Know in 2010'.
Never Shout Never was featured on the cover page of the Alternative Press music magazine for the '100 Bands You Need to Know in 2010'.

CLAIM: Grigori Vasilyevich Aleksandrov was a prominent director who worked frequently with the composer of Moya Moskva.
DECOMPOSED: Grigori Vasilyevich Aleksandrov was a prominent director.
The composer of Moya Moskva worked frequently with Grigori Vasilyevich Aleksandrov.

CLAIM: The Lusty Men is a western directed by this man. He and the writer of America America are both actors and directors.
DECOMPOSED: The Lusty Men is a western directed by an american man who is both an actor and director.
America America was written by a person who is both an actor and director.
###
CLAIM: {claim}
DECOMPOSED:"""


decompose_entity_based = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, decompose it into simpler, standalone claims that adhere to the following criteria:
1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight the main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable on its own, containing all necessary information for retrieval.
4. Avoid Adding Information: Do not include any information that is not present in the original multi-hop claim. Rely solely on the provided content.
5. Manage Redundancy: If multiple potential decomposed claims involve the same entity or fact, consolidate them into a single claim unless different aspects are critically relevant for separate retrieval paths.
6. Separate Clearly: Do not use listing and separate each decomposed claim with a newline.
[/INST]
###
CLAIM: The Lusty Men is a western film directed by Nicholas Ray. He and Elia Kazan, the writer of America America, are both actors and directors.
DECOMPOSED: The Lusty Men is a western film directed by Nicholas Ray.
Nicholas Ray is both an actor and a director.
Elia Kazan wrote the film America America.
Elia Kazan is both an actor and a director.

CLAIM: Sia heavily sampled the song 'Montagues and Capulets' by Sergei O.Prokofieff's grandfather, a Russian composer, for her song 'Taken for Granted.' The song is part of Prokofiev's ballet 'Romeo and Juliet,' which was first performed in 1940. The song 'Party Like a Russian' by Robbie Williams also samples 'Montagues and Capulets.'
DECOMPOSED: Sia heavily sampled the song 'Montagues and Capulets' for her song 'Taken for Granted.'
'Montagues and Capulets' was composed by Sergei O. Prokofieff's grandfather, a Russian composer.
The song 'Montagues and Capulets' is sampled in Robbie Williams song 'Party Like a Russian'.

CLAIM: The musician that signed Godhead (band) to their label, Marilyn Manson, and Joe Lynn Turner are both American singer, songwriters. Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
DECOMPOSED: Marilyn Manson signed the band Godhead to their label.
Marilyn Manson is an American singer and songwriter.
Joe Lynn Turner is an American singer and songwriter.
'Second Hand Life' is the tenth solo studio album by Joe Lynn Turner.

CLAIM: The Melville suburb in southern Hamilton, New Zealand contains streets named after war heroes including Douglas Melville Charles Guest. Douglas Melville Charles Guest, also known as Douglas Bader, was the subject of the 1956 British biographical film 'Reach for the Sky'. He was a Royal Air Force flying ace during the Second World War.
DECOMPOSED: The Melville suburb in southern Hamilton, New Zealand, features streets named after war heroes.
Douglas Melville Charles Guest, also known as Douglas Bader, was a Royal Air Force flying ace during the Second World War.
Douglas Bader was the subject of the 1956 British biographical film 'Reach for the Sky'.
One of the streets in Melville suburb is named after Douglas Melville Charles Guest.

CLAIM: Channel 8, which has Ong-Art Singlumpong for an executive director, has broadcast the American sports HBO World Championship Boxing that premiered in the early 1970s.
DECOMPOSED: Channel 8 is a television channel that has Ong-Art Singlumpong as an executive director.
Channel 8 has broadcast the American sports program HBO World Championship Boxing.
HBO World Championship Boxing premiered in the early 1970s.

CLAIM: The Landsman Kill, a body of water, is near the Dutchess County Fairgrounds and several historical landmarks on New York State Route 308, which intersects State Route 199 at Rock City in the state where billionaire Robert F. Kennedy was a junior senator.
DECOMPOSED: The Landsman Kill is a body of water near the Dutchess County Fairgrounds.
The Dutchess County Fairgrounds and several historical landmarks are located on New York State Route 308.
New York State Route 308 intersects with State Route 199 at Rock City.
Robert F. Kennedy, a billionaire, was a junior senator in New York state.

CLAIM: Kelly Preston, the actress that starred as a parent in Sky High (2005 film), is married to John Travolta, one of the subjects of film-critic Nigel Andrews' books. She also starred in Cheyenne Warrior, a 1994 American film about a pregnant woman who is stranded at a trading post during the American Civil War.
DECOMPOSED: Kelly Preston starred as a parent in the film Sky High (2005).
John Travolta is one of the subjects of film-critic Nigel Andrews' books.
Kelly Preston starred in Cheyenne Warrior, a 1994 American film about a pregnant woman stranded at a trading post during the American Civil War.

CLAIM: The founder of the International Grenfell Association, Sir Wilfred Thomason Grenfell, was a medical missionary to Newfoundland and Labrador. The city of Corner Brook, serviced by the Corner Brook Transit, is located in the province of Newfoundland and Labrador, Canada. The International Grenfell Association, founded by Grenfell, is headquartered in Mary's Harbour, which is also in Newfoundland and Labrador.
DECOMPOSED: Sir Wilfred Thomason Grenfell was the founder of the International Grenfell Association.
Corner Brook is serviced by the Corner Brook Transit.
The International Grenfell Association is headquartered in Mary's Harbour.

CLAIM: The 2016 romantic drama "Me Before You," directed by Thea Sharrock, features Sam Claflin as Will Traynor. Sam Claflin is known for his roles as Finnick Odair in "The Hunger Games" film series, Philip Swift in "Pirates of the Caribbean: On Stranger Tides," and Woody Harrelson's co-star in "The Lost Future."
DECOMPOSED: The 2016 romantic drama "Me Before You," is directed by Thea Sharrock.
"Me Before You" features Sam Claflin as Will Traynor.
Sam Claflin co-starred with Woody Harrelson in "The Lost Future."

###
CLAIM: {claim}
DECOMPOSED:"""

decompose_without_redundancy = """<s> [INST] Given a multi-hop claim for retrieval from a vector database, decompose it into simpler, standalone claims that adhere to the following criteria:
1. Be Precise and Contextualized: Incorporate specific details and context that are likely to match the vectors of relevant documents in the database.
2. Isolate Key Concepts and Relationships: Highlight the main subjects, actions, and notable attributes to ensure the vectors generated from these claims closely align with those of potential matching documents.
3. Ensure Standalone Validity: Each claim must be understandable and searchable on its own, containing all necessary information for retrieval.
4. Avoid Adding Information: Do not include any information that is not present in the original multi-hop claim. Rely solely on the provided content.
5. Manage Redundancy: If multiple potential decomposed claims involve the same entity or fact, consolidate them into a single claim unless different aspects are critically relevant for separate retrieval paths.
6. Separate Clearly: Do not use listing and separate each decomposed claim with a newline.
[/INST]
###
CLAIM: The Lusty Men is a western film directed by Nicholas Ray. He and Elia Kazan, the writer of America America, are both actors and directors.
DECOMPOSED: The Lusty Men is a western film directed by Nicholas Ray who is both an actor and a director.
Elia Kazan is both an actor and a director and wrote the film America America.

CLAIM: Sia heavily sampled the song 'Montagues and Capulets' by Sergei O.Prokofieff's grandfather, a Russian composer, for her song 'Taken for Granted.' The song is part of Prokofiev's ballet 'Romeo and Juliet,' which was first performed in 1940. The song 'Party Like a Russian' by Robbie Williams also samples 'Montagues and Capulets.'
DECOMPOSED: Sia heavily sampled the song 'Montagues and Capulets' for her song 'Taken for Granted.'
'Montagues and Capulets' was composed by Sergei O. Prokofieff's grandfather, a Russian composer.
The song 'Montagues and Capulets' is also sampled in Robbie Williams' song 'Party Like a Russian.

CLAIM: The musician that signed Godhead (band) to their label, Marilyn Manson, and Joe Lynn Turner are both American singer, songwriters. Second Hand Life Second Hand Life is the tenth solo studio album by Joe Lynn Turner.
DECOMPOSED: Marilyn Manson who is an American singer and songwriter, signed the band Godhead to their label.
'Second Hand Life' is the tenth solo studio album by Joe Lynn Turner who is an American singer and songwriter.

CLAIM: The Melville suburb in southern Hamilton, New Zealand contains streets named after war heroes including Douglas Melville Charles Guest. Douglas Melville Charles Guest, also known as Douglas Bader, was the subject of the 1956 British biographical film 'Reach for the Sky'. He was a Royal Air Force flying ace during the Second World War.
DECOMPOSED: The Melville suburb in southern Hamilton, New Zealand, features streets named after war heroes, including one named after Douglas Melville Charles Guest, also known as Douglas Bader.
Douglas Bader, a Royal Air Force flying ace during the Second World War, was the subject of the 1956 British biographical film 'Reach for the Sky'.

CLAIM: Channel 8, which has Ong-Art Singlumpong for an executive director, has broadcast the American sports HBO World Championship Boxing that premiered in the early 1970s.
DECOMPOSED: Channel 8 is a television channel that has Ong-Art Singlumpong as an executive director.
Channel 8 has broadcast the American sports program HBO World Championship Boxing which premiered in the early 1970s.

CLAIM: The Landsman Kill, a body of water, is near the Dutchess County Fairgrounds and several historical landmarks on New York State Route 308, which intersects State Route 199 at Rock City in the state where billionaire Robert F. Kennedy was a junior senator.
DECOMPOSED: The Landsman Kill, a body of water, is located near the Dutchess County Fairgrounds along New York State Route 308.
New York State Route 308, home to several historical landmarks, intersects with State Route 199 at Rock City.
Robert F. Kennedy, a billionaire and former junior senator, served in New York state.

CLAIM: Kelly Preston, the actress that starred as a parent in Sky High (2005 film), is married to John Travolta, one of the subjects of film-critic Nigel Andrews' books. She also starred in Cheyenne Warrior, a 1994 American film about a pregnant woman who is stranded at a trading post during the American Civil War.
DECOMPOSED: Kelly Preston starred as a parent in the film Sky High (2005) and in Cheyenne Warrior, a 1994 American film about a pregnant woman stranded at a trading post during the American Civil War.
John Travolta, Kelly Preston's husband, is one of the subjects of film-critic Nigel Andrews' books.

CLAIM: The founder of the International Grenfell Association, Sir Wilfred Thomason Grenfell, was a medical missionary to Newfoundland and Labrador. The city of Corner Brook, serviced by the Corner Brook Transit, is located in the province of Newfoundland and Labrador, Canada. The International Grenfell Association, founded by Grenfell, is headquartered in Mary's Harbour, which is also in Newfoundland and Labrador.
DECOMPOSED: Sir Wilfred Thomason Grenfell was the founder of the International Grenfell Association, which is headquartered in Mary's Harbour.
Corner Brook is serviced by the Corner Brook Transit.

CLAIM: The 2016 romantic drama "Me Before You," directed by Thea Sharrock, features Sam Claflin as Will Traynor. Sam Claflin is known for his roles as Finnick Odair in "The Hunger Games" film series, Philip Swift in "Pirates of the Caribbean: On Stranger Tides," and Woody Harrelson's co-star in "The Lost Future."
DECOMPOSED: The 2016 romantic drama "Me Before You," is directed by Thea Sharrock.
"Me Before You" features Sam Claflin who co-starred with Woody Harrelson in "The Lost Future."

###
CLAIM: {claim}
DECOMPOSED:"""

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




#################### STOPPING CRITERIA SECTION #############################
############################################################################
############################################################################
############################################################################
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


#################### CLAIM REFINEMENT SECTION ########################
######################################################################
######################################################################
######################################################################
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



#################### SUBQUESTION SECTION #############################
######################################################################
######################################################################
######################################################################
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


