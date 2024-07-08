import sys
from pathlib import Path
from db_operations import *
sys.path.append(str(Path("./cross_encoder").resolve()))
from create_sentences_dict import ClaimSentencePairsCreator
claim_sentence_creator = ClaimSentencePairsCreator(sql_db_path = 'data/wiki_wo_links.db', with_title = True)
titles = get_all_titles(claim_sentence_creator.cursor)
titles = [title[0] for title in titles]
claim_sentence_creator.populate_new_db(titles)