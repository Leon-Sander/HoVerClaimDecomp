from CoreNLP_sentence_splitter.sentence_splitter_wrapper_for_CoreNLP_En import corenlp_ssplitter, CoreNLPTokenizer
from tqdm import tqdm
import random

import sys
from pathlib import Path
sys.path.append(str(Path("../").resolve()))
from db_operations import *
from utils import *

def create_sentences_dict(data_type):
    conn, cursor = connect_to_db()
    data = load_obj(f"/home/sander/code/thesis/hover/data/hover/hover_{data_type}_release_v1.1.json")
    tok = CoreNLPTokenizer()

    supporting_facts = set()
    for item in data:
        for title, _ in item["supporting_facts"]:
            supporting_facts.add(title)

    sentences_dict = {}
    for title in tqdm(supporting_facts):
        text = get_text_from_doc(cursor, title)
        sentences = corenlp_ssplitter(tok, text)
        sentences_dict[title] = sentences

    return sentences_dict
    

def create_claim_text_label_pairs(data_type, sentences_dict, balance_labels=True, with_title=True):
    claim_and_supporting_facts = {}
    data = load_obj(f"/home/sander/code/thesis/hover/data/hover/hover_{data_type}_release_v1.1.json")
    for item in data:
        claim_and_supporting_facts[item["claim"]] = {}
        for title, sentence_level in item["supporting_facts"]:
            claim_and_supporting_facts[item["claim"]][title] = []
        for title, sentence_level in item["supporting_facts"]:
            claim_and_supporting_facts[item["claim"]][title].append(sentence_level)

    balanced_train_data =  []
    for claim, title_pairs in claim_and_supporting_facts.items():
        positive_samples = []
        negative_samples = []
        for title, sentence_level_list in title_pairs.items():  
            for i, sentence in enumerate(sentences_dict[title]):
                if with_title:
                    sentence = title + " " + sentence
                if i in sentence_level_list:
                    positive_samples.append((claim, sentence, 1))
                else:
                    negative_samples.append((claim, sentence, 0))
        if balance_labels:
            num_positive = len(positive_samples)
            negative_samples = random.sample(negative_samples, min(num_positive, len(negative_samples)))
            positive_samples = random.sample(positive_samples, min(len(negative_samples), num_positive))
        # Füge die ausgewählten Beispiele zum Trainingsdatensatz hinzu
        balanced_train_data.extend(positive_samples)
        balanced_train_data.extend(negative_samples)
    return balanced_train_data

if __name__ == "__main__":
    data_type = "train"
    sentences_dict = create_sentences_dict(data_type)
    save_obj(sentences_dict, f"sentences_dict_{data_type}_no_title.json")
    #sentences_dict = load_obj("sentences_dict_{data_type}.json")
    train_data = create_claim_text_label_pairs(data_type, sentences_dict, balance_labels = True, with_title=False)
    save_obj(train_data, "cross_encoder_claim_sentence_label_pairs_train.json")