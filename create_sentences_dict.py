from CoreNLP_sentence_splitter.sentence_splitter_wrapper_for_CoreNLP_En import corenlp_ssplitter, CoreNLPTokenizer
from utils import *
from db_operations import *
from tqdm import tqdm

def create_sentences_dict():
    conn, cursor = connect_to_db()
    data = load_obj("/home/sander/code/thesis/hover/data/hover/hover_train_release_v1.1.json")
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
    #save_obj(sentences_dict, "sentences_dict_train.json")

def create_claim_text_label_pairs():
    claim_and_supporting_facts = {}
    data = load_obj("/home/sander/code/thesis/hover/data/hover/hover_train_release_v1.1.json")
    sentences_dict = load_obj("sentences_dict_train.json")
    for item in data:
        claim_and_supporting_facts[item["claim"]] = {}
        for title, sentence_level in item["supporting_facts"]:
            claim_and_supporting_facts[item["claim"]][title] = []
        for title, sentence_level in item["supporting_facts"]:
            claim_and_supporting_facts[item["claim"]][title].append(sentence_level)

    train_data =  []
    for claim, title_pairs in claim_and_supporting_facts.items():
        for title, sentence_level_list in title_pairs.items():  
            for i, sentence in enumerate(sentences_dict[title]):
                if i in sentence_level_list:
                    label = 1
                else:
                    label = 0
                train_data.append((claim,sentence,label))
    return train_data

if __name__ == "__main__":
    train_data = create_claim_text_label_pairs()
    save_obj(train_data, "cross_encoder_claim_sentence_label_pairs.json")