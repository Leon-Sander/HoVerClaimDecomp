from CoreNLP_sentence_splitter.sentence_splitter_wrapper_for_CoreNLP_En import corenlp_ssplitter, CoreNLPTokenizer
from tqdm import tqdm
import random

import sys
from pathlib import Path
sys.path.append(str(Path("../").resolve()))
from db_operations import *
from utils import *
import sqlite3

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
        balanced_train_data.extend(positive_samples)
        balanced_train_data.extend(negative_samples)
    return balanced_train_data


class ClaimSentencePairsCreator:
    def __init__(self, sql_db_path = '/home/sander/code/thesis/hover/data/wiki_wo_links.db', with_title = True):
        conn, self.cursor = self._connect_to_db(sql_db_path)
        self.sentence_splitter = CoreNLPTokenizer()
        self.with_title = with_title
        print("Sentences will be created with title: " + str(self.with_title))
        
    def create_claim_sentence_pairs(self, claim :str, titles : list[str]):
        sentences_dict = self._create_sentences_dict(titles)
        claim_sentence_pairs = []
        for title, sentences in sentences_dict.items():
            for sentence in sentences:
                if self.with_title:
                    sentence = title + " " + sentence
                claim_sentence_pairs.append((claim, sentence))
        return claim_sentence_pairs
    
    def create_claim_sentence_pairs_from_sentences(self, claim :str, sentences : list[str]):
        claim_sentence_pairs = []
        for sentence in sentences:
            claim_sentence_pairs.append((claim, sentence))
        return claim_sentence_pairs
    
    def filter_sentences(self, claim_sentence_pairs, previous_iteration_sentences):
        """
        Filters the claim_sentence_pairs by removing the pairs that contain a sentence that was already used in the last iteration.
        """
        for claim, sentence in claim_sentence_pairs:
            if sentence in previous_iteration_sentences:
                claim_sentence_pairs.remove((claim, sentence))
        return claim_sentence_pairs

    def _create_sentences_dict(self, titles : list[str]):
        sentences_dict = {}
        for title in tqdm(titles):
            text = self._get_text_from_doc(self.cursor, title)
            sentences = corenlp_ssplitter(self.sentence_splitter, text)
            sentences_dict[title] = sentences
        return sentences_dict
    
    def populate_new_db(self, titles : list[str]):
        records = []
        error_titles = []
        for title in tqdm(titles):
            text = self._get_text_from_doc(self.cursor, title)
            try:
                sentences = corenlp_ssplitter(self.sentence_splitter, text)
            except Exception as e:
                print("Error while splitting sentences for title: " + title)
                print(e)
                error_titles.append(title)
                continue
            sentences_with_title = []
            for sentence in sentences:
                sentences_with_title.append(title + " " + sentence)

            joined_sentences = "\n\n".join(sentences)
            joined_sentences_with_title = "\n\n".join(sentences_with_title)

            records.append((title, text, joined_sentences, joined_sentences_with_title))

        with open("error_titles.txt", "w") as f:
            for title in error_titles:
                f.write(title + "\n")

        print("Inserting records into new db")
        conn, cursor = connect_to_db("/home/sander/code/thesis/hover/data/hover_with_sentences_splitted.db")
        cursor.executemany('''
            INSERT INTO documents (title, text, splitted_sentences, splitted_sentences_with_title)
            VALUES (?, ?, ?, ?);
        ''', records)
        
        conn.commit()
        conn.close()
    
    def _connect_to_db(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        return conn, cursor

    def _get_text_from_doc(self, cursor, title):
        doc = cursor.execute("SELECT text FROM documents WHERE id = ?",
                    (unicodedata.normalize("NFD", title),)).fetchall()[0][0]
        return doc
    
    def get_supporting_sentences(self, item, retrieval_key):
        sentences_dict = self._create_sentences_dict(item[retrieval_key])
        claim_and_supporting_facts = {}
        for title, sentence_level in item["supporting_facts"]:
            claim_and_supporting_facts[title] = []
        for title, sentence_level in item["supporting_facts"]:
            claim_and_supporting_facts[title].append(sentence_level)

        supported_sentences =  []

        for title, sentence_level_list in claim_and_supporting_facts.items():  
            if title in sentences_dict.keys():
                for i, sentence in enumerate(sentences_dict[title]):
                    if self.with_title:
                        sentence = title + " " + sentence
                    if i in sentence_level_list:
                        supported_sentences.append(sentence)

        return supported_sentences



class ClaimSentencePairsCreator_NewDb:
    def __init__(self, sql_db_path = '/home/sander/code/thesis/hover/data/hover_with_sentences_splitted.db', with_title = True):
        conn, self.cursor = connect_to_db(sql_db_path)
        self.with_title = with_title
        print("Sentences will be created with title: " + str(self.with_title))
        
    def create_claim_sentence_pairs(self, claim :str, titles : list[str]):
        sentences_list = self.get_splitted_sentences(titles)
        return [(claim, sentence) for sentence in sentences_list]

    def create_claim_sentence_pairs_from_sentences(self, claim :str, sentences : list[str]):
        claim_sentence_pairs = []
        for sentence in sentences:
            claim_sentence_pairs.append((claim, sentence))
        return claim_sentence_pairs
    
    def get_splitted_sentences(self, title_list):
        """
        Retrieve and split sentences associated with each title from the SQLite database in bulk.
        
        Args:
            title_list (list): List of titles for which to retrieve sentences.
        
        Returns:
            list: List of lists, each containing sentences for one title.
        """
        if self.with_title:
            column = "splitted_sentences_with_title"
        else:
            column = "splitted_sentences"
        
        try:
            placeholders = ','.join('?' for _ in title_list)
            query = f"SELECT {column} FROM documents WHERE title IN ({placeholders})"
            
            self.cursor.execute(query, title_list)
            results = self.cursor.fetchall()

            all_sentences = []
            for result in results:
                if result:
                    all_sentences.extend(result[0].split('\n\n'))

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")

        return all_sentences


    def get_supporting_sentences(self, item, retrieval_key):
        """
        Returns all supporting sentences that are within the retrieved documents
        """
        supporting_sentences = []
        for title, sentence_level in item["supporting_facts"]:
            if title in item[retrieval_key]:
                output = self.cursor.execute("SELECT splitted_sentences_with_title FROM documents WHERE title=?", (title,))
                sentences = output.fetchone()[0].split('\n\n')
                try:
                    sentence = sentences[sentence_level]
                except:
                    sentence = sentences[sentence_level -1]
                supporting_sentences.append(sentence)

        return supporting_sentences




if __name__ == "__main__":
    data_type = "dev"
    sentences_dict = create_sentences_dict(data_type)
    save_obj(sentences_dict, f"sentences_dict_{data_type}_no_title.json")
    #sentences_dict = load_obj("sentences_dict_{data_type}.json")
    train_data = create_claim_text_label_pairs(data_type, sentences_dict, balance_labels = False, with_title=False)
    save_obj(train_data, f"cross_encoder_claim_sentence_label_pairs_{data_type}.json")