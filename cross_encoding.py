from utils import load_obj
from db_operations import *
from sentence_transformers import CrossEncoder
from collections import defaultdict
cross_enc = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

def get_sentence_similarity(sentences_list):
    scores = cross_enc.predict(sentences_list)
    return scores

def get_top_k_sentences(sentences_list, scores, k):
    grouped_by_claim = defaultdict(list)
    for (claim, text), score in zip(sentences_list,scores):
        grouped_by_claim[claim].append((text, score))

    #top_texts_by_claim = {claim: sorted(texts, key=lambda x: x[1], reverse=True)[:k] for claim, texts in grouped_by_claim.items()}
    top_texts_by_claim = [(claim, '\n'.join(text for text, _ in sorted(texts, key=lambda x: x[1], reverse=True)[:k]))
                        for claim, texts in grouped_by_claim.items()]
    return top_texts_by_claim

conn, cursor = connect_to_db()
data = load_obj("/home/sander/code/thesis/hover/leon/qualitative_analysis/decomposed_qualitative_analysis_retrieval_10_9_noinstruct.json")
import unicodedata
#import spacy
from tqdm import tqdm
from CoreNLP_sentence_splitter.sentence_splitter_wrapper_for_CoreNLP_En import corenlp_ssplitter, CoreNLPTokenizer
tok = CoreNLPTokenizer()

#nlp = spacy.load("en_core_web_sm")
print("loaded")
#nltk.download('punkt')
#from nltk.tokenize import sent_tokenize

def postprocess_decomposed_claims(decomposed_claims : str) -> list[str]:
    output = []
    decomposed_claims = decomposed_claims.replace("\n\n", "\n").split("\n")
    for claim in decomposed_claims:
        if claim == "":
            continue
        output.append(claim.lstrip())
    return output

def get_text_from_doc(cursor, title):
    doc = cursor.execute("SELECT text FROM documents WHERE id = ?",
                   (unicodedata.normalize("NFD", title),)).fetchall()[0][0]
    return doc

for hop in data:
    for supported in data[hop]:
        for item in tqdm(data[hop][supported]):
            sentences_list = []
            decomposed_claims = postprocess_decomposed_claims(item["decomposed_claims"])
            for i, decomposed_claim in enumerate(decomposed_claims):
                end = (i +1) *100 -1
                start = end - 99
                retrieved_docs = item["retrieved"][start:end+1]
                for doc in retrieved_docs:
                    text = get_text_from_doc(cursor, doc)
                    #sentences =[sent.text for sent in nlp(text).sents] 
                    sentences = corenlp_ssplitter(tok, text)
                    sentences = [(decomposed_claim, sentence) for sentence in sentences]
                    #sentences = sent_tokenize(text)
                    sentences_list.extend(sentences)
            scores = get_sentence_similarity(sentences_list)
            item["claims_with_sentences"] = get_top_k_sentences(sentences_list, scores, 5)