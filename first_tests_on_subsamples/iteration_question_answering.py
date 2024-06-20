import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM, create_chain, create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import add_context_prompt, sub_question_prompt, add_key_entities_refined_prompt, question_answering_prompt
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import EnhancedBaseClaimsOutputParser, SubQuestionsOutputParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

data = load_obj("data/iterative_true_false_eval_8x7b.json")
#"data/100_data_points.json"
#qualitative_analysis_claims_10.json

for hop_count in data:
    for item in data[hop_count]:
        item["previous_iteration_sentences"] = []
        item["claim_0"] = item["claim"]
        item["not_supported_counterpart"]["claim_0"] = item["not_supported_counterpart"]["claim"]
        item["not_supported_counterpart"]["previous_iteration_sentences"] = []

checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
print("Cross Enc Loaded")

model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = create_llm_pipeline(model_id=model_id, device_map="cuda:0", load_in_8bit=False, load_in_4bit=True)
print("llm loaded")
question_answering_chain = create_chain_with_postprocessor(create_prompt(template=question_answering_prompt), 
                        llm,
                        stop=["QUESTION:", "CONTEXT:", "ANSWER:"], 
                        postprocessor=None)


def get_sorted_sentences(cross_enc, claim, retrieved_titles, do_filter, run_count, previous_iteration_sentences):
    try:
        logging.info(f"Claim: {claim}")
        logging.info(f"Retrieved titles: {retrieved_titles}")
        question_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= claim, titles=retrieved_titles)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        # Handle the exception or re-raise it
        raise

    if do_filter and run_count > 0:
        question_sentence_pairs = cross_enc.claim_sentence_creator.filter_sentences(question_sentence_pairs, previous_iteration_sentences)
        # Die 20 Sätze aller vorherigen Iteration werden raus gefiltert -> noch mit einer threshold probieren
    cross_enc_output = cross_enc.predict(question_sentence_pairs, return_probabilities=True)
    cross_enc_output_sorted = sorted(cross_enc_output, key=lambda x: x[2], reverse=True)
    sentences_sorted = [prediction_tuple[1] for prediction_tuple in cross_enc_output_sorted]
    return sentences_sorted



# hop_count +1 iterationen
run_count = 0
for hop_count in data:
    if run_count <= int(hop_count):
        # somit hop_count+1 retrievals, einmal mit base claim, n mal mit hop count claims
        for item in data[hop_count]:
            questions_batch = []
            for question_index, question in enumerate(item[f"sub_questions_{run_count}"]):
                sentences_list = get_sorted_sentences(cross_enc, question, item[f"sub_question_retrieval_{run_count}"][question_index], do_filter=False, run_count=run_count, previous_iteration_sentences=[])[:60]
                context = "\n".join(sentences_list)
                questions_batch.append({"question": question, "context" : context})

            output = question_answering_chain.batch(questions_batch)
            item[f"question_answering_output_{run_count}"] = output

            questions_batch = []
            for question_index, question in enumerate(item["not_supported_counterpart"][f"sub_questions_{run_count}"]):
                sentences_list = get_sorted_sentences(cross_enc, question, item["not_supported_counterpart"][f"sub_question_retrieval_{run_count}"][question_index], do_filter=False, run_count=run_count, previous_iteration_sentences=[])[:60]
                context = "\n".join(sentences_list)
                questions_batch.append({"question": question, "context" : context})

            output = question_answering_chain.batch(questions_batch)
            item["not_supported_counterpart"][f"question_answering_output_{run_count}"] = output            


save_obj(data, "data/iterative_question_answering_test.json")

