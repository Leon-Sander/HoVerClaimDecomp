import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM, create_chain, create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import sub_question_prompt, add_key_entities_refined_prompt, add_key_entities_and_change_prompt
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import EnhancedBaseClaimsOutputParser, SubQuestionsOutputParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

data = load_obj("data/qualitative_analysis_not_supported.json")
#qualitative_analysis_claims_10.json

for hop_count in data:
    for item in data[hop_count]:
        #item["previous_iteration_sentences"] = []
        item["claim_0"] = item["claim"]
        item["not_supported_counterpart"]["claim_0"] = item["not_supported_counterpart"]["claim"]
        #item["not_supported_counterpart"]["previous_iteration_sentences"] = []


model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = create_llm_pipeline(model_id=model_id, device_map="cuda:0", load_in_8bit=False, load_in_4bit=True)
print("llm loaded")
llm_question_generator = create_chain_with_postprocessor(create_prompt(template=sub_question_prompt), 
                        llm,
                        stop=["QUESTIONS:", "CLAIM:"], 
                        postprocessor=SubQuestionsOutputParser)


llm_base_enhancement = create_chain_with_postprocessor(create_prompt(template=add_key_entities_and_change_prompt), 
                    llm,
                    stop=["CLAIM:", "CONTEXT:", "REFINED CLAIM:"], 
                    postprocessor=EnhancedBaseClaimsOutputParser)
print("llm Loaded")
checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
print("Cross Enc Loaded")
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")
# start mit base claim und base retrieval
# 1. sub question generation 
# 2. sub question retrieval
# 3. sub question cross encoding, vergangene filtern
# 4. base claim_i cross encoding
# 5. base claim_i enhancement
# 6. base claim_i retrieval


def retrieve_doc_titles(query, task_description, vector_db, k=100):
    query = embedder.get_detailed_instruct(query=query, task_description=task_description)
    db_output = vector_db.similarity_search(query, k = k)
    retrieved = []
    for doc in db_output:
        retrieved.append(doc.metadata["title"])
    return retrieved

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

def get_top_sentences(sentences_per_question, sub_questions):
    max_sentences = sum([len(sentences) for sentences in sentences_per_question])
    top_sentences = []
    num_sentences_to_pick = min(60, max_sentences)
    for index in range(num_sentences_to_pick):
        for sentences_list in sentences_per_question:
            if len(top_sentences) >= num_sentences_to_pick:
                break
            if index < len(sentences_list):
                top_sentences.append(sentences_list[index])
    return top_sentences

#base retrieval
for hop_count in data:
    for item in data[hop_count]:
        item[f"retrieved_0"] = retrieve_doc_titles(item[f"claim_0"], embedder.task, vector_db, 100)
        item["not_supported_counterpart"][f"retrieved_0"] = retrieve_doc_titles(item["not_supported_counterpart"][f"claim_0"], embedder.task, vector_db, 100)

# hop_count +1 iterationen
for run_count in tqdm(range(5)):
    for hop_count in data:
        if run_count <= int(hop_count):
            # somit hop_count+1 retrievals, einmal mit base claim, n mal mit hop count claims
            for item in data[hop_count]:
                # sub question generation
                sub_questions = llm_question_generator.invoke({"claim": item[f"claim_{run_count}"]})
                item[f"sub_questions_{run_count}"] = sub_questions
                item[f"sub_question_retrieval_{run_count}"] = []
                
                #save_obj(data, "data/iterative_inspection.json")
                sentences_per_question = []
                for question in sub_questions:
                    retrieved = retrieve_doc_titles(query=question, task_description=embedder.question_task, vector_db=vector_db, k=100)
                    item[f"sub_question_retrieval_{run_count}"].append(retrieved)

                    sentences_sorted = get_sorted_sentences(cross_enc=cross_enc, claim=question, retrieved_titles = retrieved, do_filter=False, 
                                                            run_count=run_count, previous_iteration_sentences=[])

                    #if len(sentences_sorted) >= 20:
                    #    sentences_per_question.append(sentences_sorted[:20])
                    #else:
                    #    sentences_per_question.append(sentences_sorted)
                    sentences_per_question.append(sentences_sorted)
                top_sentences = get_top_sentences(sentences_per_question, sub_questions)
                item[f"sub_question_top_sentences_{run_count}"] = sentences_per_question
                item[f"top_base_sentences_{run_count}"] = top_sentences
                #item["previous_iteration_sentences"].extend(top_sentences)

                ##### enhancing base claim #####
                enhanced_base_claim = llm_base_enhancement.invoke({"claim": item[f"claim_{run_count}"], "context" : "\n".join(top_sentences)})
                item[f"claim_{run_count+1}"] = enhanced_base_claim
                ##### base retrieval i+1 #####
                retrieved = retrieve_doc_titles(query=item[f"claim_{run_count+1}"], task_description=embedder.task, vector_db=vector_db, k=100)
                item[f"retrieved_{run_count+1}"] = retrieved


                ###### Not Supported Counterpart
                # sub question generation
                sub_questions = llm_question_generator.invoke({"claim": item["not_supported_counterpart"][f"claim_{run_count}"]})
                item["not_supported_counterpart"][f"sub_questions_{run_count}"] = sub_questions
                item["not_supported_counterpart"][f"sub_question_retrieval_{run_count}"] = []
                
                sentences_per_question = []
                for question in sub_questions:
                    retrieved = retrieve_doc_titles(query=question, task_description=embedder.question_task, vector_db=vector_db, k=100)
                    item["not_supported_counterpart"][f"sub_question_retrieval_{run_count}"].append(retrieved)

                    sentences_sorted = get_sorted_sentences(cross_enc=cross_enc, claim=question, retrieved_titles = retrieved, do_filter=False, 
                                        run_count=run_count, previous_iteration_sentences=[])

                    #if len(sentences_sorted) > 20:
                    #    sentences_per_question.append(sentences_sorted[:20])
                    #else:
                    #    sentences_per_question.append(sentences_sorted)
                    sentences_per_question.append(sentences_sorted)
                
                top_sentences = get_top_sentences(sentences_per_question, sub_questions)

                item["not_supported_counterpart"][f"sub_question_top_sentences_{run_count}"] = sentences_per_question
                item["not_supported_counterpart"][f"top_base_sentences_{run_count}"] = top_sentences
                #item["not_supported_counterpart"]["previous_iteration_sentences"].extend(top_sentences)

                ##### enhancing base claim #####
                enhanced_base_claim = llm_base_enhancement.invoke({"claim": item["not_supported_counterpart"][f"claim_{run_count}"], "context" : "\n".join(top_sentences)})
                item["not_supported_counterpart"][f"claim_{run_count+1}"] = enhanced_base_claim
                # base retrieval i+1
                query = embedder.get_detailed_instruct(query=item["not_supported_counterpart"][f"claim_{run_count+1}"], task_description=embedder.task)
                db_output = vector_db.similarity_search(query, k = 100)
                retrieved = []
                for doc in db_output:
                    retrieved.append(doc.metadata["title"])
                item["not_supported_counterpart"][f"retrieved_{run_count+1}"] = retrieved
                #save_obj(data, "data/iterative_inspection.json")

save_obj(data, "data/iterative_not_supported_subquestions_no_filter_change_prompt_60.json")

