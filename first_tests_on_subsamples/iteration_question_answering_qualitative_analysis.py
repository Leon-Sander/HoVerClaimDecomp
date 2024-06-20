import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import sub_question_prompt, question_answering_prompt, add_key_entities_based_on_question_answering_prompt
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import EnhancedBaseClaimsOutputParser, SubQuestionsOutputParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def get_sorted_sentences(cross_enc, claim, retrieved_titles, do_filter, run_count, previous_iteration_sentences):
    try:
        logging.info(f"Claim: {claim}")
        logging.info(f"Retrieved titles: {retrieved_titles}")
        question_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= claim, titles=retrieved_titles)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

    if do_filter and run_count > 0:
        question_sentence_pairs = cross_enc.claim_sentence_creator.filter_sentences(question_sentence_pairs, previous_iteration_sentences)
    cross_enc_output = cross_enc.predict(question_sentence_pairs, return_probabilities=True)
    cross_enc_output_sorted = sorted(cross_enc_output, key=lambda x: x[2], reverse=True)
    sentences_sorted = [prediction_tuple[1] for prediction_tuple in cross_enc_output_sorted]
    return sentences_sorted

def prepare_data(data):
    for hop_count in data:
        for key in data[hop_count]:
            for item in data[hop_count][key]:
                item["claim_0"] = item["claim"]
    return data

def retrieve_doc_titles(query, task_description, vector_db, embedder, k=100):
    query = embedder.get_detailed_instruct(query=query, task_description=task_description)
    db_output = vector_db.similarity_search(query, k = k)
    retrieved = []
    for doc in db_output:
        retrieved.append(doc.metadata["title"])
    return retrieved

def final_retrieval(data, final_run_count):
    for hop_count in data:
        for key in data[hop_count]:
            for item in data[hop_count][key]:
                retrieved = retrieve_doc_titles(item[f"claim_{final_run_count+1}"], embedder.task, vector_db, embedder, k=100)
                item[f"retrieved_{final_run_count+1}"] = retrieved
    return data

if __name__ == "__main__":
    
    data = load_obj("data/qualitative_analysis_claims_10_base.json")
    data = prepare_data(data)

    checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
    cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
    print("Cross Enc Loaded")

    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = create_llm_pipeline(model_id=model_id, device_map="cuda:0", load_in_8bit=False, load_in_4bit=True)
    print("llm loaded")

    question_generator_chain = create_chain_with_postprocessor(create_prompt(template=sub_question_prompt), 
                            llm,
                            stop=["QUESTIONS:", "CLAIM:"], 
                            postprocessor=SubQuestionsOutputParser)

    question_answering_chain = create_chain_with_postprocessor(create_prompt(template=question_answering_prompt), 
                            llm,
                            stop=["QUESTION:", "CONTEXT:", "ANSWER:"], 
                            postprocessor=EnhancedBaseClaimsOutputParser)

    base_refinement_chain = create_chain_with_postprocessor(create_prompt(template=add_key_entities_based_on_question_answering_prompt), 
                        llm,
                        stop=["CLAIM:", "CONTEXT:", "REFINED CLAIM:"], 
                        postprocessor=EnhancedBaseClaimsOutputParser)

    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
    vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
    print("Embeder and Vector Db loaded")

    for run_count in tqdm(range(5)):
        for hop_count in data:
            for key in data[hop_count]:
                if run_count <= int(hop_count):
                    #### base retrieval ####
                    for item in data[hop_count][key]:
                        retrieved = retrieve_doc_titles(item[f"claim_{run_count}"], embedder.task, vector_db, embedder, k=100)
                        item[f"retrieved_{run_count}"] = retrieved

                    # sub question generation
                    claims_batch = []
                    for item in data[hop_count][key]:
                        claims_batch.append({"claim": item[f"claim_{run_count}"]})

                    sub_questions_batch_output = question_generator_chain.batch(claims_batch)
                    for sub_questions_index, sub_questions in enumerate(sub_questions_batch_output):
                        data[hop_count][key][sub_questions_index][f"sub_questions_{run_count}"] = sub_questions


                    for item in data[hop_count][key]:
                        item[f"sub_question_retrieval_{run_count}"] = []
                        sentences_per_question = []
                        for question in item[f"sub_questions_{run_count}"]:
                            retrieved = retrieve_doc_titles(query=question, task_description=embedder.question_task, vector_db=vector_db, embedder=embedder, k=100)
                            item[f"sub_question_retrieval_{run_count}"].append(retrieved)
                        

                    ##### question answering #####
                    for item in data[hop_count][key]:
                        questions_batch = []
                        for question_index, question in enumerate(item[f"sub_questions_{run_count}"]):
                            sentences_list = get_sorted_sentences(cross_enc, question, item[f"sub_question_retrieval_{run_count}"][question_index], do_filter=False, run_count=run_count, previous_iteration_sentences=[])[:60]
                            context = "\n".join(sentences_list)
                            questions_batch.append({"question": question, "context" : context})

                        output = question_answering_chain.batch(questions_batch)
                        item[f"question_answering_output_{run_count}"] = output

                    ##### base refinement #####
                    claims_batch = []
                    for item in data[hop_count][key]:
                        context = ""
                        for i, sub_question in enumerate(item[f"sub_questions_{run_count}"]):
                            answer = item[f"question_answering_output_{run_count}"][i]
                            context += f"{sub_question}\n{answer}\n"
                        context = context.rstrip("\n")
                        claims_batch.append({"claim": item[f"claim_{run_count}"], "context": context})


                    refinement_batch_output = base_refinement_chain.batch(claims_batch)
                    for refined_claim_index, refined_claim in enumerate(refinement_batch_output):
                        data[hop_count][key][refined_claim_index][f"claim_{run_count+1}"] = refined_claim



    #data = final_retrieval(data, run_count)
    save_obj(data, "data/iterative_qualitative_analysis_question_answering_60.json")

