import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM
from custom_mistral_embedder import CustomMistralEmbedder
from concurrent.futures import ThreadPoolExecutor
from utils import load_obj, load_vectordb, save_obj
from create_sentences_dict import ClaimSentencePairsCreator_NewDb
import traceback
from tqdm import tqdm
import logging
import time
import torch
import gc

def log_time(func):
    """ Decorator to log the execution time of a function. """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def process_claims(model, batch_input, task_type):
    if task_type == "subquestion_generation":
        output = model.generate_subquestions(batch_input)
    elif task_type == "claim_refinement":
        output = model.generate_claim_refinement(batch_input)
    elif task_type == "decomposition":
        output = model.generate_decomposition(batch_input)
    elif task_type == "cross_enc":
        output = model.predict_parallel(batch_input, return_probabilities=True)
    else:
        raise ValueError("Invalid model task type")
    return output


def run_in_parallel(claims, task_type, llm1, llm2):
    if len(claims) < 2:
        return process_claims(llm1, claims, task_type)
    mid_point = len(claims) // 2
    first_half = claims[:mid_point]
    second_half = claims[mid_point:]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_claims, llm1, first_half, task_type),
            executor.submit(process_claims, llm2, second_half, task_type)
        ]
        results = [future.result() for future in futures]
    
    combined_results = results[0] + results[1]
    return combined_results


@log_time
def claim_refinement(data, run_count):
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm1 = TransformerLLM(model_id, device_map="cuda:0")
    llm2 = TransformerLLM(model_id, device_map="cuda:1")
    print("llm Loaded")
    try:
        for hop_count in data:
            if run_count <= int(hop_count) - 1:
                for key in data[hop_count]:
                    for j in tqdm(range(0, len(data[hop_count][key]), 10), desc=f'Refining Base Claims {run_count}, {hop_count}, {key}'):
                        base_claims_context = []
                        indices = []
                        # Collect claims and sentences for refinement only if they haven't been refined for the next run_count
                        for k in range(10):
                            idx = j + k
                            if idx < len(data[hop_count][key]):
                                if f"claim_{run_count + 1}" in data[hop_count][key][idx]:
                                    continue
                                claim = data[hop_count][key][idx][f"claim_{run_count}"]
                                top_sentences = data[hop_count][key][idx][f"top_sentences_{run_count}"]
                                top_sentences = "\n".join(top_sentences)
                                base_claims_context.append((claim, top_sentences))
                                indices.append(idx)
                        
                        if base_claims_context:  # Check if there are any claims to refine
                            enhanced_claims = run_in_parallel(claims=base_claims_context, task_type="claim_refinement", llm1=llm1, llm2=llm2)
                            
                            # Update enhanced
                            for l, enhanced_claim in enumerate(enhanced_claims):
                                data[hop_count][key][indices[l]][f"claim_{run_count + 1}"] = enhanced_claim
                            del enhanced_claims
                        else:
                            continue
                        torch.cuda.empty_cache()
                        gc.collect()
    except Exception as e:
        save_obj(data, "data/ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved under data/ZWISCHENERGEBNIS.json')
        print(traceback.format_exc())

    del llm1
    del llm2
    torch.cuda.empty_cache()
    gc.collect()
    return data

@log_time
def generate_decomposition(data, run_count):
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm1 = TransformerLLM(model_id, device_map="cuda:0")
    llm2 = TransformerLLM(model_id, device_map="cuda:1")
    print("llm Loaded")
    try:
        for hop_count in data:
            if run_count <= int(hop_count) - 1:
                for key in data[hop_count]:
                    all_claims = []
                    indices = []
                    for idx, item in enumerate(data[hop_count][key]):
                        if f"decomposed_claims_{run_count}" not in item:
                            all_claims.append(item[f"claim_{run_count}"])
                            indices.append(idx) 

                    if not all_claims:
                        print(f"No decompositions to generate for {run_count}, {hop_count}, {key}")
                        continue

                    for i in tqdm(range(0, len(all_claims), 50), desc=f'Decomposition {run_count}, {hop_count}, {key}'):
                        batch_claims = all_claims[i:i + 50]
                        batch_indices = indices[i:i + 50]
                        decomposed_batch = run_in_parallel(claims=batch_claims, task_type="decomposition",llm1=llm1, llm2=llm2)
                        
                        for idx, decomposed_claims in enumerate(decomposed_batch):
                            original_idx = batch_indices[idx]
                            data[hop_count][key][original_idx][f"decomposed_claims_{run_count}"] = decomposed_claims
                            data[hop_count][key][original_idx][f"decomposed_claims_retrieval_{run_count}"] = []
                        del decomposed_batch

                    torch.cuda.empty_cache()
                    gc.collect()
    except Exception as e:
        save_obj(data, "data/ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved under data/ZWISCHENERGEBNIS.json')
        print(traceback.format_exc())
    del llm1
    del llm2
    torch.cuda.empty_cache()
    gc.collect()
    return data

@log_time
def generate_subquestions(data, run_count):
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm1 = TransformerLLM(model_id, device_map="cuda:0")
    llm2 = TransformerLLM(model_id, device_map="cuda:1")
    print("llm Loaded")
    try:
        for hop_count in data:
            if run_count <= int(hop_count) - 1:
                for key in data[hop_count]:
                    all_claims = []
                    indices = []
                    for idx, item in enumerate(data[hop_count][key]):
                        if f"sub_questions_{run_count}" in item:
                            if item[f"sub_questions_{run_count}"]:
                                continue

                        all_claims.append(item[f"claim_{run_count}"])
                        indices.append(idx)

                    if not all_claims:
                        print(f"No subquestions to generate for {run_count}, {hop_count}, {key}")
                        continue

                    for i in tqdm(range(0, len(all_claims), 50), desc=f'Subquestion Generation {run_count}, {hop_count}, {key}'):
                        batch_claims = all_claims[i:i + 50]
                        batch_indices = indices[i:i + 50]
                        sub_questions_batch = run_in_parallel(claims=batch_claims, task_type="subquestion_generation",llm1=llm1, llm2=llm2)
                        
                        # Collect sub-questions and prepare base claim enhancement context
                        for idx, sub_questions in enumerate(sub_questions_batch):
                            original_idx = batch_indices[idx]
                            data[hop_count][key][original_idx][f"sub_questions_{run_count}"] = sub_questions
                            data[hop_count][key][original_idx][f"sub_question_retrieval_{run_count}"] = []
                        del sub_questions_batch

                    torch.cuda.empty_cache()
                    gc.collect()
    except Exception as e:
        save_obj(data, "data/ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved under data/ZWISCHENERGEBNIS.json')
        print(traceback.format_exc())
    del llm1
    del llm2
    torch.cuda.empty_cache()
    gc.collect()
    return data

@log_time
def subquestion_retrieval(data, run_count):
    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
    vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
    print("Embeder and Vector Db loaded")
    try:
        for hop_count in data:
            if run_count <= int(hop_count) - 1:
                for key in data[hop_count]:
                    for item in data[hop_count][key]:
                        if f"sub_question_retrieval_{run_count}" in item:
                            if item[f"sub_question_retrieval_{run_count}"]:
                                continue

                        for question in item[f"sub_questions_{run_count}"]:
                            query = embedder.get_detailed_instruct(query=question, task_description=embedder.question_task)
                            db_output = vector_db.similarity_search(query, k=100)
                            retrieved = [doc.metadata["title"] for doc in db_output]
                            item[f"sub_question_retrieval_{run_count}"].append(retrieved)
                    torch.cuda.empty_cache()
                    gc.collect()
    except Exception as e:
        save_obj(data, "data/ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved under data/ZWISCHENERGEBNIS.json')
        print(traceback.format_exc())
    del embedder
    del vector_db
    torch.cuda.empty_cache()
    gc.collect()
    return data

@log_time
def base_retrieval_for_next_iter(data, run_count):
    """
    Base retrieval for claim_(run_count +1)
    """
    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
    vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
    print("Embeder and Vector Db loaded")
    try:
        for hop_count in data:
            if run_count <= int(hop_count):
                for key in data[hop_count]:
                    for item in data[hop_count][key]:
                        if f"retrieved_{run_count + 1}" in item:
                            continue
                        query = embedder.get_detailed_instruct(query=item[f"claim_{run_count + 1}"], task_description=embedder.task)
                        db_output = vector_db.similarity_search(query, k=100)
                        retrieved = [doc.metadata["title"] for doc in db_output]
                        item[f"retrieved_{run_count + 1}"] = retrieved
    except Exception as e:
        save_obj(data, "data/ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved under data/ZWISCHENERGEBNIS.json')
        print(traceback.format_exc())
    del embedder
    del vector_db
    torch.cuda.empty_cache()
    gc.collect()
    return data





def create_batches_cross_enc(data, batch_size, run_count):
    batches = []
    info_indices = []  # To track where each batch item comes from
    current_batch = []
    current_batch_size = 0
    claim_sentence_creator = ClaimSentencePairsCreator_NewDb()
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for item_idx, item in enumerate(data[hop_count][key]):
                    claim = item[f"claim_{run_count}"]
                    retrieved = item[f"retrieved_{run_count}"]
                    claim_sentence_pairs = claim_sentence_creator.create_claim_sentence_pairs(claim=claim, titles=retrieved)
                    
                    if current_batch_size + 1 <= batch_size:
                        current_batch.extend(claim_sentence_pairs)
                        current_batch_size += 1
                    else:
                        batches.append(current_batch)
                        current_batch = claim_sentence_pairs
                        current_batch_size = 1

                    # Store index information for result integration
                    info_indices.append((hop_count, key, item_idx, len(claim_sentence_pairs)))

    # Add any remaining items that didn't fill up the last batch
    if current_batch:
        batches.append(current_batch)

    return batches, info_indices

def process_batches_cross_enc(batches):
    checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
    cross_enc1 = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict",with_title = True)
    cross_enc1 = cross_enc1.to('cuda:0')
    cross_enc2 = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict",with_title = True)
    cross_enc2 = cross_enc2.to('cuda:1')
    print("Cross Enc Loaded")
    results = []
    for batch in tqdm(batches, desc="Processing batches cross enc"):
        predictions = run_in_parallel(claims=batch, task_type="cross_enc", llm1=cross_enc1, llm2=cross_enc2)
        results.extend(predictions)
        torch.cuda.empty_cache()
        gc.collect()
    del cross_enc1
    del cross_enc2
    torch.cuda.empty_cache()
    gc.collect()
    return results

def reintegrate_cross_enc_results(data, results, info_indices, run_count, threshold):
    try:
        result_index = 0
        for (hop_count, key, item_idx, num_pairs) in info_indices:
            item = data[hop_count][key][item_idx]
            relevant_results = results[result_index:result_index + num_pairs]
            prediction_sorted = sorted(relevant_results, key=lambda x: x[2], reverse=True)
            sentences_sorted = [sentence[1] for sentence in prediction_sorted][:threshold]
            
            item[f"top_sentences_{run_count}"] = sentences_sorted
            result_index += num_pairs
        return data
    except Exception as e:
        save_obj(data, "data/ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved under data/ZWISCHENERGEBNIS.json')
        print(traceback.format_exc())

@log_time
def cross_encode(data, batch_size, run_count, threshold):
    batches, info_indices = create_batches_cross_enc(data, batch_size, run_count)
    results = process_batches_cross_enc(batches)
    return reintegrate_cross_enc_results(data, results, info_indices, run_count, threshold)


def reintegrate_cross_enc_results_for_questions(data, results, info_indices, run_count, threshold):
    try:
        result_index = 0  # Index to track the position in results
        for hop_count, key, item_idx, question_idx, num_pairs in info_indices:
            relevant_results = results[result_index:result_index + num_pairs]
            prediction_sorted = sorted(relevant_results, key=lambda x: x[2], reverse=True)
            sentences_sorted = [sentence[1] for sentence in prediction_sorted][:threshold]

            #sub_question_top_sentences_0
            if f"sub_question_sentences_{run_count}" not in data[hop_count][key][item_idx]:
                data[hop_count][key][item_idx][f"sub_question_sentences_{run_count}"] = []
    
            data[hop_count][key][item_idx][f"sub_question_sentences_{run_count}"].append(sentences_sorted)

            result_index += num_pairs
        return data
    except Exception as e:
        save_obj(data, "data/ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved under data/ZWISCHENERGEBNIS.json')
        print(traceback.format_exc())
        
def create_batches_cross_enc_for_subquestions(data, batch_size, run_count):
    batches = []
    info_indices = []
    current_batch = []
    current_batch_size = 0
    claim_sentence_creator = ClaimSentencePairsCreator_NewDb()
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for item_idx, item in enumerate(data[hop_count][key]):
                    for question_idx, question in enumerate(item[f"sub_questions_{run_count}"]):
                        if f"sub_question_sentences_{run_count}" in item:
                            if len(item[f"sub_question_sentences_{run_count}"]) >= question_idx + 1:
                                continue
                        retrieved = item[f"sub_question_retrieval_{run_count}"][question_idx]
                        question_sentence_pairs = claim_sentence_creator.create_claim_sentence_pairs(claim=question, titles=retrieved)
                        
                        # Append to current batch if under batch size
                        if current_batch_size + 1 <= batch_size:
                            current_batch.extend(question_sentence_pairs)
                            current_batch_size += 1
                            info_indices.append((hop_count, key, item_idx, question_idx, len(question_sentence_pairs)))
                        else:
                            # If batch is full, start a new one
                            batches.append(current_batch)
                            current_batch = question_sentence_pairs
                            current_batch_size = 1
                            info_indices.append((hop_count, key, item_idx, question_idx, len(question_sentence_pairs)))

    # Ensure the last batch is added if not empty
    if current_batch:
        batches.append(current_batch)

    return batches, info_indices

@log_time
def cross_encode_sub_questions(data, batch_size, run_count, threshold):
    batches, info_indices = create_batches_cross_enc_for_subquestions(data, batch_size, run_count)
    results = process_batches_cross_enc(batches)
    return reintegrate_cross_enc_results_for_questions(data, results, info_indices, run_count, threshold)


def create_top_sentences(data, run_count, threshold=60):
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for item in data[hop_count][key]:
                    if f"top_sentences_{run_count}" in item:
                        continue
                    
                    try:
                        top_sentences_list = []
                        for index in range(threshold):
                            if len(top_sentences_list) >= threshold:
                                break
                            for sentences_list in item[f"sub_question_sentences_{run_count}"]:
                                if len(top_sentences_list) >= threshold:
                                    break
                                if index < len(sentences_list):
                                    if sentences_list[index] not in top_sentences_list:
                                        top_sentences_list.append(sentences_list[index])


                        item[f"top_sentences_{run_count}"] = top_sentences_list
                    except KeyError:
                        uid = item["uid"]
                        logging.info(f"Error in top sentence generation, item {uid}, hop_count {hop_count}, key {key}, run_count {run_count} had an errounous generation, setting to previous run results.")
                        item[f"claim_{run_count}"] = item[f"claim_{run_count-1}"]
                        item[f"retrieved_{run_count}"] = item[f"retrieved_{run_count-1}"]
                        item[f"sub_questions_{run_count}"] = item[f"sub_questions_{run_count-1}"]
                        item[f"sub_question_retrieval_{run_count}"] = item[f"sub_question_retrieval_{run_count-1}"]
                        item[f"top_sentences_{run_count}"] = item[f"top_sentences_{run_count-1}"]
    return data
