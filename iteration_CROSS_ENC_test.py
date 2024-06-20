import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM
from custom_mistral_embedder import CustomMistralEmbedder
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from datetime import datetime
from create_sentences_dict import ClaimSentencePairsCreator_NewDb
logging.basicConfig(filename='threshold_tests_timing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


data = load_obj("data/iteration_full_data.json")
#data = load_obj("data/iterative_FULL_DATASET_with_questions_60_no_filter_ZWISCHENERGEBNIS.json")

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

def cross_enc_in_parallel(claims, task_type, cross_enc1, cross_enc2, cross_enc3, cross_enc4):
    if len(claims) < 4:
        return process_claims(cross_enc1, claims, task_type)
    # Divide claims into four parts
    quarter_point = len(claims) // 4
    first_quarter = claims[:quarter_point]
    second_quarter = claims[quarter_point:2*quarter_point]
    third_quarter = claims[2*quarter_point:3*quarter_point]
    fourth_quarter = claims[3*quarter_point:]

    # Execute tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_claims, cross_enc1, first_quarter, task_type),
            executor.submit(process_claims, cross_enc2, second_quarter, task_type),
            executor.submit(process_claims, cross_enc3, third_quarter, task_type),
            executor.submit(process_claims, cross_enc4, fourth_quarter, task_type)
        ]
        results = [future.result() for future in futures]

    # Combine results from all futures
    combined_results = []
    for result in results:
        combined_results.extend(result)

    return combined_results

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
def claim_refinement(run_count):
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm1 = TransformerLLM(model_id, device_map="cuda:0")
    llm2 = TransformerLLM(model_id, device_map="cuda:1")
    print("llm Loaded")
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for j in tqdm(range(0, len(data[hop_count][key]), 10), desc=f'Refining Base Claims {run_count}, {hop_count}, {key}'):
                    base_claims_context = []
                    # Collect claims and sentences for refinement only if they haven't been refined for the next run_count
                    for k in range(10):
                        idx = j + k
                        if idx < len(data[hop_count][key]) and f"claim_{run_count + 1}" not in data[hop_count][key][idx]:
                            claim = data[hop_count][key][idx][f"claim_{run_count}"]
                            top_sentences = data[hop_count][key][idx][f"top_sentences_{run_count}"]
                            top_sentences = "\n".join(top_sentences)
                            base_claims_context.append((claim, top_sentences))
                    
                    if base_claims_context:  # Check if there are any claims to refine
                        enhanced_claims = run_in_parallel(claims=base_claims_context, task_type="claim_refinement", llm1=llm1, llm2=llm2)
                        
                        # Update enhanced
                        for l, enhanced_claim in enumerate(enhanced_claims):
                            data[hop_count][key][j + l][f"claim_{run_count + 1}"] = enhanced_claim
                        del enhanced_claims
                    torch.cuda.empty_cache()
                    gc.collect()
    del llm1
    del llm2
    torch.cuda.empty_cache()
    gc.collect()

def create_batches_cross_enc(batch_size, run_count):
    batches = []
    info_indices = []  # To track where each batch item comes from
    current_batch = []
    current_batch_size = 0
    claim_sentence_creator = ClaimSentencePairsCreator_NewDb()
    for hop_count in data:
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
    """cross_enc3 = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict",with_title = True)
    cross_enc3 = cross_enc1.to('cuda:2')
    cross_enc4 = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict",with_title = True)
    cross_enc4 = cross_enc2.to('cuda:3')"""
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
    result_index = 0
    for (hop_count, key, item_idx, num_pairs) in info_indices:
        item = data[hop_count][key][item_idx]
        relevant_results = results[result_index:result_index + num_pairs]
        prediction_sorted = sorted(relevant_results, key=lambda x: x[2], reverse=True)
        sentences_sorted = [sentence[1] for sentence in prediction_sorted][:threshold]
        
        item[f"top_sentences_{run_count}"] = sentences_sorted
        result_index += num_pairs


def cross_encode(batch_size, run_count, threshold):
    batches, info_indices = create_batches_cross_enc(batch_size, run_count)
    results = process_batches_cross_enc(batches)
    reintegrate_cross_enc_results(data, results, info_indices, run_count, threshold)

@log_time
def final_claim_retrieval(run_count):
    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
    vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
    print("Embeder and Vector Db loaded")
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
                torch.cuda.empty_cache()
                gc.collect()
    del embedder
    del vector_db
    torch.cuda.empty_cache()
    gc.collect()


try:
    for threshold in tqdm([5, 10, 20, 60, 80, 125, 300],desc="Thresholds"):
        data = load_obj("data/iteration_full_data.json")
        for run_count in range(1):
            cross_encode(batch_size=10, run_count=run_count, threshold=threshold)
            claim_refinement(run_count)
            final_claim_retrieval(run_count)


        save_obj(data, f"data/iterative_cross_test_{threshold}_FULL_DATA.json")

except Exception as e:
    save_obj(data, f"data/iterative_cross_test_{threshold}_ZWISCHENERGEBNIS.json")
    print(e)
    raise e



