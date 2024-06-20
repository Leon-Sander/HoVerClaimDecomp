import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM
from custom_mistral_embedder import CustomMistralEmbedder
from utils import load_obj, load_vectordb, save_obj
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
import logging
import time
from datetime import datetime

logging.basicConfig(filename='pipeline_timing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#data = load_obj("data/iteration_full_data.json")
data = load_obj("data/iterative_FULL_DATASET_with_questions_60_no_filter_ZWISCHENERGEBNIS.json")

model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm1 = TransformerLLM(model_id, device_map="cuda:1")
llm2 = TransformerLLM(model_id, device_map="cuda:2")

print("llm Loaded")
checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
print("Cross Enc Loaded")
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")


def log_time(func):
    """ Decorator to log the execution time of a function. """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def process_claims(llm, batch_claims, task_type):
    if task_type == "subquestion_generation":
        output = llm.generate_subquestions(batch_claims)
    elif task_type == "claim_refinement":
        output = llm.generate_claim_refinement(batch_claims)
    else:
        raise ValueError("Invalid llm task type")
    return output


def run_in_parallel(claims, task_type):
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
def generate_subquestions(run_count):
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                all_claims = []
                indices = []
                for idx, item in enumerate(data[hop_count][key]):
                    if f"sub_questions_{run_count}" not in item:
                        all_claims.append(item[f"claim_{run_count}"])
                        indices.append(idx) 

                if not all_claims:
                    print(f"No subquestions to generate for {run_count}, {hop_count}, {key}")
                    continue

                for i in tqdm(range(0, len(all_claims), 50), desc=f'Subquestion Generation {run_count}, {hop_count}, {key}'):
                    batch_claims = all_claims[i:i + 50]
                    batch_indices = indices[i:i + 50]
                    sub_questions_batch = run_in_parallel(claims=batch_claims, task_type="subquestion_generation")
                    
                    # Collect sub-questions and prepare base claim enhancement context
                    for idx, sub_questions in enumerate(sub_questions_batch):
                        original_idx = batch_indices[idx]
                        data[hop_count][key][original_idx][f"sub_questions_{run_count}"] = sub_questions
                        data[hop_count][key][original_idx][f"sub_question_retrieval_{run_count}"] = []
                    del sub_questions_batch

                torch.cuda.empty_cache()
                gc.collect()

@log_time
def subquestion_retrieval(run_count):
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for item in data[hop_count][key]:
                    if f"sub_question_retrieval_{run_count}" in item and item[f"sub_question_retrieval_{run_count}"]:
                        continue

                    for question in item[f"sub_questions_{run_count}"]:
                        query = embedder.get_detailed_instruct(query=question, task_description=embedder.question_task)
                        db_output = vector_db.similarity_search(query, k=100)
                        retrieved = [doc.metadata["title"] for doc in db_output]
                        item[f"sub_question_retrieval_{run_count}"].append(retrieved)
                torch.cuda.empty_cache()
                gc.collect()

@log_time
def cross_encoding(run_count):
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for item in data[hop_count][key]:
                    if f"top_sentences_{run_count}" in item:
                        continue 

                    sentences_per_question = []
                    for idx, question in enumerate(item[f"sub_questions_{run_count}"]):
                        retrieved = item[f"sub_question_retrieval_{run_count}"][idx]
                        question_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim=question, titles=retrieved)
                        prediction = cross_enc.predict(question_sentence_pairs, return_probabilities=True)
                        prediction_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
                        sentences_sorted = [sentence[1] for sentence in prediction_sorted]
                        sentences_per_question.append(sentences_sorted[:60])

                    # Collect sentences for base refinement
                    top_sentences = []
                    for index in range(60):
                        for sentences_list in sentences_per_question:
                            if len(top_sentences) >= 60:
                                break
                            if index < len(sentences_list):
                                top_sentences.append(sentences_list[index])

                    item[f"top_sentences_{run_count}"] = "\n".join(top_sentences)
                    #all_top_sentences.append("\n".join(top_sentences))

@log_time
def claim_refinement(run_count):
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
                            base_claims_context.append((claim, top_sentences))
                    
                    if base_claims_context:  # Check if there are any claims to refine
                        enhanced_claims = run_in_parallel(claims=base_claims_context, task_type="claim_refinement")
                        
                        # Update enhanced
                        for l, enhanced_claim in enumerate(enhanced_claims):
                            data[hop_count][key][j + l][f"claim_{run_count + 1}"] = enhanced_claim
                        del enhanced_claims
                    torch.cuda.empty_cache()
                    gc.collect()

@log_time
def base_retrieval(run_count):
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for item in data[hop_count][key]:
                    if f"retrieved_{run_count + 1}" in item:
                        continue
                    query = embedder.get_detailed_instruct(query=item[f"claim_{run_count + 1}"], task_description=embedder.task)
                    db_output = vector_db.similarity_search(query, k=100)
                    retrieved = [doc.metadata["title"] for doc in db_output]
                    item[f"retrieved_{run_count + 1}"] = retrieved
                    
try:
    for run_count in tqdm(range(4), desc='Run Count'):
        generate_subquestions(run_count)
        torch.cuda.empty_cache()
        gc.collect()

        subquestion_retrieval(run_count)
        torch.cuda.empty_cache()
        gc.collect()

        cross_encoding(run_count)
        torch.cuda.empty_cache()
        gc.collect()

        claim_refinement(run_count)
        torch.cuda.empty_cache()
        gc.collect()

        base_retrieval(run_count)
        torch.cuda.empty_cache()
        gc.collect()


    save_obj(data, "data/iterative_FULL_DATASET_with_questions_60_no_filter.json")


except Exception as e:
    # print traceback
    save_obj(data, "data/iterative_FULL_DATASET_with_questions_60_no_filter_ZWISCHENERGEBNIS2.json")
    print(e)
    print("################SAVING ZWISCHENERGEBNIS################")
    