import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'
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
from prompt_templates import decompose_9shot_instruct, decompose_entity_based, decompose_without_redundancy
logging.basicConfig(filename='decomp_baseline_logging.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


data = load_obj("data/iteration_full_data.json")
#data = load_obj("data/decomp_baseline_TFIDF_FULL_DATASET_100.json")
#data = load_obj("data/iterative_FULL_DATASET_with_questions_60_no_filter_ZWISCHENERGEBNIS.json")

prompt_templates = [decompose_9shot_instruct, decompose_entity_based, decompose_without_redundancy]

for index, template in enumerate(prompt_templates):
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm1 = TransformerLLM(model_id, device_map="cuda:0", decomposed_template = template)
    llm2 = TransformerLLM(model_id, device_map="cuda:1", decomposed_template = template)
    #llm1 = ""
    #llm2 = ""
    print("llm Loaded")

    #embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
    #vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
    #print("Embeder and Vector Db loaded")


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
        elif task_type == "decomposition":
            output = llm.generate_decomposition(batch_claims)
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
    def generate_decomposition(run_count):
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
                        print(f"No subquestions to generate for {run_count}, {hop_count}, {key}")
                        continue

                    for i in tqdm(range(0, len(all_claims), 50), desc=f'Decomposition {run_count}, {hop_count}, {key}'):
                        batch_claims = all_claims[i:i + 50]
                        batch_indices = indices[i:i + 50]
                        decomposed_batch = run_in_parallel(claims=batch_claims, task_type="decomposition")
                        
                        for idx, decomposed_claims in enumerate(decomposed_batch):
                            original_idx = batch_indices[idx]
                            data[hop_count][key][original_idx][f"decomposed_claims_{run_count}"] = decomposed_claims
                            data[hop_count][key][original_idx][f"decomposed_claims_retrieval_{run_count}"] = []
                        del decomposed_batch

                    torch.cuda.empty_cache()
                    gc.collect()

    @log_time
    def decomposed_claim_retrieval(run_count):
        torch.cuda.empty_cache()
        gc.collect()
        embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
        vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
        print("Embeder and Vector Db loaded")
        for hop_count in data:
            if run_count <= int(hop_count) - 1:
                for key in data[hop_count]:
                    for item in data[hop_count][key]:
                        #if f"decomposed_claims_retrieval_{run_count}" in item and item[f"decomposed_claims_retrieval_{run_count}"]:
                        #    continue
                        item[f"decomposed_claims_retrieval_{run_count}"] = []
                        for claim in item[f"decomposed_claims_{run_count}"]:
                            query = embedder.get_detailed_instruct(query=claim, task_description=embedder.task)
                            db_output = vector_db.similarity_search(query, k=100)
                            retrieved = [doc.metadata["title"] for doc in db_output]
                            item[f"decomposed_claims_retrieval_{run_count}"].append(retrieved)
                    torch.cuda.empty_cache()
                    gc.collect()
        del embedder
        del vector_db

                        
    try:
        run_count = 0
        generate_decomposition(run_count)
        del llm1
        del llm2
        torch.cuda.empty_cache()
        gc.collect()

        decomposed_claim_retrieval(run_count)
        torch.cuda.empty_cache()
        gc.collect()

        save_obj(data, f"data/decomp_baseline_FULL_DATASET_{index}.json")


    except Exception as e:
        # print traceback
        save_obj(data, f"data/decomp_baseline_FULL_DATASET_{index}_ZWISCHENERGEBNIS.json")
        print(e)
        print("################SAVING ZWISCHENERGEBNIS################")
        