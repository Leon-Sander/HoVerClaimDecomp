import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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





def decomposed_claim_retrieval(data):
    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
    vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
    print("Embeder and Vector Db loaded")
    for run_count in range(5):
        for hop_count in data:
            if run_count <= int(hop_count):
                for key in data[hop_count]:
                    for item in data[hop_count][key]:
                        if f"decomposed_claims_retrieval1000_{run_count}" in item:
                            if item[f"decomposed_claims_retrieval1000_{run_count}"]:
                                continue
                        else:
                            item[f"decomposed_claims_retrieval1000_{run_count}"] = []

                        for claim in item[f"decomposed_claims_{run_count}"]:
                            query = embedder.get_detailed_instruct(query=claim, task_description=embedder.task)
                            db_output = vector_db.similarity_search(query, k=1000)
                            retrieved = [doc.metadata["title"] for doc in db_output]
                            item[f"decomposed_claims_retrieval1000_{run_count}"].append(retrieved)
                    torch.cuda.empty_cache()
                    gc.collect()
    del embedder
    del vector_db
    torch.cuda.empty_cache()
    gc.collect()
    return data

if __name__ == "__main__":

    data = load_obj("data/iterative_decomp_FULL_DATASET_1_combined.json")
    data = decomposed_claim_retrieval(data)
    save_obj(data, "data/iterative_decomp_FULL_DATASET_1_combined_1000.json")