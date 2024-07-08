import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pipeline_parts import generate_decomposition, decomposed_claim_retrieval
from utils import load_obj, save_obj
from tqdm import tqdm
import torch
import logging


logging.basicConfig(filename='decomp_pipeline_timing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


data = load_obj("data/iteration_full_data.json")

                    
try:
    for run_count in tqdm(range(5), desc='Run Count'):
        data = generate_decomposition(data, run_count)
        data = decomposed_claim_retrieval(data, run_count)


    save_obj(data, f"data/iterative_decomp_FULL_DATASET.json")

except Exception as e:
    # print traceback
    save_obj(data, f"data/iterative_decomp_FULL_DATASET_ZWISCHENERGEBNIS.json")
    print(e)
    print("################SAVING ZWISCHENERGEBNIS################")
    