import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils import load_obj, load_vectordb, save_obj
from pipeline_parts import generate_decomposition, decomposed_claim_retrieval, assemble_top_decomposed_retrieval_documents
import logging
logging.basicConfig(filename='decomp_baseline_logging.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    data = load_obj("data/iteration_full_data.json")

    try:
        run_count = 0
        data = generate_decomposition(data, run_count)
        data = decomposed_claim_retrieval(data, run_count)
        data = assemble_top_decomposed_retrieval_documents(data, 0)
        save_obj(data, "data/decomp_baseline_FULL_DATASET_9shot_INSTRUCT.json")

    except Exception as e:
        save_obj(data, "data/decomp_baseline_FULL_DATASET_9shot_INSTRUCT_ZWISCHENERGEBNIS.json")
        print(e)
        print("################SAVING ZWISCHENERGEBNIS################")
        