import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'

from utils import load_obj, save_obj
from tqdm import tqdm
import logging

from pipeline_parts import claim_refinement, base_retrieval_for_next_iter, cross_encode

logging.basicConfig(filename='threshold_tests_timing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        for threshold in tqdm([5, 10, 20, 60],desc="Thresholds"): # 80, 125, 300
            data = load_obj("data/iteration_full_data.json")
            for run_count in range(1):
                data = cross_encode(data, batch_size=10, run_count=run_count, threshold=threshold)
                data = claim_refinement(data, run_count)
                data = base_retrieval_for_next_iter(data, run_count)


            save_obj(data, f"data/iterative_cross_test_{threshold}_FULL_DATA.json")

    except Exception as e:
        save_obj(data, f"data/iterative_cross_test_{threshold}_ZWISCHENERGEBNIS.json")
        print(e)
        raise e



