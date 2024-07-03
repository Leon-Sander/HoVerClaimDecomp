import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import load_obj, save_obj
from tqdm import tqdm
import logging
import traceback
from pipeline_parts import claim_refinement, base_retrieval_for_next_iter, cross_encode, create_top_sentences

logging.basicConfig(filename='pipeline_timing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    data = load_obj("data/iteration_full_data.json")
    #data = load_obj("data/ZWISCHENERGEBNIS.json")
    #data = load_obj("data/iterative_FULL_DATASET_with_questions_NEW.json")

    try:
        for run_count in tqdm(range(0,4), desc='Run Count'):
            logging.info(f'Starting run count: {run_count}')
            logging.info('Cross encoding for base retrieval')
            data = cross_encode(data=data , batch_size=10, run_count=run_count, threshold=60)
            logging.info('Claim refinement')
            data = claim_refinement(data, run_count)
            logging.info('Base retrieval for next iteration')
            data = base_retrieval_for_next_iter(data = data, run_count=run_count)
            logging.info(f'Completed run count: {run_count}')


        save_obj(data, "data/iterative_FULL_DATASET_subquestion_ABLATION.json")
        logging.info('Data saved successfully')


    except Exception as e:
        # print traceback
        save_obj(data, "data/iterative_FULL_DATASET_subquestion_ABLATION_ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved successfully')
        print(traceback.format_exc())
        #print("################SAVING ZWISCHENERGEBNIS################")
    