import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import load_obj, save_obj
from tqdm import tqdm
import logging
import traceback
from pipeline_parts import claim_refinement, base_retrieval_for_next_iter, cross_encode_sub_questions, generate_subquestions, subquestion_retrieval, create_top_sentences, generate_decomposition, decomposed_claim_retrieval

logging.basicConfig(filename='pipeline_timing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    #data = load_obj("data/iteration_full_data.json")
    data = load_obj("data/baleen_comparison.json")
    try:
        run_count = 0
        logging.info(f'Starting run count: {run_count}')
        #logging.info('Generating subquestions')
        #data = generate_subquestions(data, run_count)
        logging.info('Retrieving subquestions')
        data = subquestion_retrieval(data, run_count)
        logging.info('Cross encoding sub questions')
        data = cross_encode_sub_questions(data=data , batch_size=10, run_count=run_count, threshold=100)
        logging.info('Creating top sentences')
        data = create_top_sentences(data, run_count, threshold=60)
        logging.info('Claim refinement')
        data = claim_refinement(data, run_count)
        logging.info('Decomposing')
        data = generate_decomposition(data, run_count + 1)
        logging.info('Decomposed Retrieval')
        data = decomposed_claim_retrieval(data, run_count +1)


        save_obj(data, "data/baleen_comparison_finished.json")
        logging.info('Data saved successfully')


    except Exception as e:
        # print traceback
        save_obj(data, "data/baleen_comparison_ZWISCHENERGEBNIS.json")
        logging.info('Zwischenergebnis saved successfully')
        print(traceback.format_exc())
        #print("################SAVING ZWISCHENERGEBNIS################")
    


