import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM, create_chain, create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import decision_prompt_3shot, decision_prompt_9shot
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import TrueFalseParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

data = load_obj("data/iterative_qualitative_analysis_not_supported_filter_60.json")
#qualitative_analysis_claims_10.json


model_id="mistralai/Mistral-7B-Instruct-v0.2"
#model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
true_false_evaluator = create_chain_with_postprocessor(create_prompt(template=decision_prompt_9shot), 
                        create_llm_pipeline(model_id=model_id,
                                        device_map="cuda:0", load_in_8bit=False, load_in_4bit=False),
                        stop=["Evaluation:", "Claim:", "Enhanced Claim:", "Base Claim:"], 
                        postprocessor=TrueFalseParser)

# hop_count +1 iterationen
for run_count in tqdm(range(5)):
    for hop_count in data:
        if run_count <= int(hop_count):
            # somit hop_count+1 retrievals, einmal mit base claim, n mal mit hop count claims
            supp_batch = []
            not_supp_batch = []
            for item in data[hop_count]:
                supp_batch.append({"base_claim": item["claim_0"], "enhanced_claim": item[f"claim_{run_count}"]})
                not_supp_batch.append({"base_claim": item["not_supported_counterpart"]["claim_0"], "enhanced_claim": item["not_supported_counterpart"][f"claim_{run_count}"]})
                
                
                
            supp_output = true_false_evaluator.batch(supp_batch)
            not_supp_output = true_false_evaluator.batch(not_supp_batch)
            for index, item in enumerate(data[hop_count]):
                item[f"keep_evaluating_{run_count}"] = True if supp_output[index] == "True" else False
                item["not_supported_counterpart"][f"keep_evaluating_{run_count}"] = True if not_supp_output[index] == "True" else False


save_obj(data, "data/iterative_true_false_eval_2.json")

