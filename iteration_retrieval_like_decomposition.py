import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM, create_chain, create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import add_context_prompt, sub_question_prompt, add_key_entities_prompt, decompose_9shot_instruct
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import EnhancedBaseClaimsOutputParser, SubQuestionsOutputParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm

decomposed = load_obj("data/iterative_test_decomposition_atomar.json")
base_60 = load_obj("data/iterative_test_base_60.json")
base_60_no_filter = load_obj("data/iterative_test_base_60_no_filter.json")
questions_double_cross = load_obj("data/iterative_test_with_questions.json")
questions_60 = load_obj("data/iterative_test_with_questions_60_sentences.json")
questions_60_no_filter = load_obj("data/iterative_test_with_questions_60_sentences_no_filter.json")

embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")

def retrieve(query, k ):
    db_output = vector_db.similarity_search(query, k = k)
    retrieved = []
    for doc in db_output:
        retrieved.append(doc.metadata["title"])
    return retrieved

# hop_count +1 iterationen
for run_count in tqdm(range(5)):
    for hop_count in decomposed:
        if run_count <= int(hop_count):
            # somit hop_count+1 retrievals, einmal mit base claim, n mal mit hop count claims
            for key in decomposed[hop_count]:
                for i, item in enumerate(decomposed[hop_count][key]):
                    k = 100 * len(item[f"decomposed_claims_{run_count}"])

                    
                    query = embedder.get_detailed_instruct(query=base_60[hop_count][key][i][f"claim_{run_count}"], task_description=embedder.task)
                    retrieved = retrieve(query, k)
                    item[f"base_60_like_decomposed_{run_count}"] = retrieved
                    query = embedder.get_detailed_instruct(query=base_60_no_filter[hop_count][key][i][f"claim_{run_count}"], task_description=embedder.task)
                    retrieved = retrieve(query, k)
                    item[f"base_60_no_filter_like_decomposed_{run_count}"] = retrieved
                    query = embedder.get_detailed_instruct(query=questions_double_cross[hop_count][key][i][f"claim_{run_count}"], task_description=embedder.task)
                    retrieved = retrieve(query, k)
                    item[f"questions_double_cross_like_decomposed_{run_count}"] = retrieved
                    query = embedder.get_detailed_instruct(query=questions_60[hop_count][key][i][f"claim_{run_count}"], task_description=embedder.task)
                    retrieved = retrieve(query, k)
                    item[f"questions_60_like_decomposed_{run_count}"] = retrieved
                    query = embedder.get_detailed_instruct(query=questions_60_no_filter[hop_count][key][i][f"claim_{run_count}"], task_description=embedder.task)
                    retrieved = retrieve(query, k)
                    item[f"questions_60_no_filter_like_decomposed_{run_count}"] = retrieved

save_obj(decomposed, "data/iterative_comparison_with_decomposed_atomar.json")

