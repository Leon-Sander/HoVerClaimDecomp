import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

data = load_obj("data/iterative_test_with_questions_60_sentences.json")

model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_decomposed_claims_generator = create_chain_with_postprocessor(create_prompt(template=decompose_9shot_instruct), 
                        create_llm_pipeline(model_id=model_id,
                                        device_map="cuda:0", load_in_8bit=False, load_in_4bit=True),
                        stop=["CLAIM:", "CLAIMS:"], 
                        postprocessor=SubQuestionsOutputParser)

print("llm Loaded")

embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")

# hop_count +1 iterationen
for run_count in tqdm(range(5)):
    for hop_count in data:
        if run_count <= int(hop_count):
            # somit hop_count+1 retrievals, einmal mit base claim, n mal mit hop count claims
            for key in data[hop_count]:
                for item in data[hop_count][key]:
                    # sub question generation
                    decomposed_claims = llm_decomposed_claims_generator.invoke({"claim": item[f"claim_{run_count}"]})
                    item[f"decomposed_claims_{run_count}"] = decomposed_claims
                    item[f"decomposed_claims_retrieval_{run_count}"] = []
                    
                    sentences_per_question = []
                    for claim in decomposed_claims:
                        query = embedder.get_detailed_instruct(query=claim, task_description=embedder.task)
                        db_output = vector_db.similarity_search(query, k = 100)
                        retrieved = []
                        for doc in db_output:
                            retrieved.append(doc.metadata["title"])
                        
                        item[f"decomposed_claims_retrieval_{run_count}"].extend(retrieved)

                    query = embedder.get_detailed_instruct(query=item[f"claim_{run_count}"], task_description=embedder.task)
                    k = 100 * len(decomposed_claims)
                    db_output = vector_db.similarity_search(query, k = k)
                    retrieved_base = []
                    for doc in db_output:
                        retrieved_base.append(doc.metadata["title"])
                    item[f"base_retrieved_like_decomposed_{run_count+1}"] = retrieved_base

save_obj(data, "data/iterative_test_decomposition.json")

