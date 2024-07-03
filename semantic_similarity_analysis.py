import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from custom_mistral_embedder import CustomMistralEmbedder
from utils import load_obj, load_vectordb, save_obj
from db_operations import get_doc_by_title_new_db, connect_to_db
from tqdm import tqdm
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
#vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")

cursor, connection = connect_to_db("/home/sander/code/thesis/hover/data/hover_with_sentences_splitted.db")

data = load_obj("/home/sander/code/thesis/hover/leon/data/decomp_baseline_FULL_DATASET_FINAL.json")
retrieval_key = "decomposed_claims_retrieval_100_mistral_no_filter"
base_retrieval_key = "retrieved_0"
decomposed_claims_key = "decomposed_claims_0"

worse_results_counter = 0
better_results_counter = 0
same_results_counter = 0
single_answer = 0
perfect_result = 0
semantic_similaritys = {}
for hop_count in data:
    for key in data[hop_count]:
        for item_index, item in tqdm(enumerate(data[hop_count][key])):
            #item["retrieved"].extend(item["base_retrieved"])
            #item["retrieved"] = item["base_retrieved"]
            item["decomposed_claims"] = "None"
            found = []
            found_base = []
            not_found = []
            not_found_base = []
            for fact in item["supporting_facts"]:
                if fact[0] in item[retrieval_key]:
                    found.append(fact[0])
                else:
                    not_found.append(fact[0])
                if fact[0] in item[base_retrieval_key]:
                    found_base.append(fact[0])
                else:
                    not_found_base.append(fact[0])
            #if len(not_found) > len(not_found_base):
             #   worse_results_counter += 1
            if len(found) > len(found_base):
                better_results_counter += 1
                semantic_similaritys[item["claim_0"]] = {}
                facts = [fact for fact in found if fact not in found_base]
                for fact in facts:
                    
                    text = get_doc_by_title_new_db(cursor, fact)
                    query = embedder.get_detailed_instruct(embedder.task, item["claim_0"])
                    multi_hop_similarity = embedder.semantic_similarity(query, text)
                    semantic_similaritys[item["claim_0"]][fact] = {"base" : multi_hop_similarity, "decomposed" : []}
                    for decomposed_claim in item[decomposed_claims_key]:
                        query = embedder.get_detailed_instruct(embedder.task, decomposed_claim)
                        decomposed_similarity = embedder.semantic_similarity(query, text)
                        semantic_similaritys[item["claim_0"]][fact]["decomposed"].append(decomposed_similarity)

save_obj(semantic_similaritys, "/home/sander/code/thesis/hover/leon/data/semantic_similaritys_analysis_with_task_COSINE.json")