import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from custom_mistral_embedder import CustomMistralEmbedder
from custom_bert_embedder import CustomBertEmbedder
from tqdm import tqdm
from utils import load_obj, load_vectordb, save_obj
from huggingface_llm_loading import DecomposedClaimsOutputParser

def hover_similarity_search(embedder_name, dataset_type : str = "dev", k : int = 100):
    output_parser = DecomposedClaimsOutputParser()
    data = load_obj("qualitative_analysis_decomposed_claims_10_9shot_noinstruct_for_context_comparison.json")
    output = {
        "2": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "3": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "4": {"SUPPORTED": [], "NOT_SUPPORTED": []}}
    for hop_count in data:
        for key in data[hop_count]:
            for item in data[hop_count][key]:

                new_item = item.copy()
                new_item["retrieved_decomposed_claims"] = []
                decomposed_claims = output_parser.parse(item["decomposed_claims"])
                for decomposed_claim in decomposed_claims:
                    #print(decomposed_claim)
                    if embedder_name == "mistral":
                        query = embedder.get_detailed_instruct(query=decomposed_claim, task_description=embedder.task)
                    else:
                        query = item["claim"]

                    db_output = vector_db.similarity_search(query, k = k)
                    retrieved = []
                    for doc in db_output:
                        retrieved.append(doc.metadata["title"])

                    new_item["retrieved_decomposed_claims"].extend(retrieved)
                output[hop_count][key].append(new_item)

    output_file_path = f'decomposed_qualitative_analysis_retrieval_10_for_context_comparison.json'
    save_obj(obj=output ,path=output_file_path)

def mistral_base_search(embedder_name, dataset_type : str = "dev", k : int = 100):
    data = load_obj("/home/sander/code/thesis/hover/leon/qualitative_analysis/base_context_decompose_retrieval_comparison.json")
    output = {
        "2": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "3": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "4": {"SUPPORTED": [], "NOT_SUPPORTED": []}}
    for hop_count in data:
        for key in data[hop_count]:
            for item in data[hop_count][key]:

                new_item = item.copy()
                if embedder_name == "mistral":
                    query = embedder.get_detailed_instruct(query=item["claim"], task_description=embedder.task)
                else:
                    query = item["claim"]
                claims_list = item["decomposed_claims_with_context"].split("\n")
                k_retrv = len(claims_list) * k
                db_output = vector_db.similarity_search(query, k = k_retrv)
                retrieved = []
                for doc in db_output:
                    retrieved.append(doc.metadata["title"])

                new_item["base_retrieved_k"] = retrieved
                output[hop_count][key].append(new_item)

    output_file_path = f'qualitative_analysis/base_context_decompose_retrieval_comparison2.json'
    save_obj(obj=output ,path=output_file_path)

if __name__ == '__main__': 
    embedder_name = "mistral"
    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
    vector_db = load_vectordb(embedder, f"chroma_db_{embedder_name}", "wiki_data")
    #hover_similarity_search(embedder_name=embedder_name, dataset_type="dev", k = 100)
    mistral_base_search(embedder_name=embedder_name, dataset_type="dev", k = 100)