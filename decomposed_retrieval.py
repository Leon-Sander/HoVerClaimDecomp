import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from custom_mistral_embedder import CustomMistralEmbedder
from custom_bert_embedder import CustomBertEmbedder
from tqdm import tqdm
from utils import load_obj, load_vectordb, save_obj


def hover_similarity_search(embedder_name, dataset_type : str = "dev", k : int = 100):
    #data = load_obj(f"/home/sander/code/thesis/hover/data/hover/hover_{dataset_type}_release_v1.1.json")
    data = load_obj("enriched_claims_decomposed.json")
    output = []
    for item in tqdm(data):
        new_item = item.copy()
        new_item["retrieved"] = []
        decomposed_claims = item["decomposed_claims"].replace("\n\n", "\n").split("\n")
        for decomposed_claim in decomposed_claims:
            if embedder_name == "mistral":
                query = embedder.get_detailed_instruct(query=decomposed_claim.lstrip(), task_description=embedder.task)
            else:
                query = item["claim"]

            db_output = vector_db.similarity_search(query, k = k)
            retrieved = []
            for doc in db_output:
                retrieved.append(doc.metadata["title"])

            new_item["retrieved"].extend(retrieved)
        output.append(new_item)

    output_file_path = f'decomposed_{embedder_name}_retrieval_output_{dataset_type}_{k}_enriched_decomposed.json'
    save_obj(obj=output ,path=output_file_path)

if __name__ == '__main__': 
    embedder_name = "mistral"
    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
    vector_db = load_vectordb(embedder, f"chroma_db_{embedder_name}", "wiki_data")
    hover_similarity_search(embedder_name=embedder_name, dataset_type="dev", k = 100)
    #hover_similarity_search(embedder_name=embedder_name, dataset_type="dev", k = 1000)
    #hover_similarity_search(embedder_name=embedder_name, dataset_type="train", k = 100)
    #hover_similarity_search(embedder_name=embedder_name, dataset_type="train", k = 1000)