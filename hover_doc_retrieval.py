import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from custom_mistral_embedder import CustomMistralEmbedder
from custom_bert_embedder import CustomBertEmbedder
from tqdm import tqdm
from utils import load_obj, load_vectordb, save_obj


def hover_document_retrieval(embedder_name, dataset_type : str = "dev", k : int = 100):
    data = load_obj(f"data/hover_{dataset_type}_release_v1.1.json")

    for item in tqdm(data):
        
        if embedder_name == "mistral":
            query = embedder.get_detailed_instruct(query=item["claim"], task_description=embedder.task)
        else:
            query = item["claim"]

        db_output = vector_db.similarity_search(query, k = k)
        retrieved = []
        for doc in db_output:
            retrieved.append(doc.metadata["title"])

        item["retrieved"] = retrieved

    output_file_path = f'{embedder_name}_retrieval_output_{dataset_type}_{k}.json'
    save_obj(obj=data ,path=output_file_path)

if __name__ == '__main__': 
    embedder_name = "mistral"
    embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
    vector_db = load_vectordb(embedder, f"chroma_db_{embedder_name}", "wiki_data")
    hover_document_retrieval(embedder_name=embedder_name, dataset_type="dev", k = 100)
    #hover_document_retrieval(embedder_name=embedder_name, dataset_type="dev", k = 1000)
    #hover_document_retrieval(embedder_name=embedder_name, dataset_type="train", k = 100)
    #hover_document_retrieval(embedder_name=embedder_name, dataset_type="train", k = 1000)