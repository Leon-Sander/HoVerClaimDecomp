import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from custom_mistral_embedder import CustomMistralEmbedder
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import torch


data = load_obj("data/iterative_TFIDF_full_data_output.json")
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1, device="cuda:0")
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")

def base_retrieval(run_count):
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                for item in data[hop_count][key]:
                    #if f"retrieved_{run_count + 1}" in item:
                    #    continue
                    query = embedder.get_detailed_instruct(query=item[f"claim_{run_count + 1}"], task_description=embedder.task)
                    db_output = vector_db.similarity_search(query, k=100)
                    retrieved = [doc.metadata["title"] for doc in db_output]
                    item[f"retrieved_{run_count + 1}"] = retrieved
                    

for run_count in tqdm(range(4), desc='Run Count'):
    base_retrieval(run_count)
    torch.cuda.empty_cache()
    gc.collect()


save_obj(data, "data/iterative_FULL_DATASET_with_questions_60_no_filter_RETRIEVAL_KORRIGIERT.json")
