
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7' #,4,5'
from custom_mistral_embedder import CustomMistralEmbedder
from custom_bert_embedder import CustomBertEmbedder
import chromadb
import sqlite3
from typing import List, Tuple
import time
from tqdm import tqdm
import torch

def connect_to_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor

db_path = '../data/wiki_wo_links.db'
conn, cursor = connect_to_db(db_path)



def get_documents_batch(cursor, offset: int, batch_size: int) -> List[Tuple]:
    query = f"SELECT * FROM documents LIMIT {batch_size} OFFSET {offset}"
    cursor.execute(query)
    results = cursor.fetchall()
    return results
single_gpu_batch_size = 1012
gpu_count = 1  
batch_size = single_gpu_batch_size * gpu_count


#embedder = CustomMistralEmbedder(gpu_count=gpu_count, batch_size=single_gpu_batch_size)
embedder = CustomBertEmbedder(gpu_count=gpu_count, batch_size=single_gpu_batch_size)
#persistent_client = chromadb.PersistentClient("test_db")
#collection = persistent_client.get_collection(
#        name="wiki_data"
        #metadata={"hnsw:space": "cosine"} # l2 is the default
#    )
persistent_client = chromadb.PersistentClient("chroma_db_bert")
collection = persistent_client.create_collection(
        name="wiki_data",
        metadata={"hnsw:space": "cosine"} 
    )

offset = 0
total_iterations = 5200
for _ in tqdm(range(total_iterations), desc='Processing', unit='batch'):    
    documents = get_documents_batch(cursor, offset, batch_size)
    if not documents:
        print("DONE")
        break 

    texts = [doc[1] for doc in documents]
    embeddings = embedder.batch_embed_text_tensor_multiple_gpus(texts)

    metadatas = [{"title": doc[0]} for doc in documents]
    ids = [doc[0] for doc in documents]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    offset += batch_size

conn.close()
