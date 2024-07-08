
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from custom_mistral_embedder import CustomMistralEmbedder
from custom_bert_embedder import CustomBertEmbedder
from db_operations import connect_to_db, get_documents_batch, get_total_document_count
import chromadb
from tqdm import tqdm
import torch
import gc

db_path = '/data/wiki_wo_links.db'
conn, cursor = connect_to_db(db_path)

single_gpu_batch_size = 512
gpu_count = 1  
batch_size = single_gpu_batch_size * gpu_count


embedder = CustomMistralEmbedder(gpu_count=gpu_count, batch_size=single_gpu_batch_size)
#embedder = CustomBertEmbedder(gpu_count=gpu_count, batch_size=single_gpu_batch_size)
#chroma_db_mistral
persistent_client = chromadb.PersistentClient("tes2tes2t")
collection = persistent_client.create_collection(
        name="wiki_data",
        metadata={"hnsw:space": "cosine"} 
    )

offset = 0
total_number_of_documents = get_total_document_count(cursor)
print(f"total number of documents to embed: {total_number_of_documents}")
total_iterations = total_number_of_documents // batch_size + 1
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
    torch.cuda.empty_cache()
    gc.collect()

conn.close()
