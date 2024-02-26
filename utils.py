import json
from langchain_community.vectorstores import Chroma
import chromadb

def save_obj(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def load_obj(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def load_vectordb(embeddings, db_path, collection):
    persistent_client = chromadb.PersistentClient(db_path)

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection,
        embedding_function=embeddings,
    )

    return langchain_chroma