import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM, create_chain, create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import add_context_prompt, add_key_entities_refined_prompt
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import EnhancedBaseClaimsOutputParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
data = load_obj("data/train_qualitative_analysis_100.json")

checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
print("Cross Enc Loaded")
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")

prompt_text = ""

for hop_count in data:
    for key in data[hop_count]:
        for item in tqdm(data[hop_count][key]):
            query = embedder.get_detailed_instruct(query=item["claim"], task_description=embedder.task)
            gold = [title[0] for title in item["supporting_facts"]]
            db_output = vector_db.similarity_search(query, k = 10)
            retrieved = []
            for doc in db_output:
                title = doc.metadata["title"]
                if title not in gold:
                    retrieved.append(title)

            claim_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= item["claim"], titles=retrieved + gold)
            prediction = cross_enc.predict(claim_sentence_pairs, return_probabilities=True)
            
            
            
            prediciton_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
            sentences_sorted = [item[1] for item in prediciton_sorted][:10]
            item["sentences"] = sentences_sorted
            sentences = "\n".join(sentences_sorted)
            prompt = f"""
CLAIM: {item["claim"]}
CONTEXT: {sentences}
REFINED CLAIM:"""
            prompt_text += prompt + "\n" + str(item["supporting_facts"]) + "\n\n"

save_obj(data, "data/train_prompt_key_ents_100.json")
with open("data/prompt_key_ents.txt", "w") as f:
    f.write(prompt_text)