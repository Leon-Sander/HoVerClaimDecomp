import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM, DecomposedClaimsOutputParser, EnhancedClaimsOutputParser, create_chain, create_llm_pipeline, create_prompt
from prompt_templates import add_context_prompt
from custom_mistral_embedder import CustomMistralEmbedder

from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm

data = load_obj("data/iteration_base.json") # ein base retrieval bereits durchgeführt
#qualitative_analysis_claims_10.json
"""data = load_obj("data/qualitative_analysis_claims_10_base.json")
for hop_count in data:
    for key in data[hop_count]:
        for item in data[hop_count][key]:
            item["previous_iteration_sentences"] = []
            item["claim_0"] = item["claim"]
            item["retrieved_0"] = item["base_retrieved"]
save_obj(data, "data/iteration_base.json")            
"""


model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = create_chain(create_prompt(template=add_context_prompt), 
                        create_llm_pipeline(model_id=model_id,
                                        device_map="cuda:0", load_in_8bit=False, load_in_4bit=True),
                        stop=["CONTEXT:", "CLAIM:"])

#llm = TransformerLLM(model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1", prompt_template= add_context_prompt, postprocessor=EnhancedClaimsOutputParser)
print("llm Loaded")
checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
print("Cross Enc Loaded")
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")
# start mit base claim und base retrieval
# 1. cross encoding, vergangene filtern
# 2. claim enhancement
# 3. retrieval

# hop_count +1 iterationen
for run_count in tqdm(range(5)):
    for hop_count in data:
        if run_count <= int(hop_count):
            # somit hop_count+1 Iterationen
            for key in data[hop_count]:
                claims_and_context = []
                for item in data[hop_count][key]:
                    # cross encoding
                    claim_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= item[f"claim_{run_count}"], titles=item[f"retrieved_{run_count}"])
                    if run_count > 0:
                        claim_sentence_pairs = cross_enc.claim_sentence_creator.filter_sentences(claim_sentence_pairs, item["previous_iteration_sentences"])
                        # Die 20 Sätze aller vorherigen Iteration werden raus gefiltert -> noch mit einer threshold probieren
                    prediction = cross_enc.predict(claim_sentence_pairs, return_probabilties=True)
                    prediciton_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
                    sentences_sorted = [item[1] for item in prediciton_sorted]
                    sentences_sorted_top20 = sentences_sorted[:20]

                    item[f"sentences_{run_count}"] = sentences_sorted_top20
                    item["previous_iteration_sentences"].extend(sentences_sorted_top20)

                    context = "\n".join(sentences_sorted_top20)
                    
                    claims_and_context.append({"claim": item[f"claim_{run_count}"], "context" : context})
                    #claims_and_context.append((item[f"claim_{run_count}"],context))
                
                # claim enhancement
                print(f"Running llm prediction, run count {run_count}, hop count {hop_count}, key {key}")
                #output = llm.predict_base(claims_and_context, postprocess=True)
                output = llm.batch(claims_and_context)
                print(output)
                for i in range(len(data[hop_count][key])):
                    data[hop_count][key][i][f"claim_{run_count+1}"] = output[i]

                # retrieval
                for item in data[hop_count][key]:
                    query = embedder.get_detailed_instruct(query=item[f"claim_{run_count+1}"], task_description=embedder.task)
                    db_output = vector_db.similarity_search(query, k = 100)
                    retrieved = []
                    for doc in db_output:
                        retrieved.append(doc.metadata["title"])
                    item[f"retrieved_{run_count+1}"] = retrieved

save_obj(data, "data/iterative_test2.json")
