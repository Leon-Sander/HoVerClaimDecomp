import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import sub_question_prompt, add_key_entities_refined_prompt
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import EnhancedBaseClaimsOutputParser, SubQuestionsOutputParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')


data = load_obj("data/iteration_full_data.json")

model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = create_llm_pipeline(model_id=model_id, device_map="cuda:0", load_in_8bit=False, load_in_4bit=True)
print("llm loaded")
llm_question_generator = create_chain_with_postprocessor(create_prompt(template=sub_question_prompt), 
                        llm,
                        stop=["QUESTIONS:", "CLAIM:"], 
                        postprocessor=SubQuestionsOutputParser)

llm_base_enhancement = create_chain_with_postprocessor(create_prompt(template=add_key_entities_refined_prompt), 
                        llm,
                        stop=["CONTEXT:", "CLAIM:"], 
                        postprocessor=EnhancedBaseClaimsOutputParser)

checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
print("Cross Enc Loaded")
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")

for run_count in tqdm(range(4)):
    for hop_count in data:
        if run_count <= int(hop_count) -1:
            # somit hop_count+1 retrievals, einmal mit base claim, n mal mit hop count claims
            for key in data[hop_count]:
                for item in data[hop_count][key]:
                    # sub question generation
                    sub_questions = llm_question_generator.invoke({"claim": item[f"claim_{run_count}"]})
                    item[f"sub_questions_{run_count}"] = sub_questions
                    item[f"sub_question_retrieval_{run_count}"] = []
                    
                    sentences_per_question = []
                    for question in sub_questions:
                        query = embedder.get_detailed_instruct(query=question, task_description=embedder.question_task)
                        db_output = vector_db.similarity_search(query, k = 100)
                        retrieved = []
                        for doc in db_output:
                            retrieved.append(doc.metadata["title"])
                        
                        item[f"sub_question_retrieval_{run_count}"].append(retrieved)


                        question_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= question, titles=retrieved)
                        prediction = cross_enc.predict(question_sentence_pairs, return_probabilities=True)
                        prediciton_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
                        sentences_sorted = [sentence[1] for sentence in prediciton_sorted]
                        sentences_per_question.append(sentences_sorted[:60])

                    
                    top_sentences = []
                    num_sentences_to_pick = 60
                    for index in range(num_sentences_to_pick):
                        for sentences_list in sentences_per_question:
                            if len(top_sentences) >= num_sentences_to_pick:
                                break
                            if index < len(sentences_list):
                                top_sentences.append(sentences_list[index])

                    #item[f"sub_question_top_sentences_{run_count}"] = sentences_per_question
                    #item[f"top_base_sentences_{run_count}"] = top_sentences

                    ##### enhancing base claim #####
                    enhanced_base_claim = llm_base_enhancement.invoke({"claim": item[f"claim_{run_count}"], "context" : "\n".join(top_sentences)})
                    item[f"claim_{run_count+1}"] = enhanced_base_claim
                
                    # base retrieval i+1
                    query = embedder.get_detailed_instruct(query=item[f"claim_{run_count+1}"], task_description=embedder.task)
                    db_output = vector_db.similarity_search(query, k = 100)
                    retrieved = []
                    for doc in db_output:
                        retrieved.append(doc.metadata["title"])
                    item[f"retrieved_{run_count+1}"] = retrieved

save_obj(data, "data/iterative_parallelism_with_questions_60_no_filter.json")





