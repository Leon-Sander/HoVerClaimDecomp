import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
sys.path.append(str(Path("./cross_encoder").resolve()))
from cross_encoder.model import TextClassificationModel
from huggingface_llm_loading import TransformerLLM, create_chain, create_llm_pipeline, create_prompt, create_chain_with_postprocessor
from prompt_templates import add_context_prompt, sub_question_prompt, add_key_entities_refined_prompt
from custom_mistral_embedder import CustomMistralEmbedder
from output_parsers import EnhancedBaseClaimsOutputParser, SubQuestionsOutputParser
from utils import load_obj, load_vectordb, save_obj
from tqdm import tqdm
import torch

data = load_obj("data/iteration_full_data.json")
data2 = {}
for hop_count in data:
    data2[hop_count] = {}
    for key in data[hop_count]:
        data2[hop_count][key] = []
        for item in data[hop_count][key]:
            data2[hop_count][key].append({})

model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = TransformerLLM(model_id, device_map = "cuda:1")

print("llm Loaded")
checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)
print("Cross Enc Loaded")
embedder = CustomMistralEmbedder(gpu_count=1, batch_size=1)
vector_db = load_vectordb(embedder, "chroma_db_mistral", "wiki_data")
print("Embeder and Vector Db loaded")
for run_count in tqdm(range(4), desc='Run Count'):
    for hop_count in data:
        if run_count <= int(hop_count) - 1:
            for key in data[hop_count]:
                # Collect all claims in this group for batching
                all_claims = [item[f"claim_{run_count}"] for item in data[hop_count][key]]
                all_sub_questions = []
                all_top_sentences = []

                # Process subquestions in batches of 100
                for i in tqdm(range(0, len(all_claims), 50), desc='Processing Claims in Batches of 50'):
                    batch_claims = all_claims[i:i + 50]
                    sub_questions_batch = llm.generate_subquestions(batch_claims)
                    
                    # Collect sub-questions and prepare base claim enhancement context
                    for idx, sub_questions in enumerate(sub_questions_batch):
                        data[hop_count][key][i + idx][f"sub_questions_{run_count}"] = sub_questions
                        sentences_per_question = []
                        data[hop_count][key][i + idx][f"sub_question_retrieval_{run_count}"] = []

                        for question in sub_questions:
                            query = embedder.get_detailed_instruct(query=question, task_description=embedder.question_task)
                            db_output = vector_db.similarity_search(query, k=100)
                            retrieved = [doc.metadata["title"] for doc in db_output]
                            data[hop_count][key][i + idx][f"sub_question_retrieval_{run_count}"].append(retrieved)

                            question_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim=question, titles=retrieved)
                            prediction = cross_enc.predict(question_sentence_pairs, return_probabilities=True)
                            prediction_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
                            sentences_sorted = [sentence[1] for sentence in prediction_sorted]
                            sentences_per_question.append(sentences_sorted[:60])

                        # Collect sentences for base refinement
                        top_sentences = []
                        for index in range(60):
                            for sentences_list in sentences_per_question:
                                if len(top_sentences) >= 60:
                                    break
                                if index < len(sentences_list):
                                    top_sentences.append(sentences_list[index])

                        all_top_sentences.append("\n".join(top_sentences))
                    del sub_questions_batch

                torch.cuda.empty_cache()
                # Process base claim refinement in batches of 10
                for j in tqdm(range(0, len(all_claims), 10), desc='Refining Base Claims'):
                    base_claims_context = [(all_claims[j + k], all_top_sentences[j + k]) for k in range(10) if j + k < len(all_claims)]
                    enhanced_claims = llm.generate_claim_refinement(base_claims_context)
                    
                    # Update enhanced claims and retrieve new context
                    for k, enhanced_claim in enumerate(enhanced_claims):
                        data[hop_count][key][j + k][f"claim_{run_count + 1}"] = enhanced_claim
                        data2[hop_count][key][j + k][f"top_sentences_{run_count + 1}"] = all_top_sentences[j + k]
                        
                        query = embedder.get_detailed_instruct(query=enhanced_claim, task_description=embedder.task)
                        db_output = vector_db.similarity_search(query, k=100)
                        retrieved = [doc.metadata["title"] for doc in db_output]
                        data[hop_count][key][j + k][f"retrieved_{run_count + 1}"] = retrieved
                    del enhanced_claims



save_obj(data, "data/iterative_FULL_DATASET_with_questions_60_no_filter.json")
save_obj(data2, "data/iterative_FULL_DATASET_sentences.json")

