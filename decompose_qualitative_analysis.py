import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from huggingface_llm_loading import create_prompt, create_llm_pipeline, create_chain
from utils import load_obj, save_obj
from  tqdm import tqdm
from prompt_templates import decompose_9shot_instruct, decompose_9shot_noinstruct, decompose_6_2, decompose_6_3, decompose_add_context, decompose_add_context_noinstruct

model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
chain = create_chain(create_prompt(template=decompose_9shot_noinstruct), 
                        create_llm_pipeline(model_id=model_id,
                                        device_map="cuda:0", load_in_8bit=False, load_in_4bit=True),
                        stop=["MULTI-HOP CLAIM:"])#["CLAIM:","CONTEXT:", "</s>"])

if __name__ == "__main__":
    output = {
        "2": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "3": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "4": {"SUPPORTED": [], "NOT_SUPPORTED": []}}
    
    #data = load_obj("data/qualitative_analysis_claims_10.json")
    data = load_obj("qualitative_analysis_claims_10_base_sentences.json")

    
    for hop_count in data:
        for supported in data[hop_count]:
            for obj in tqdm(data[hop_count][supported]):
                top_k_sentences = sorted(obj["claim_sentence_pairs"], key=lambda x: x[2], reverse=True)[:20]
                sentences = [t[1] for t in top_k_sentences]
                context = "\n".join(sentences)

                answer = chain.invoke({"claim" : obj["claim"]})#, "context" : context})
                obj["decomposed_claims"] = answer
                output[hop_count][supported].append(obj)

    save_obj(data,"qualitative_analysis_decomposed_claims_10_9shot_noinstruct_for_context_comparison.json")