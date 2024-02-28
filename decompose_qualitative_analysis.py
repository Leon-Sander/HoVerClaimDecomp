import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from huggingface_llm_loading import create_prompt, create_llm_pipeline, create_chain
from utils import load_obj, save_obj
from  tqdm import tqdm
from prompt_templates import decompose_9shot

model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
chain = create_chain(create_prompt(template=decompose_9shot), 
                        create_llm_pipeline(model_id=model_id,
                                        device_map="cuda:0", load_in_8bit=False, load_in_4bit=True),
                        stop=["MULTI-HOP CLAIM:"])

if __name__ == "__main__":
    output = {
        "2": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "3": {"SUPPORTED": [], "NOT_SUPPORTED": []},
        "4": {"SUPPORTED": [], "NOT_SUPPORTED": []}}
    
    data = load_obj("qualitative_analysis_claims.json")

    for hop_count in data:
        for supported in data[hop_count]:
            for obj in tqdm(data[hop_count][supported]):
                answer = chain.invoke({"claim" : obj["claim"]})
                obj["decomposed_claims"] = answer
                output[hop_count][supported].append(obj)

    save_obj(data,"qualitative_analysis_decomposed_claims.json")