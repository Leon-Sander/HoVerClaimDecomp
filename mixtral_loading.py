import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, MistralForCausalLM
from langchain.prompts import PromptTemplate
from utils import load_obj, save_obj
from tqdm import tqdm
from prompt_templates import decompose_9shot, decompose_enriched

def create_prompt(template):
    #<s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_llm_pipeline(model_id,device_map="cuda:0", load_in_8bit=False, load_in_4bit=False):                                     
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map , load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)
    #model = MistralForCausalLM.from_pretrained(model_id, device_map=device_map , load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    print("HF Pipeline created")
    llm_pipeline = HuggingFacePipeline(pipeline=hf_pipe)
    print("Langchain Pipeline created")
    return llm_pipeline


def create_chain(prompt, llm_pipeline):
    chain = prompt | llm_pipeline.bind(stop=["MULTI-HOP CLAIM:"])
    return chain

if __name__ == "__main__":
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    #model_id= "mistralai/Mistral-7B-v0.1"
    chain = create_chain(create_prompt(template=decompose_9shot), 
                         create_llm_pipeline(model_id=model_id,
                                            device_map="cuda:0", load_in_8bit=False, load_in_4bit=True))

    data = load_obj(f"data/hover_dev_300_claims.json")
    decomposed = []
    for item in tqdm(data):
        decomposed_claim = chain.invoke({"claim": item["claim"]})
        copied_item = item.copy()
        copied_item["decomposed_claims"] = decomposed_claim
        decomposed.append(copied_item)
    
    save_obj(decomposed, "decomposed_300_9shot2.json")
