import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch
from langchain.prompts import PromptTemplate
from output_parsers import *
#from langchain.llms.outputs import Generation

def postprocess_decomposed_claims(decomposed_claims : str) -> list[str]:
    output = []
    decomposed_claims = decomposed_claims.replace("\n\n", "\n").split("\n")
    for claim in decomposed_claims:
        if claim == "":
            continue
        output.append(claim.lstrip())
    return output

def create_prompt(template):
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_llm_pipeline(model_id,device_map="cuda:0", load_in_8bit=False, load_in_4bit=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )                                  
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map , quantization_config = bnb_config)
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    print("HF Pipeline created")
    llm_pipeline = HuggingFacePipeline(pipeline=hf_pipe)
    print("Langchain Pipeline created")
    return llm_pipeline


def create_chain(prompt, llm_pipeline, stop=["MULTI-HOP CLAIM:"]):
    chain = prompt | llm_pipeline.bind(stop=stop)
    return chain

def create_chain_with_postprocessor(prompt, llm_pipeline, stop=["MULTI-HOP CLAIM:"], postprocessor = None):
    if postprocessor is None:
        print("Chain created without postprocessor")
        return create_chain(prompt, llm_pipeline, stop)
    chain = prompt | llm_pipeline.bind(stop=stop) | postprocessor()
    return chain


class StopWordsCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever specified stop words are generated.

    Args:
        stop_words (List[str]): 
            A list of strings where each string is a stop word that should end the generation.
    """

    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(stop_word in text for stop_word in self.stop_words)

class StopwordCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words):
        self.stop_word_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in stop_words]  # each stop word is a single token

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] > 0 and input_ids[0, -1] in self.stop_word_ids:
            return True
        return False

class TransformerLLM():

    def __init__(self, model_id, prompt_template, postprocessor = None, stop_words=["CONTEXT:", "CLAIM:"]) -> None:
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map="cuda:0")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stopping_criteria = StoppingCriteriaList([StopwordCriteria(self.tokenizer, stop_words)])

        self.prompt_template = prompt_template
        if postprocessor is not None:
            self.postprocessor = postprocessor()

    def predict_decomposition(self, input_list, placeholder_text = "claim", postprocess = True):
        texts = []
        for input_text in input_list:
            texts.append(self.prompt_template.replace("{"+ placeholder_text +"}", input_text))

        inputs = self.tokenizer(texts, return_tensors="pt",padding=True).to("cuda:0")
        output_ids = self.model.generate(**inputs, max_new_tokens=512, stopping_criteria=self.stopping_criteria)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if postprocess:
            print("running postprocessor")
            return self.postprocessor.parse_batch(outputs)
            #output[0].split("DECOMPOSED CLAIMS:")[-1]
        else:
            return outputs
        
    def predict_base(self, claim_context_pairs, postprocess = True):
        texts = []
        for claim, context in claim_context_pairs:
            filled_prompt = self.prompt_template.replace("{claim}", claim)
            texts.append(filled_prompt.replace("{context}", context))

        inputs = self.tokenizer(texts, return_tensors="pt",padding=True).to("cuda:0")
        output_ids = self.model.generate(**inputs, max_new_tokens=512, stopping_criteria=self.stopping_criteria)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if postprocess:
            print("running postprocessor")
            return self.postprocessor.parse_batch(outputs)
            #output[0].split("DECOMPOSED CLAIMS:")[-1]
        else:
            return outputs