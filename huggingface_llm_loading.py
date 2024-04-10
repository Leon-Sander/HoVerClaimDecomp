import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, MistralForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
#from langchain.llms.outputs import Generation

"""class SimpleTrueFalseParser(BaseOutputParser[str]):
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> str:
        # Assuming the first generation contains the desired output
        output_text = result[0].text.strip()
        
        # Extract "True" or "False" from the beginning of the output
        if output_text.startswith("True"):
            return "True"
        elif output_text.startswith("False"):
            return "False"
        else:
            return "Error: Output does not start with True or False"

    @property
    def _type(self) -> str:
        return "simple_true_false_parser"
    """

class EnhancedClaimsOutputParser(BaseOutputParser[list[str]]):
    def __init__(self):
        super().__init__()
    
    def parse(self, text: str) -> list[str]:
        return text.split("ENHANCED CLAIM:")[-1].lstrip()
    
    def parse_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.parse(text) for text in texts]

class DecomposedClaimsOutputParser(BaseOutputParser[list[str]]):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> list[str]:
        output = []
        decomposed_claims = text.split("DECOMPOSED CLAIMS:")[-1]
        decomposed_claims = decomposed_claims.replace("\n\n\n", "\n")
        decomposed_claims = decomposed_claims.replace("\n\n", "\n").split("\n")
        for claim in decomposed_claims:
            if claim == "":
                continue
            output.append(claim.lstrip())
        return output

    def parse_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.parse(text) for text in texts]

    @property
    def _type(self) -> str:
        return "decomposed_claims_output_parser"

    def dict(self, **kwargs: any) -> dict:
        output_parser_dict = super().dict(**kwargs)
        return output_parser_dict

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
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map , load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)
    #model = MistralForCausalLM.from_pretrained(model_id, device_map=device_map , load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit)
    hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    print("HF Pipeline created")
    llm_pipeline = HuggingFacePipeline(pipeline=hf_pipe)
    print("Langchain Pipeline created")
    return llm_pipeline


def create_chain(prompt, llm_pipeline, stop=["MULTI-HOP CLAIM:"]):
    chain = prompt | llm_pipeline.bind(stop=stop)
    return chain

def create_chain_with_postprocessor(prompt, llm_pipeline, stop=["MULTI-HOP CLAIM:"], postprocessor = DecomposedClaimsOutputParser):
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
        # Convert the current output tokens to text
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Check if any of the stop words appear in the text
        return any(stop_word in text for stop_word in self.stop_words)

class StopwordCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words):
        self.stop_word_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in stop_words]  # assuming each stop word is a single token

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is a stopword
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

#model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
#model_id= "mistralai/Mistral-7B-Instruct-v0.2"
#chain = create_chain(create_prompt(), 
                        #create_llm_pipeline(model_id=model_id,
                         #               device_map="cuda:0", load_in_8bit=True, load_in_4bit=False))