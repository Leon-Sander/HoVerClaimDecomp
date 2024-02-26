from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, MistralForCausalLM
from langchain.prompts import PromptTemplate
#from langchain.llms.output_parsers import BaseOutputParser
#from langchain.llms.outputs import Generation
from typing import List

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

#model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
#model_id= "mistralai/Mistral-7B-Instruct-v0.2"
#chain = create_chain(create_prompt(), 
                        #create_llm_pipeline(model_id=model_id,
                         #               device_map="cuda:0", load_in_8bit=True, load_in_4bit=False))