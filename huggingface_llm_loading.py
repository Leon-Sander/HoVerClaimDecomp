import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, StoppingCriteriaList, LogitsProcessorList
from stopping_criteria import StopwordCriteria, StopwordLogitsProcessor
import torch
from langchain.prompts import PromptTemplate
from output_parsers import *
from torch.nn import DataParallel
from prompt_templates import sub_question_prompt, add_key_entities_refined_prompt_4shot, decompose_9shot_instruct,decompose_without_redundancy, decompose_entity_based


def create_prompt(template):
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_llm_pipeline(model_id,device_map="cuda"):
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


class TransformerLLM():

    def __init__(self, model_id, device_map) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map=self.device_map)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        subquestion_stop_words = ["\n\nCLAIM:", "CLAIM:", "\nCLAIM: ", "CLAIM: ", "QUESTIONS:", "QUESTIONS: ", "\nQUESTIONS: ", ",\nQUESTIONS:"]
        logits_processor = StopwordLogitsProcessor(self.tokenizer, subquestion_stop_words, self.tokenizer.eos_token_id)
        self.logits_processor_subquestions = LogitsProcessorList([logits_processor])
        self.prompt_template_subquestions = sub_question_prompt
        self.output_parser_subquestions = TransformerSubQuestionOutputParser()

        claim_refinement_stop_words = ["\n\nCLAIM:", "CLAIM:", "\nCLAIM: ", "CLAIM: ", "REFINED:"]
        self.logits_processor_claim_refinement = LogitsProcessorList([StopwordLogitsProcessor(self.tokenizer, claim_refinement_stop_words, self.tokenizer.eos_token_id)])
        self.prompt_template_claim_refinement = add_key_entities_refined_prompt_4shot
        self.output_parser_claim_refinement = TransformerClaimRefinementOutputParser()

        decomposition_stop_words = ["\n\nCLAIM:", "CLAIM:", "\nCLAIM: ", "CLAIM: ", "DECOMPOSED:"]
        self.logits_processor_decomposition = LogitsProcessorList([StopwordLogitsProcessor(self.tokenizer, decomposition_stop_words, self.tokenizer.eos_token_id)])
        self.prompt_template_decomposition = decompose_entity_based
        self.output_parser_decomposition = TransformerDecomposedClaimsOutputParser()
        


    def generate_decomposition(self, claim_batch, postprocess = True):
        texts = []
        for claim in claim_batch:
            texts.append(self.prompt_template_decomposition.replace("{claim}", claim))

        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt",padding=True).to(self.device_map)
            #output_ids = self.model.generate(**inputs, max_new_tokens=512, stopping_criteria=self.stopping_criteria_subquestions)
            output_ids = self.model.generate(**inputs, max_new_tokens=320, logits_processor=self.logits_processor_decomposition)
            del inputs
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            del output_ids
        if postprocess:
            #print("running postprocessor")
            return self.output_parser_decomposition.parse_batch(outputs)
        else:
            return outputs
        
    def generate_claim_refinement(self, claim_context_pairs, postprocess = True):
        texts = []
        for claim, context in claim_context_pairs:
            filled_prompt = self.prompt_template_claim_refinement.replace("{claim}", claim)
            texts.append(filled_prompt.replace("{context}", context))

        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt",padding=True).to(self.device_map)
            output_ids = self.model.generate(**inputs, max_new_tokens=320, logits_processor=self.logits_processor_claim_refinement)
            del inputs
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            del output_ids

        if postprocess:
            #print("running postprocessor")
            return self.output_parser_claim_refinement.parse_batch(outputs)
        else:
            return outputs
        
    def generate_subquestions(self, claim_batch, postprocess = True):
        texts = []
        for claim in claim_batch:
            texts.append(self.prompt_template_subquestions.replace("{claim}", claim))

        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt",padding=True).to(self.device_map)
            #output_ids = self.model.generate(**inputs, max_new_tokens=512, stopping_criteria=self.stopping_criteria_subquestions)
            output_ids = self.model.generate(**inputs, max_new_tokens=320, logits_processor=self.logits_processor_subquestions)
            del inputs
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            del output_ids
        if postprocess:
            #print("running postprocessor")
            return self.output_parser_subquestions.parse_batch(outputs)
        else:
            return outputs
        
    def forward(self, input_text, **kwargs):
        generation_type = kwargs["kwargs"]["generation_type"]
        if generation_type == "decompose":
            return self.predict_decomposition(input_text)
        elif generation_type == "subquestions":
            return self.generate_subquestions(input_text)
        elif generation_type == "claim_refinement":
            return self.generate_base_refinement(input_text)
        else:
            print("Invalid type")
            return None