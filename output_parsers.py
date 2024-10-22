from langchain_core.output_parsers.base import BaseOutputParser

class EnhancedBaseClaimsOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        text = text.replace("\n\n", "")
        text = text.replace("###", "")
        text = text.lstrip()
        return text

    @property
    def _type(self) -> str:
        return "Simple Parser"
    
class TrueFalseParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        # Assuming the first generation contains the desired output
        output_text = text.strip()
        
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

class SubQuestionsOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> list[str]:
        text = text.replace("\n\n", "")
        text = text.replace("###", "")
        text = text.lstrip()
        question_list = text.split("\n")
        question_list = [question for question in question_list if question != ""]
        return question_list

    @property
    def _type(self) -> str:
        return "Parsing LLM Output into a list of Questions or decomposed claims, can be used for both."




# output parsers for transformer models
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
    

class EnhancedClaimsOutputParser(BaseOutputParser[list[str]]):
    def __init__(self):
        super().__init__()
    
    def parse(self, text: str) -> list[str]:
        return text.split("ENHANCED CLAIM:")[-1].lstrip()
    
    def parse_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.parse(text) for text in texts]
    
class TransformerSubQuestionOutputParser(BaseOutputParser[list[str]]):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> list[str]:
        subquestions = text.split("QUESTIONS:")[-1]
        subquestions = subquestions.replace("###", "")
        subquestions = subquestions.strip()
        subquestions = subquestions.replace("\n\n\n", "\n")
        subquestions = subquestions.replace("\n\n", "\n").split("\n")
        question_list = [question.lstrip() for question in subquestions if question != "" and question != "CLAIM:"]
        return question_list

    def parse_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.parse(text) for text in texts]

    @property
    def _type(self) -> str:
        return "subquestion_output_parser"

    def dict(self, **kwargs: any) -> dict:
        output_parser_dict = super().dict(**kwargs)
        return output_parser_dict


class TransformerClaimRefinementOutputParser(BaseOutputParser[list[str]]):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> list[str]:
        refined_claim = text.split("REFINED:")[-1]
        refined_claim = refined_claim.replace("###", "")
        refined_claim = refined_claim.strip()
        return refined_claim

    def parse_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.parse(text) for text in texts]

    @property
    def _type(self) -> str:
        return "refined_claim_output_parser"

    def dict(self, **kwargs: any) -> dict:
        output_parser_dict = super().dict(**kwargs)
        return output_parser_dict
    
class TransformerDecomposedClaimsOutputParser(BaseOutputParser[list[str]]):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> list[str]:
        subquestions = text.split("DECOMPOSED:")[-1]
        subquestions = subquestions.replace("###", "")
        subquestions = subquestions.strip()
        subquestions = subquestions.replace("\n\n\n", "\n")
        subquestions = subquestions.replace("\n\n", "\n").split("\n")
        question_list = [question.lstrip() for question in subquestions if question != "" and question != "CLAIM:"]
        return question_list

    def parse_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.parse(text) for text in texts]

    @property
    def _type(self) -> str:
        return "decomposition_output_parser"

    def dict(self, **kwargs: any) -> dict:
        output_parser_dict = super().dict(**kwargs)
        return output_parser_dict