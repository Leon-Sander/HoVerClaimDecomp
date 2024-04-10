import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
from pathlib import Path

sys.path.append(str(Path("../").resolve()))
from utils import load_obj, save_obj
from cross_encoder.create_sentences_dict import ClaimSentencePairsCreator
from cross_encoder.model import TextClassificationModel

from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/qnli-electra-base')
model_type = "cross_enc"
#checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-16_23-01-03/bestckpt_epoch=1_val_loss=0.18.ckpt" # without title
checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)

data = load_obj("../qualitative_analysis/qualitative_analysis_train_base_retrieval_10.json")

from tqdm import tqdm
index = []
for hop_count in data:
    for supported in data[hop_count]:
        for obj in tqdm(data[hop_count][supported]):
            claim_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= obj["claim"], titles=obj["base_retrieved"])
            supporting_sentences = cross_enc.claim_sentence_creator.get_supporting_sentences(obj, "base_retrieved")
            if model_type == "transformer":
                output = model.predict(claim_sentence_pairs)
                prediction = []
                for i, tpl in enumerate(claim_sentence_pairs):
                    prediction.append((tpl[0], tpl[1], output[i]))
                
            else:
                prediction = cross_enc.predict(claim_sentence_pairs, return_probabilties=True)
            
            
            
            prediciton_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
            sentences_sorted = [item[1] for item in prediciton_sorted]
            for sentence in supporting_sentences:
                if sentence in sentences_sorted:
                    index.append(sentences_sorted.index(sentence))

def percentage_of_numbers_below_threshold(numbers : list[int], threshold: int):
    return sum(1 for x in numbers if x < threshold) / len(numbers) * 100

def calculate_percentages_refined(numbers : list[int],thresholds: list[int]):
    """
    Calculates the percentages of numbers in the list that are below certain thresholds.
    This version uses a separate function to count numbers below each threshold.
    
    Parameters:
    - numbers: List of numbers to evaluate.
    - thresholds: List of thresholds to check against.
    
    Returns:
    - A dictionary with thresholds as keys and percentages of numbers below each threshold as values.
    """
    percentages = {}
    
    for threshold in thresholds:
        percentages[threshold] = percentage_of_numbers_below_threshold(numbers, threshold)
    return percentages

thresholds = [5, 10, 15, 20, 25, 30, 40]
print(calculate_percentages_refined(index, thresholds))