import os
from tqdm import tqdm
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append(str(Path("../").resolve()))
from utils import load_obj, save_obj
from cross_encoder.create_sentences_dict import ClaimSentencePairsCreator, ClaimSentencePairsCreator_NewDb
from cross_encoder.model import TextClassificationModel

from sentence_transformers import CrossEncoder



def run_on_sample_data():
    index = []
    for hop_count in data:
        for supported in data[hop_count]:
            for obj in tqdm(data[hop_count][supported]):
                claim_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= obj["claim"], titles=obj["base_retrieved"])
                supporting_sentences = cross_enc.claim_sentence_creator.get_supporting_sentences(obj, "base_retrieved")
                if model_type == "transformer":
                    output = cross_enc.predict(claim_sentence_pairs)
                    prediction = []
                    for i, tpl in enumerate(claim_sentence_pairs):
                        prediction.append((tpl[0], tpl[1], output[i]))
                    
                else:
                    prediction = cross_enc.predict(claim_sentence_pairs, return_probabilities=True)
                
                
                
                prediciton_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
                sentences_sorted = [item[1] for item in prediciton_sorted]
                obj["sentences"] = sentences_sorted
                for sentence in supporting_sentences:
                    if sentence in sentences_sorted:
                        index.append(sentences_sorted.index(sentence))

def run_on_whole_dataset():
    index = []
    for obj in tqdm(data):
        claim_sentence_pairs = cross_enc.claim_sentence_creator.create_claim_sentence_pairs(claim= obj["claim"], titles=obj["retrieved"])
        supporting_sentences = cross_enc.claim_sentence_creator.get_supporting_sentences(obj, "retrieved")
        if model_type == "transformer":
            output = cross_enc.predict(claim_sentence_pairs)
            prediction = []
            for i, tpl in enumerate(claim_sentence_pairs):
                prediction.append((tpl[0], tpl[1], output[i]))
            
        else:
            prediction = cross_enc.predict(claim_sentence_pairs, return_probabilities=True)
        
        
        
        prediciton_sorted = sorted(prediction, key=lambda x: x[2], reverse=True)
        sentences_sorted = [item[1] for item in prediciton_sorted]
        obj["sentences"] = sentences_sorted
        for sentence in supporting_sentences:
            if sentence in sentences_sorted:
                index.append(sentences_sorted.index(sentence))
    return index

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

def baseline_eval(data):
    new_claim_sentence_creator = ClaimSentencePairsCreator_NewDb()
    data = load_obj("data/mistral_retrieval_output_dev_100.json")
    total_supporting_sentences = 0
    found_supporting_sentences = 0
    for obj in tqdm(data):
        claim_sentence_pairs = new_claim_sentence_creator.get_splitted_sentences(obj["retrieved"])
        supporting_sentences = new_claim_sentence_creator.get_supporting_sentences(obj, "retrieved")
        top60 = claim_sentence_pairs[:60] 
        
        for sentence in supporting_sentences:
            if sentence in top60:
                found_supporting_sentences += 1
            total_supporting_sentences += 1


    print(f"From all the supporting sentences within the 300 sentences, {found_supporting_sentences} were found in the top 60 sentences. That is {found_supporting_sentences/total_supporting_sentences*100}%")


if __name__ == "__main__":
    #model = CrossEncoder('cross-encoder/qnli-electra-base')
    model_type = "cross_enc"
    #checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-16_23-01-03/bestckpt_epoch=1_val_loss=0.18.ckpt" # without title
    checkpoint_path = "/home/sander/code/thesis/hover/leon/cross_encoder/checkpoints/2024-03-14_17-16-50/bestckpt_epoch=2_val_loss=0.19.ckpt"
    cross_enc = TextClassificationModel.load_from_checkpoint(checkpoint_path, model_name="bert-base-uncased", mode="predict", with_title=True)

    data = load_obj("../data/mistral_retrieval_output_dev_100.json")
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    index = run_on_whole_dataset()
    print(calculate_percentages_refined(index, thresholds))
    save_obj(data, "cross_enc_threshold_calculation.json")
    save_obj(data, "../data/cross_enc_threshold_calculation.json")