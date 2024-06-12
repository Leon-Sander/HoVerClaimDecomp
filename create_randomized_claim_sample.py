import sys
from pathlib import Path

sys.path.insert(0, str(Path("../").resolve()))
from utils import load_obj, save_obj
import random

from nltk.tokenize import word_tokenize
from tqdm import tqdm

def categorize_and_add_data(data_list, nested_dict, number_of_claims):
    #randomized
    for hop_count in nested_dict.keys():
        supported_data = [d for d in data_list if d['num_hops'] == int(hop_count) and d['label'] == 'SUPPORTED']
        not_supported_data = [d for d in data_list if d['num_hops'] == int(hop_count) and d['label'] == 'NOT_SUPPORTED']

        nested_dict[hop_count]['SUPPORTED'] = random.sample(supported_data, number_of_claims)
        nested_dict[hop_count]['NOT_SUPPORTED'] = random.sample(not_supported_data, number_of_claims)
        
    return nested_dict

def create_qualitative_analysis_data_sample(number_of_claims, data_path ,save_path = None):

    data = load_obj(data_path)
    data_sample = {
        "2": {"SUPPORTED": None, "NOT_SUPPORTED": None},
        "3": {"SUPPORTED": None, "NOT_SUPPORTED": None},
        "4": {"SUPPORTED": None, "NOT_SUPPORTED": None}
    }
    data_sample = categorize_and_add_data(data, data_sample, number_of_claims)
    if save_path:
        save_obj(data_sample, save_path)
    return data_sample

def create_counterpart_data_sample(save_path = None, data_path = "data/dev_claims_with_not_supported_counterpart.json", number_of_claims = 10):
    #randomized
    data = load_obj(data_path)
    #output_sample = {"2": [], "3": [], "4": []}
    output_sample = {"3": [], "4": []}
    for hop_count in output_sample.keys():
        supported_data = [item for item in data if item['num_hops'] == int(hop_count) and item['label'] == 'SUPPORTED']
        output_sample[hop_count] = random.sample(supported_data, number_of_claims)
    
    if save_path:
        save_obj(output_sample, save_path)
    return output_sample

def get_similar_claims(claim, hover_data):
    """To find NOT_SUPPORTED claims which are created out of negation, or entity relacement from SUPPORTED claims and vice versa
    """
    claim_words = word_tokenize(claim)
    similar_claims_found = []
    #hover_data = load_obj("/home/sander/code/thesis/hover/data/hover/hover_dev_release_v1.1.json")
    for claim_index, item in enumerate(hover_data):
        if item["label"] == "NOT_SUPPORTED":
            claim_to_check_words = word_tokenize(item['claim'])
            if len(list(set(claim_words).intersection(claim_to_check_words))) > 0.7 * len(list(set(claim_words))):
                similar_claims_found.append((claim_index, item["claim"]))
    return similar_claims_found

def get_not_supported_counterpart_claims(data):
    for item in data:
        if "similar_claims" in item.keys():
            if item["similar_claims"] != []:
                for index, claim in item["similar_claims"]:
                    if item["supporting_facts"] == data[index]["supporting_facts"]:
                        item["not_supported_counterpart"] = data[index]

    return [item for item in data if "not_supported_counterpart" in item.keys()]
    


def create_not_supported_counterpart_dataset(data_path, save_path):
    data = load_obj(data_path)
    for item in tqdm(data):
        if item["label"] == "SUPPORTED":
            item["similar_claims"] = get_similar_claims(item["claim"], data)

    not_supported_counterpart_data =get_not_supported_counterpart_claims(data)
    save_obj(not_supported_counterpart_data, save_path)


def prepare_for_iteration(items):
    for item in items:
        item["previous_iteration_sentences"] = []
        item["claim_0"] = item["claim"]
        item["retrieved_0"] = item["retrieved"]
        del(item["claim"])
        del(item["retrieved"])
    return items

def create_full_dataset(data, save_path=None):
    """
    One retrieval has to be done beforehand
    """
    data_categorized = {
        "2": {"SUPPORTED": None, "NOT_SUPPORTED": None},
        "3": {"SUPPORTED": None, "NOT_SUPPORTED": None},
        "4": {"SUPPORTED": None, "NOT_SUPPORTED": None}
    }
        
    for hop_count in data_categorized.keys():
        supported_data = [d for d in data if d['num_hops'] == int(hop_count) and d['label'] == 'SUPPORTED']
        not_supported_data = [d for d in data if d['num_hops'] == int(hop_count) and d['label'] == 'NOT_SUPPORTED']
        supported_data = prepare_for_iteration(supported_data)
        not_supported_data = prepare_for_iteration(not_supported_data)

        data_categorized[hop_count]['SUPPORTED'] = supported_data
        data_categorized[hop_count]['NOT_SUPPORTED'] = not_supported_data
    
    if save_path:
        save_obj(data_categorized, save_path)

    return data_categorized


if __name__ == "__main__":
    #create_not_supported_counterpart_dataset(data_path = "/home/sander/code/thesis/hover/data/hover/hover_train_release_v1.1.json",
    #                                         save_path = "data/train_not_supported_counterpart.json")
    #create_counterpart_data_sample(save_path = "data/100_data_points.json", 
    #                               data_path = "data/train_not_supported_counterpart.json", 
    #                               number_of_claims = 25)
    data_path = "/home/sander/code/thesis/hover/data/hover/hover_train_release_v1.1.json"
    create_qualitative_analysis_data_sample(number_of_claims = 50, data_path = data_path ,save_path = "data/train_qualitative_analysis_100.json")