import unicodedata
from utils import load_obj

def is_successful_retrieval(obj, retrieval_key):
    #print(len(obj[retrieval_key]))
    retrieved_titles = set(unicodedata.normalize('NFD', title) for title in obj[retrieval_key])
    
    for fact in obj.get('supporting_facts', []):
        normalized_fact_title = unicodedata.normalize('NFD', fact[0])
        if normalized_fact_title not in retrieved_titles:
            return False
            
    return True


def calculate_success_percentage(file_path, dataset_type, embedder_name, k):
    #file_path = f'{embedder_name}_retrieval_output_{dataset_type}_{k}.json'
    #file_path ="decomposed_mistral_retrieval_output_dev_100_perclaim_few_shot.json"
    #file_path = "mistral_retrieval_output_dev_100_likedecomposed_few_shot.json"
    
    data = load_obj(file_path)
    hop_counts = {2: {'total': 0, 'successful': 0},
                    3: {'total': 0, 'successful': 0},
                    4: {'total': 0, 'successful': 0}}

    for obj in data:
        num_hops = obj['num_hops']
        # only for supported claims to compare with baleen
        if obj["label"] == "SUPPORTED":
            hop_counts[num_hops]["total"] += 1
            if is_successful_retrieval(obj, retrieval_key="retrieved"):
                hop_counts[num_hops]['successful'] += 1

    success_percentages = {}
    for num_hops, counts in hop_counts.items():
        if counts['total'] > 0:
            success_percentage = (counts['successful'] / counts['total']) * 100
            success_percentages[num_hops] = success_percentage


    average_total_percentage = (success_percentages[2] + success_percentages[3] + success_percentages[4]) / 3
    # Example variables
    method = embedder_name
    dataset = dataset_type
    retrieved = k
    hops_2 = round(success_percentages[2],2)
    hops_3 = round(success_percentages[3],2)
    hops_4 = round(success_percentages[4],2)
    avg_total = round(average_total_percentage,2)

    latex_line = f"{method} & {dataset} & {retrieved} & {hops_2}\% & {hops_3}\% & {hops_4}\% & {avg_total}\% \\\\ \\hline\n"    
    print(latex_line)
    print(hop_counts)
    return success_percentages, average_total_percentage

def main():
    embedder_name = "mistral"
    file_path = "mistral_retrieval_output_dev_100_multi_hop_prompt.json"
    #calculate_success_percentage(dataset_type="train", embedder_name=embedder_name, k=100)
    calculate_success_percentage(file_path = file_path, dataset_type="dev", embedder_name=embedder_name, k=100)
    #calculate_success_percentage(dataset_type="train", embedder_name=embedder_name, k=1000)
    #calculate_success_percentage(dataset_type="dev", embedder_name=embedder_name, k=1000)

if __name__ == "__main__":
    main()