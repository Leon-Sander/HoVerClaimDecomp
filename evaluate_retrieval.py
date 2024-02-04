import unicodedata
from utils import load_obj

def is_successful_retrieval(obj):
    retrieved_titles = set(unicodedata.normalize('NFD', title) for title in obj['retrieved'])
    
    for fact in obj.get('supporting_facts', []):
        normalized_fact_title = unicodedata.normalize('NFD', fact[0])
        if normalized_fact_title not in retrieved_titles:
            return False
            
    return True

def calculate_success_percentage(dataset_type, embedder_name, k):
    file_path = f'{embedder_name}_retrieval_output_{dataset_type}_{k}.json'
    data = load_obj(file_path)

    if dataset_type == "dev":
        hop_counts = {2: {'total': 1126, 'successful': 0},
                    3: {'total': 1835, 'successful': 0},
                    4: {'total': 1039, 'successful': 0}}
    elif dataset_type == "train":
        hop_counts = {2: {'total': 9052, 'successful': 0},
                    3: {'total': 6084, 'successful': 0},
                    4: {'total': 3035, 'successful': 0}}# 

    for obj in data:
        num_hops = obj['num_hops']
        if is_successful_retrieval(obj):
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
    return success_percentages, average_total_percentage

def main():
    embedder_name = "bert"
    calculate_success_percentage(dataset_type="train", embedder_name=embedder_name, k=100)
    calculate_success_percentage(dataset_type="dev", embedder_name=embedder_name, k=100)
    calculate_success_percentage(dataset_type="train", embedder_name=embedder_name, k=1000)
    calculate_success_percentage(dataset_type="dev", embedder_name=embedder_name, k=1000)

if __name__ == "__main__":
    main()