import jsonlines
from specfix.utils import read_jsonl, unify_model_name
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import argparse


taos = {
    "deepseek-v3": {
        "entropy": 0.935, 
        "consistency": 0.334,
        "ambiguity": 0.351,
    },
    "qwen2.5-coder-32b-instruct": {
        "entropy": 0.610, 
        "consistency": 0.950,
        "ambiguity": 0.500,
    },
    "gpt-4o": {
        "entropy": 0.971,
        "consistency": 0.000,
        "ambiguity": 0.375
    }
}    

taos = {
    "deepseek-v3": {
        "entropy": 0.0, 
        "consistency": 1.0,
        "ambiguity": 0.351,
    },
    "qwen2.5-coder-32b-instruct": {
        "entropy": 0.0, 
        "consistency": 1.0,
        "ambiguity": 0.500,
    },
    "gpt-4o": {
        "entropy": 0.0,
        "consistency": 1.0,
        "ambiguity": 0.5
    }
}    

ratings_data = {
    "we_entropy": [],
    "we_consistency": [],
    "we_ambiguity": [],
    "woe_entropy": [],
    "woe_consistency": [],
    "woe_ambiguity": [],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model")
    options = parser.parse_args()

    model = unify_model_name(options.model)

    humaneval_dataset_path = f"../../../results/original/{model}-original-humaneval.jsonl"
    mbpp_dataset_path = f"../../../results/original/{model}-original-mbpp.jsonl"
    ground_truth_path = "ground_truth.jsonl"

    humaneval_problems = read_jsonl(humaneval_dataset_path)
    mbpp_problems = read_jsonl(mbpp_dataset_path)
    ground_truth = read_jsonl(ground_truth_path)
    for gt in ground_truth:
        p = None
        for problem in humaneval_problems:
            if gt["task_id"] == problem["task_id"]:
                p = problem
                break;
        if p == None:
            for problem in mbpp_problems:
                if gt["task_id"] == problem["task_id"]:
                    p = problem
                    break;
        if p == None:
            print(gt["task_id"], ": No corresponding original")
            continue
        
        if p["original_clusters"] == None or p["woe_clusters"] == None:
            continue
        we_H = p["original_clusters"]["entropy"]
        we_S = p["original_clusters"]["weighted_test_consistency"]
        we_D = we_H * (1-we_S)
        
        woe_H = p["woe_clusters"]["entropy"]
        woe_S = p["woe_clusters"]["weighted_test_consistency"]
        woe_D = woe_H * (1-woe_S)
        
        
        ground_truth_decision = 1 if gt["answer"] == "Ambiguous" else 0

        # --- Collect ratings for each metric type ---

        ratings_data["we_entropy"].append([1 if we_H > taos[model]["entropy"] else 0] + [ground_truth_decision])
        ratings_data["we_consistency"].append([1 if (1 - we_S) > taos[model]["consistency"] else 0] + [ground_truth_decision])
        ratings_data["we_ambiguity"].append([1 if we_D > taos[model]["ambiguity"] else 0] + [ground_truth_decision])

        # WOE Metrics
        ratings_data["woe_entropy"].append([1 if woe_H > taos[model]["entropy"] else 0] + [ground_truth_decision])
        ratings_data["woe_consistency"].append([1 if (1 - woe_S) > taos[model]["consistency"] else 0] + [ground_truth_decision])
        ratings_data["woe_ambiguity"].append([1 if woe_D > taos[model]["ambiguity"] else 0] + [ground_truth_decision])

        

    kappa_results = {}
    for metric_key, ratings_list in ratings_data.items():
        ratings_matrix = np.array(ratings_list)
        agg_data, categories = aggregate_raters(ratings_matrix, n_cat=2) # Specify 2 categories (0 and 1)
        kappa = fleiss_kappa(agg_data, method='fleiss')
        kappa_results[metric_key] = kappa


    for key, value in kappa_results.items():
        print(f"{key}: {value}")
        
if __name__ == "__main__":
    main()
