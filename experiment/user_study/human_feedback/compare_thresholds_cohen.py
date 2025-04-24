import jsonlines
from specfix.utils import read_jsonl, unify_model_name
import numpy as np
# from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters # Keep if you want to compare
from sklearn.metrics import cohen_kappa_score # Use Cohen's Kappa
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

# taos = {
#     "deepseek-v3": {
#         "entropy": 0.0, 
#         "consistency": 1.0,
#         "ambiguity": 0.5,
#     },
#     "qwen2.5-coder-32b-instruct": {
#         "entropy": 0.0, 
#         "consistency": 1.0,
#         "ambiguity": 0.500,
#     },
#     "gpt-4o": {
#         "entropy": 0.0,
#         "consistency": 1.0,
#         "ambiguity": 0.5
#     }
# }    
ratings_data_cohen = {
    "we_entropy": {"metric": [], "ground_truth": []},
    "we_consistency": {"metric": [], "ground_truth": []},
    "we_ambiguity": {"metric": [], "ground_truth": []},
    "woe_entropy": {"metric": [], "ground_truth": []},
    "woe_consistency": {"metric": [], "ground_truth": []},
    "woe_ambiguity": {"metric": [], "ground_truth": []},
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

    valid_problems_count = 0 # Keep track of problems used

    for gt in ground_truth:
        p = None
        # ... (find corresponding problem p - same as before) ...
        for problem in humaneval_problems:
            if gt["task_id"] == problem["task_id"]:
                p = problem
                break
        if p is None:
            for problem in mbpp_problems:
                if gt["task_id"] == problem["task_id"]:
                    p = problem
                    break
        if p is None:
            # print(gt["task_id"], ": No corresponding original") # Optional: print missing
            continue

        if p["original_clusters"] is None or p["woe_clusters"] is None:
            continue

        valid_problems_count += 1 # Increment count for valid problems

        we_H = p["original_clusters"]["entropy"]
        we_S = p["original_clusters"]["weighted_test_consistency"]
        we_D = we_H * (1-we_S)

        woe_H = p["woe_clusters"]["entropy"]
        woe_S = p["woe_clusters"]["weighted_test_consistency"]
        woe_D = woe_H * (1-woe_S)

        ground_truth_decision = 1 if gt["answer"] == "Ambiguous" else 0

        # --- Collect ratings for each metric type ---
        # Metric decisions
        metric_we_entropy = 1 if we_H > taos[model]["entropy"] else 0
        metric_we_consistency = 1 if (1 - we_S) > taos[model]["consistency"] else 0
        metric_we_ambiguity = 1 if we_D > taos[model]["ambiguity"] else 0
        metric_woe_entropy = 1 if woe_H > taos[model]["entropy"] else 0
        metric_woe_consistency = 1 if (1 - woe_S) > taos[model]["consistency"] else 0
        metric_woe_ambiguity = 1 if woe_D > taos[model]["ambiguity"] else 0

        # Append to lists for Cohen's Kappa
        ratings_data_cohen["we_entropy"]["metric"].append(metric_we_entropy)
        ratings_data_cohen["we_entropy"]["ground_truth"].append(ground_truth_decision)

        ratings_data_cohen["we_consistency"]["metric"].append(metric_we_consistency)
        ratings_data_cohen["we_consistency"]["ground_truth"].append(ground_truth_decision)

        ratings_data_cohen["we_ambiguity"]["metric"].append(metric_we_ambiguity)
        ratings_data_cohen["we_ambiguity"]["ground_truth"].append(ground_truth_decision)

        ratings_data_cohen["woe_entropy"]["metric"].append(metric_woe_entropy)
        ratings_data_cohen["woe_entropy"]["ground_truth"].append(ground_truth_decision)

        ratings_data_cohen["woe_consistency"]["metric"].append(metric_woe_consistency)
        ratings_data_cohen["woe_consistency"]["ground_truth"].append(ground_truth_decision)

        ratings_data_cohen["woe_ambiguity"]["metric"].append(metric_woe_ambiguity)
        ratings_data_cohen["woe_ambiguity"]["ground_truth"].append(ground_truth_decision)


    print(f"Calculating Kappa based on {valid_problems_count} problems.")

    kappa_results_cohen = {}
    for metric_key, rating_lists in ratings_data_cohen.items():
        # Ensure there are ratings to compare
        if not rating_lists["metric"] or not rating_lists["ground_truth"]:
             print(f"Skipping {metric_key}: No data.")
             kappa_results_cohen[metric_key] = np.nan # Or handle as appropriate
             continue

        metric_ratings = rating_lists["metric"]
        gt_ratings = rating_lists["ground_truth"]

        # Check if all ratings are the same for either rater - Kappa is undefined/meaningless
        if len(set(metric_ratings)) <= 1 or len(set(gt_ratings)) <= 1:
             print(f"Skipping {metric_key}: All ratings are identical for at least one 'rater'. Kappa is ill-defined.")
             # Cohen's Kappa is often 0 or NaN in this case, depending on implementation.
             # It signifies no variance to measure agreement against chance.
             kappa_results_cohen[metric_key] = 0.0 # Or np.nan depending on desired handling
             # You might also want to report the constant value observed.
             # print(f"  Metric constant value: {metric_ratings[0] if metric_ratings else 'N/A'}")
             # print(f"  Ground Truth constant value: {gt_ratings[0] if gt_ratings else 'N/A'}")
             continue


        kappa = cohen_kappa_score(metric_ratings, gt_ratings)
        kappa_results_cohen[metric_key] = kappa

    print("\nCohen's Kappa Results:")
    for key, value in kappa_results_cohen.items():
        print(f"{key}: {value:.4f}") # Format for readability

if __name__ == "__main__":
    main()