import argparse
import math
import matplotlib.pyplot as plt
from specfix.cluster import Clusters
from specfix.utils import read_jsonl, calculate_mcc


def tune_threshold(results, ground_truth, model_name, label):
    threshold_dict = {}
    for threshold in range(0, math.ceil(max(results) * 1000), 1):
        threshold = threshold / 1000
        predicts = [1 if result > threshold else 0 for result in results]
        mcc = calculate_mcc(ground_truth, predicts)
        threshold_dict[threshold] = mcc
    # Find the best threshold
    best_threshold = max(threshold_dict, key=threshold_dict.get)
    plot_best_threshold(threshold_dict, best_threshold, model_name, label)
    return best_threshold


def plot_best_threshold(threshold_dict, best_threshold, model_name, label):
    max_mcc = threshold_dict[best_threshold]
    x_max = best_threshold
    y_max = max_mcc

    plt.figure(figsize=(10, 6))

    plt.plot(threshold_dict.keys(), threshold_dict.values(), label='MCC Curve')

    plt.scatter(x_max, y_max, color='red', zorder=5, label='Max MCC')

    plt.text(
        x=x_max + 0.05,
        y=y_max - 0.01,
        s=f'({x_max:.3f}, {y_max:.3f})',
        fontsize=10,
        ha='left'
    )

    plt.axvline(x=x_max, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=y_max, color='gray', linestyle='--', linewidth=1)

    plt.xlabel(f"{label} Threshold")
    plt.ylabel("MCC")
    plt.title(f"{model_name.capitalize()} - Best {label} Threshold: {x_max:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figure/{model_name}_{label.lower()}.png")
    plt.show()



def read_ground_truth():
    ground_truth = []
    humaneval = read_jsonl("humaneval_pilot_ground_truth.jsonl")
    humaneval_woe = read_jsonl("humaneval_woe_pilot_ground_truth.jsonl")
    mbpp = read_jsonl("mbpp_pilot_ground_truth.jsonl")
    mbpp_woe = read_jsonl("mbpp_woe_pilot_ground_truth.jsonl")
    # taco = read_jsonl("taco_lite_pilot_ground_truth.jsonl")
    # taco_woe = read_jsonl("taco_lite_woe_pilot_ground_truth.jsonl")
    return humaneval + humaneval_woe + mbpp + mbpp_woe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True)
    parser.add_argument("-d", "--dmetric", dest="metric", required=True, help="weighted_test_consistency, entropy, or ambiguity")

    options = parser.parse_args()
    model_name = options.model
    metric = options.metric

    ground_truth_problems = read_ground_truth()

    ambiguity_results = []
    humaneval_path = f"../../results/original/{model_name}-original-humaneval.jsonl"
    mbpp_path = f"../../results/original/{model_name}-original-mbpp.jsonl"
    humaneval_problems = read_jsonl(humaneval_path)
    mbpp_problems = read_jsonl(mbpp_path)
    all_problems = humaneval_problems[:50] + mbpp_problems[:50]
    for i, problem in enumerate(all_problems):
        # clusters = Clusters()
        # clusters.deserialize(problem["clusters"])
        if problem["original_clusters"] == None or problem["woe_clusters"] == None:
            # remove from ground truth set
            for i in range(len(ground_truth_problems) - 1, -1, -1):
                if ground_truth_problems[i]["task_id"] == problem["task_id"]:
                    del ground_truth_problems[i]
            
            continue
        
        
        if (metric == "ambiguity"):
            we_m = (problem["original_clusters"]["entropy"] * (1 - problem["original_clusters"]["weighted_test_consistency"])) 
            woe_m = (problem["woe_clusters"]["entropy"] * (1 - problem["woe_clusters"]["weighted_test_consistency"])) 
            ambiguity_results.append(we_m)
            ambiguity_results.append(woe_m)
        else:
            
            ambiguity_results.append(problem["original_clusters"][metric])
            ambiguity_results.append(problem["woe_clusters"][metric])
        # ambiguity_results.append(clusters.entropy)
        # ambiguity_results.append(len(clusters.cluster_list))
        # ambiguity_results.append(clusters.ambiguity)
        
    ground_truths = []
    for i, problem in enumerate(ground_truth_problems):
        ground_truths.append(problem["ground_truth"]) 
    
    
    best_threshold = tune_threshold(ambiguity_results, ground_truths, model_name, metric)
    print(f"Best threshold for all_datasets: {best_threshold}")


if __name__ == "__main__":
    main()
