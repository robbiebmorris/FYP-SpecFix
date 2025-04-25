from specfix.utils import (
    read_jsonl,
)
from collections import Counter
import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from collections import Counter


data = read_jsonl("user_study_with_feedback.jsonl")
# Define the categories
categories = ["Ambiguous", "Unambiguous"]
category_map = {category: i for i, category in enumerate(categories)}
num_categories = len(categories)
num_raters = 3

# Create the N x k aggregate matrix
aggregate_counts = []

for row in data:
    feedback_list = row.get("human_feedback")
    if feedback_list and len(feedback_list) == num_raters:
        # Count occurrences of each category for this task
        counts = Counter(feedback_list)
        # Create a row for the aggregate matrix, ensuring correct order
        task_counts = [0] * num_categories
        for category, count in counts.items():
            if category in category_map:
                task_counts[category_map[category]] = count
            else:
                print(f"Warning: Unknown category '{category}' found in task {row.get('task_id')}. Skipping.")
                # Decide how to handle unknown categories if necessary
        aggregate_counts.append(task_counts)
    else:
         print(f"Warning: Task {row.get('task_id')} does not have exactly {num_raters} ratings. Skipping.")


# Convert to NumPy array (required by statsmodels)
# Shape will be N_tasks x N_categories
aggregate_matrix = np.array(aggregate_counts)
kappa_value = fleiss_kappa(aggregate_matrix, method='fleiss')


print("Aggregate Matrix (N_tasks x N_categories):")
print(f"Categories order: {categories}")
print(aggregate_matrix)

print(f"Fleiss' Kappa: {kappa_value:.4f}")