from specfix.utils import read_jsonl, count_passk, count_passk_ambiguous


def analyze(model, dataset):
    results = read_jsonl(f"test_based_repair/{model}/{dataset}.jsonl")
    for result in results:
        if result["original_result"] == True and result["repaired_result"] == False:
            print(result["requirement"])
            print(result["repaired_requirement"])
            print(result["original_result"])
            print(result["repaired_result"])
            print(result["original_clusters"]["entropy"])
            print(result["original_clusters"]["weighted_test_consistency"])
            print(result["repaired_failed_inputs_outputs"])
            print()


def count_repairs(path):
    results = read_jsonl(path)
    total = 0
    total_fail = 0
    total_success = 0
    entropy_reduced = 0
    entropy_increased = 0
    noop = 0
    for result in results:
        # print(result["result"].keys())
        if result["result"]["repaired_passk"]:
            total = total + 1
        else:
            continue
        if result["result"]["original_passk"] > result["result"]["repaired_passk"]:
            total_fail = total_fail + 1
        if result["result"]["original_passk"] < result["result"]["repaired_passk"]:
            total_success = total_success + 1
            # print(result["requirement"])
            # print(result["repaired_requirement"])
            # print("original_passk", result["result"]["original_passk"])
            # print("original entropy", result["original_clusters"]["entropy"])
            # print("original t-consistency", result["original_clusters"]["weighted_test_consistency"])
            # print('-' *10)
            # print("repaired_passk", result["result"]["repaired_passk"])
            # print("repaired entropy", result["repaired_clusters"]["entropy"])
            # print("repaired t-consistency", result["repaired_clusters"]["weighted_test_consistency"])
            # print(result["repaired_failed_inputs_outputs"])
            # print()
        if (result["original_clusters"]["entropy"] > result["repaired_clusters"]["entropy"]):
            entropy_reduced = entropy_reduced + 1
            # print(result["requirement"])
            # print(result["repaired_requirement"])
            # print("original_passk", result["result"]["original_passk"])
            # print("original entropy", result["original_clusters"]["entropy"])
            # print("original t-consistency", result["original_clusters"]["weighted_test_consistency"])
            # print('-' *10)
            # print("repaired_passk", result["result"]["repaired_passk"])
            # print("repaired entropy", result["repaired_clusters"]["entropy"])
            # print("repaired t-consistency", result["repaired_clusters"]["weighted_test_consistency"])
            # # print(result["repaired_failed_inputs_outputs"])
            # print()
        if (result["original_clusters"]["entropy"] < result["repaired_clusters"]["entropy"]):
            entropy_increased = entropy_increased + 1
        
        if (result["original_clusters"]["entropy"] == result["repaired_clusters"]["entropy"] and result["result"]["original_passk"] == result["result"]["repaired_passk"]):
            noop = noop + 1 
            print(result["requirement"])
            print(result["repaired_requirement"])
            print("original_passk", result["result"]["original_passk"])
            print("original entropy", result["original_clusters"]["entropy"])
            print("original t-consistency", result["original_clusters"]["weighted_test_consistency"])
            print('-' *10)
            print("repaired_passk", result["result"]["repaired_passk"])
            print("repaired entropy", result["repaired_clusters"]["entropy"])
            print("repaired t-consistency", result["repaired_clusters"]["weighted_test_consistency"])
            print()
            

    print('-' * 10)
    print("Total repairs:", total)
    print("Total failures:", total_fail)
    print("Total successes:", total_success)
    print("Total entropy reduced:", entropy_reduced)
    print("Total entropy increased:", entropy_increased)
    print("No-ops:", noop)

print("deepseek (humaneval, mbpp):")
count_repairs("../../results/Robbie_dataset/specfix_deepseek_humaneval.jsonl")
count_repairs("../../results/Robbie_dataset/specfix_deepseek_mbpp.jsonl")

print("gpt4o (humaneval, mbpp)")
count_repairs("../../results/Robbie_dataset/specfix_gpt_humaneval.jsonl")
count_repairs("../../results/Robbie_dataset/specfix_gpt_mbpp.jsonl")

print("qwen (humaneval, mbpp)")
count_repairs("../../results/Robbie_dataset/specfix_qwen_humaneval.jsonl")
count_repairs("../../results/Robbie_dataset/specfix_qwen_mbpp.jsonl")

# count_result("deepseek-v3", "mbpp_25_woe")
# analyze("deepseek-v3", "taco_lite_0.125")
# count_passk("test_based_repair","qwen2.5-coder-32b-instruct", "taco_lite_0.306")
# count_passk_ambiguous("test_based_repair","qwen2.5-coder-32b-instruct", "taco_lite_0.306")
# count_ambiguity("test_based_repair","qwen2.5-coder-32b-instruct", "taco_lite_0.306")
# count_passk("test_based_repair","deepseek-v3", "taco_lite_0.125")
# count_passk_ambiguous("test_based_repair","deepseek-v3", "taco_lite_0.125")
# count_ambiguity("test_based_repair","deepseek-v3", "taco_lite_0.125")