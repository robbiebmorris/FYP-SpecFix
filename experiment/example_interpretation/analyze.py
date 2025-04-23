from specfix.utils import read_jsonl, count_passk, count_passk_ambiguous, count_entropy, count_entropy_ambiguous
import sys
sys.set_int_max_str_digits(0)

def print_r(result):
    print("original requirement:", result["requirement"])
    print("repaired requirement:", result["repaired_requirement"])
    print("original passk", result["original_passk"])
    print("repaired passk", result["repaired_passk"])
    print("original entropy:", result["original_clusters"]["entropy"])
    print("original t_consistency:", result["original_clusters"]["weighted_test_consistency"])
    print("original programs:")
    i = 1
    for c in result["original_clusters"]["cluster_list"]:
        print(f"Cluster {i} (size of cluster: {len(c["programs_str"])}, probability: {c["probability"]}, consistency: {c["test_consistency"]}")
        i = i + 1
        print("\n", c["programs_str"][0], "\n")
        print(c["failed_input_output_examples"])
    if (result["repaired_clusters"]):
        print("repaired entropy:", result["repaired_clusters"]["entropy"])
        print("repaired t_consistency:", result["repaired_clusters"]["weighted_test_consistency"])
        print("repaired programs:")
        i = 1
        for c in result["repaired_clusters"]["cluster_list"]:
            print(f"Cluster {i} (size of cluster: {len(c["programs_str"])}, probability: {c["probability"]}, consistency: {c["test_consistency"]}")
            i = i + 1
            print("\n", c["programs_str"][0], "\n")
            print(c["failed_input_output_examples"])
    print("original_generated_programs", result["original_generated_programs"])
    print("repaired_generated_programs", result["repaired_generated_programs"])
    print("original_input_output", result["original_failed_inputs_outputs"])
    print("repaired_input_output", result["repaired_failed_inputs_outputs"])
    print("=======================")
        
def print_problem(label, model, dataset, task_id):
    results = read_jsonl(f"{label}/{model}/{dataset}.jsonl")
    for result in results:
        if result["task_id"] == task_id:
            print_r(result)
            
def find_passk_problems(label, model, dataset):
    results = read_jsonl(f"{label}/{model}/{dataset}.jsonl")
    we_results = read_jsonl(f"{label}/{model}/humaneval_0.jsonl")
    woe_result_list = []
    we_result_list = []
    io_interp_list = []
    total = 0
    count = 0
    for result in results:
        print(result.keys())
        if result["original_passk"] != 1.0:
            total += 1
            if (result["repaired_requirement"] == result["requirement"]):
                count += 1
            else:
                for r in we_results:
                    if r["task_id"] == result["task_id"]:
                        we_result_list.append(r["original_passk"])
                        woe_result_list.append(result["original_passk"])
                        io_interp_list.append(result["repaired_passk"])
            

            print(result["requirement"])
            print(result["repaired_requirement"])
    print(f"{dataset} we pass@1: {sum(we_result_list) / len(we_result_list)}, woe pass@1: {sum(woe_result_list) / len(woe_result_list)}")
    print(f"{dataset} io-interp pass@1: {sum(io_interp_list) / len(io_interp_list)}")
    print("unrepaired by humans count: ", count)
    print("total unsolved: ", total)


def analyze_1(label, model, dataset):
    results = read_jsonl(f"{label}/{model}/{dataset}.jsonl")
    for result in results:
        if result["original_passk"] > result["repaired_passk"]:
            # print_r(result)
            print("===================")
            print(f"Original: (pass@1: {result['original_passk']})", result["requirement"])
            print(f"IE: (pass@1: {result['repaired_passk']})", result["repaired_requirement"])
            print("===================\n\n")


def analyze(label, model):
    woe_results = read_jsonl(f"{label}/{model}/humaneval_woe.jsonl")
    we_results = read_jsonl(f"{label}/{model}/humaneval.jsonl")
    total = 0
    count = 0
    for woe_r in woe_results:
        for we_r in we_results:
            if woe_r["repaired_passk"] < we_r["original_passk"] and woe_r["task_id"] == we_r["task_id"]:
                print("===================")
                print("task_id", we_r["task_id"])
                print(f"WOE: (pass@1: {woe_r['original_passk']})", woe_r["requirement"])
                print(f"WE: (pass@1: {we_r['original_passk']})", we_r["requirement"])
                print(f"WOE_IO: (pass@1: {woe_r['repaired_passk']})", woe_r["repaired_requirement"])
                print(f"WE_IO: (pass@1: {we_r['repaired_passk']})", we_r["repaired_requirement"])
                total += 1
                print("===================\n\n")

    print("Total failures:", total)
    print("Missed repairs count:", count)
    # results = read_jsonl("original_humaneval")
    # for r in results:
    #     if original_cluster[""]


# find_passk_problems("clarify_gpt_zero_shot","deepseek-v3", "humaneval_woe_0")
            
# analyze_1("results", "qwen2.5-coder-32b-instruct", "humaneval_woe")
            
analyze("results", "qwen2.5-coder-32b-instruct")
# print_problem("results", "qwen2.5-coder-32b-instruct", "humaneval_woe", "HumanEval/154") # cycpattern_check
# print_problem("results", "qwen2.5-coder-32b-instruct", "humaneval", "HumanEval/154")


# count_passk("results","qwen2.5-coder-32b-instruct", "humaneval")
# count_passk_ambiguous("results","qwen2.5-coder-32b-instruct", "humaneval")
# count_passk("results","qwen2.5-coder-32b-instruct", "humaneval_woe")
# count_passk_ambiguous("results","qwen2.5-coder-32b-instruct", "humaneval_woe")

# count_passk("results","gpt-4o", "humaneval")
# count_passk_ambiguous("results","gpt-4o", "humaneval")
# count_passk("results","gpt-4o", "humaneval_woe")
# count_passk_ambiguous("results","gpt-4o", "humaneval_woe")

# count_passk("results","deepseek-v3", "humaneval")
# count_passk_ambiguous("results","deepseek-v3", "humaneval")
# count_passk("results","deepseek-v3", "humaneval_woe")
# count_passk_ambiguous("results","deepseek-v3", "humaneval_woe")
