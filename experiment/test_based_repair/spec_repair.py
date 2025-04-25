import argparse
import ast
import sys
from os.path import dirname, abspath
import jsonlines
import concurrent.futures

from specfix.evaluator import SpecFixAccuracyEvaluator
from specfix.utils import (
    get_evalplus_inputs_outputs,
    construct_output_file,
    read_jsonl,
    get_taco_lite_inputs_outputs,
    unify_model_name
)
from specfix.tester import differential_tester, ground_truth_tester

sys.set_int_max_str_digits(0)


def process_problem(i, problem, inputs, outputs, evaluator, n_programs, model_name):
    result, original_clusters = evaluator.specfix_detect(problem, n_programs=n_programs)
    if result:
        repaired_requirement, repaired_clusters = evaluator.specfix_repair(original_clusters,
                                                                            n_programs)
        problem_woe = problem
        problem_woe["requirement"] = evaluator.remove_example(repaired_requirement)
        print(problem_woe)
        
        _, repaired_requirement_woe_clusters = evaluator.specfix_detect(problem_woe,
                                                                        n_programs=n_programs)
        # original_passk, original_generated_programs, original_failed_inputs_outputs = evaluator.pass_k_sample(
        #     problem["requirement"], inputs[i], outputs[i], problem["entry_point"], 1, 10
        # )
        repaired_passk, repaired_generated_programs, repaired_failed_inputs_outputs = evaluator.pass_k_sample(
            repaired_requirement, inputs[i], outputs[i], problem["entry_point"], 1, 10
        )
        repaired_woe_passk, repaired_woe_generated_programs, repaired_woe_failed_inputs_outputs = evaluator.pass_k_sample(
            problem_woe["requirement"], inputs[i], outputs[i], problem["entry_point"], 1, 10
        )
        result = {
            "task_id": problem["task_id"],
            "requirement": problem["requirement"],
            "repaired_requirement": repaired_requirement,
            "repaired_requirement_woe": problem_woe["requirement"],
            "original_clusters": original_clusters.serialize(),
            "repaired_clusters": repaired_clusters.serialize(),
            "repaired_woe_clusters": repaired_requirement_woe_clusters.serialize(),
            "original_passk": problem["original_passk"],
            "original_generated_programs": problem["original_generated_programs"],
            "original_failed_inputs_outputs": problem["original_failed_inputs_outputs"],
            "repaired_passk": repaired_passk,
            "repaired_generated_programs": repaired_generated_programs,
            "repaired_failed_inputs_outputs": str(repaired_failed_inputs_outputs),
            "repaired_woe_passk": repaired_woe_passk,
            "repaired_woe_generated_programs": repaired_woe_generated_programs,
            "repaired_woe_failed_inputs_outputs": str(repaired_woe_failed_inputs_outputs),
        }
    else:
        result = {
            "task_id": problem["task_id"],
            "requirement": problem["requirement"],
            "repaired_requirement": None,
            "repaired_woe_requirement": None,
            "original_clusters": original_clusters.serialize(),
            "repaired_clusters": None,
            "repaired_woe_clusters": None,
            "original_passk": problem["original_passk"],
            "original_generated_programs": problem["original_generated_programs"],
            "original_failed_inputs_outputs": problem["original_failed_inputs_outputs"],
            "repaired_passk": problem["original_passk"],
            "repaired_generated_programs": None,
            "repaired_failed_inputs_outputs": None,
            "repaired_woe_passk": None,
            "repaired_woe_generated_programs": None,
            "repaired_woe_failed_inputs_outputs": None,
        }
    return i, result, f"{i} Complete."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Dataset: humaneval, mbpp")
    parser.add_argument("-n", "--program_number", type=int, default=20)
    parser.add_argument("-m", "--model", required=True)
    options = parser.parse_args()

    evaluator = SpecFixAccuracyEvaluator(
        differential_tester=differential_tester,
        ground_truth_tester=ground_truth_tester,
        model=options.model,
    )

    n_programs = options.program_number
    model_name = unify_model_name(options.model)
    dataset_path = f"../../dataset/{options.dataset}.jsonl"

    inputs, outputs = (
        get_evalplus_inputs_outputs(options.dataset)
        if options.dataset in ["humaneval", "mbpp"]
        else get_taco_lite_inputs_outputs()
    )

    
    original_problems = read_jsonl(f"../original_passk/results/{model_name}/{options.dataset}.jsonl")
    problems = read_jsonl(dataset_path)

    for problem, original_problem in zip(problems, original_problems):
        problem["original_passk"] = original_problem["original_passk"]
        problem["original_generated_programs"] = original_problem["generated_programs"]
        problem["original_failed_inputs_outputs"] = original_problem["failed_inputs_outputs"]

    tasks = [(i, problem) for i, problem in enumerate(problems)]

    output_file = output_file = construct_output_file(
        dirname(abspath(__file__)),
        model_name,
        options.dataset,
        "results"
    )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor, \
            jsonlines.open(output_file, mode='w', flush=True) as writer:

        results = executor.map(
            lambda args: process_problem(
                *args, inputs, outputs, evaluator, n_programs, model_name
            ),
            tasks
        )

        for i, writer_dict, log_msg in results:
            print(log_msg)
            writer.write(writer_dict)


if __name__ == "__main__":
    main()
