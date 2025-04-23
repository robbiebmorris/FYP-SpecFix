import argparse
import ast
import sys
from os.path import dirname, abspath
import jsonlines
import concurrent.futures
import json
import re

from specfix.evaluator import SpecFixAccuracyEvaluator
from specfix.utils import (
    get_evalplus_inputs_outputs,
    construct_output_file,
    read_jsonl,
    unify_model_name
)
from specfix.tester import differential_tester, ground_truth_tester
import configparser

sys.set_int_max_str_digits(0)


def parse_problem(problem):
    requirement = problem['requirement']
    repaired_requirement = problem['repaired_requirement']
    examples = problem['input_output_examples']
    entry_point = problem['entry_point']
    task_id = problem['task_id']
    return requirement, repaired_requirement, entry_point, examples, task_id


def process_problem(i, problem, inputs, outputs, evaluator, n_programs, model_name, n_shot):
    requirement, repaired_requirement, entry_point, examples, task_id = parse_problem(problem)

    log_messages = []
    log_messages.append(f"Case {task_id}:\n{requirement}")
    
    repaired_clusters, repaired_passk, repaired_pass_rate, repaired_generated_programs, repaired_failed_inputs_outputs = None, None, None, None, None
    test_inputs = ast.literal_eval(problem['llm_generated_inputs'][model_name])
    programs = evaluator.parallel_generate_programs(repaired_requirement, entry_point, n_programs)
    
    #  get_clusters(self, requirement, programs, test_inputs, entry_point, examples=None):
    original_clusters = evaluator.get_clusters(requirement, programs, test_inputs, entry_point, examples)
    evaluator.get_test_consistency(original_clusters)
    log_messages.append(f"Case {task_id}:\nClusters entropy: {original_clusters.entropy}")
    
    original_passk, original_pass_rate, original_generated_programs, original_failed_inputs_outputs = evaluator.pass_k_and_pass_rate(
        requirement,
        inputs[i],
        outputs[i],
        entry_point,
        1, 10
    )
    
    print(repaired_requirement)
    if (repaired_requirement != None):
        log_messages.append(f"Case {task_id}:\nRepaired Requirement:\n{repaired_requirement}")
        programs = evaluator.parallel_generate_programs(repaired_requirement, entry_point, n_programs)
        repaired_clusters = evaluator.get_clusters(repaired_requirement, programs, test_inputs, entry_point, examples)
        evaluator.get_test_consistency(repaired_clusters)
        log_messages.append(f"Case {task_id}:\nClusters entropy: {repaired_clusters.entropy}")
        
        repaired_passk, repaired_pass_rate, repaired_generated_programs, repaired_failed_inputs_outputs = evaluator.pass_k_and_pass_rate(
            repaired_requirement,
            inputs[i],
            outputs[i],
            entry_point,
            1, 10
        )
    
    writer_dict = {
        "task_id": task_id,
        "requirement": requirement,
        "repaired_requirement": repaired_requirement,
        "original_clusters": original_clusters.serialize(),
        "original_passk": original_passk,
        "original_pass_rate": original_pass_rate,
        "original_failed_inputs_outputs": original_failed_inputs_outputs,
        "original_generated_programs": original_generated_programs,
        "repaired_clusters": None if repaired_clusters is None else repaired_clusters.serialize(),
        "repaired_passk": original_passk if repaired_passk is None else repaired_passk,
        "repaired_pass_rate": original_pass_rate if repaired_pass_rate is None else repaired_pass_rate,
        "repaired_failed_inputs_outputs": None if repaired_failed_inputs_outputs is None else str(repaired_failed_inputs_outputs),
        "repaired_generated_programs": None if repaired_generated_programs is None else repaired_generated_programs,
    }
    return i, writer_dict, repaired_passk, "\n".join(log_messages)
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True)
    parser.add_argument("-n", "--program_number", dest="number", type=int, default=20)
    parser.add_argument("-woe", "--without_example", dest="without_example", action='store_true')
    options = parser.parse_args()

    model_name = unify_model_name(options.model)
    dataset = "humaneval"
    wo_example = "_woe" if options.without_example else ""
    dataset_path = f"../../dataset/humaneval{wo_example}.jsonl"
    n_programs = options.number
    n_shot = options.n_shot
    
    specfix_accuracy_evaluator = SpecFixAccuracyEvaluator(
        differential_tester=differential_tester,
        ground_truth_tester=ground_truth_tester,
        model=model_name,
        temperature=0,
    )


    inputs, outputs = get_evalplus_inputs_outputs(dataset)
    
    output_file = construct_output_file(dirname(abspath(__file__)), model_name, f"{dataset}{wo_example}", f"results")
    
    problems = read_jsonl(dataset_path)    
    tasks = [(i, problem) for i, problem in enumerate(problems)]
    
    # Get the llm examples for gpt-4o from our we dataset
    if wo_example == "_woe" and model_name == "gpt-4o":
        with open("../../dataset/humaneval.jsonl") as f:
            we_problems = {
                entry['task_id']: entry
                for line in f
                for entry in [json.loads(line)]
            }

        for (i, problem) in tasks:
            wep = we_problems.get(problem['task_id'])
            problem['llm_generated_inputs']['gpt-4o'] = wep['llm_generated_inputs']['gpt-4o']
  
    # Load examples from file
    examples = {
        k: v for line in open("data/examples.jsonl")
        for k, v in json.loads(line).items()
    }

    repaired_tasks = []
    for (i, problem) in tasks:
        example = examples.get(problem['task_id'], '')
        if example == '':
            problem["repaired_requirement"] = None
            repaired_tasks.append((i, problem))
            continue  # skip if there's no example

        # Inject the example before closing triple quotes
        problem["repaired_requirement"] = re.sub(
            r'("""|\'\'\')\s*$',
            example + r'\1',
            problem["requirement"],
            flags=re.DOTALL
        )

        repaired_tasks.append((i, problem))

    # Replace old tasks with repaired ones (optional)
    tasks = repaired_tasks
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor, \
            jsonlines.open(output_file, mode='w', flush=True) as writer:

        results = executor.map(
            lambda args: process_problem(
                *args, inputs, outputs, specfix_accuracy_evaluator, n_programs, model_name, n_shot
            ),
            tasks
        )

        for i, writer_dict, repaired_passk, log_msg in results:
            print(log_msg)
            writer.write(writer_dict)


if __name__ == "__main__":
    main()
