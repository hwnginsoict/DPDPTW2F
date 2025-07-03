from utils_new import crossover_operator, mutation_operator, calculate_fitness, create_individual_pickup
from utils import create_individual_pickup_lerk, crossover_operator_lerk, mutation_operator_lerk, calculate_fitness_lerk
from LERK_utils import crossover_LERK, mutation_LERK, calculate_fitness_LERK, create_individual_LERK
from graph.graph import Graph
import time
from moo_algorithm.pfg_moea import run_pfgmoea
import json
import numpy as np

def main(number = 8, type = "LC2", index = 1, seed = 0, num = 100, max_gen = 100):
    # 1) Prepare data and create an initial population of individuals

    # filepath = f'.\\data\\dpdptw\\{number}00\\{type}_{number}_{index}.csv'
    filepath = f'./data/dpdptw/{number}00/{type}_{number}_{index}.csv'

    graph = Graph(filepath)
    indi_list_lerk = [create_individual_pickup_lerk(graph) for _ in range(num)]
    indi_list = [create_individual_pickup(graph) for _ in range(num)]
    indi_list_original = [create_individual_LERK(graph) for _ in range(num)]

    np.random.seed(seed)

    start_time_original = time.time()
    original_results = run_pfgmoea(4, graph, indi_list_original, num, max_gen, 5, 0.01, crossover_LERK, mutation_LERK, 0.9, 0.1, calculate_fitness_LERK)
    original_time = time.time() - start_time_original
    original_results["time"] = original_time

    start_time_proposed = time.time()
    proposed_results = run_pfgmoea(4, graph, indi_list, num, max_gen, 5, 0.01, crossover_operator, mutation_operator, 0.9, 0.1, calculate_fitness)
    proposed_time = time.time() - start_time_proposed
    proposed_results["time"] = proposed_time

    start_time_pfg = time.time()
    pfg_results = run_pfgmoea(4, graph, indi_list_lerk, num, max_gen, 5, 0.01, crossover_operator_lerk, mutation_operator_lerk, 0.9, 0.1, calculate_fitness_lerk)
    pfg_time = time.time() - start_time_pfg
    pfg_results["time"] = pfg_time
    
    # 4) Combine and store both results in the same JSON
    final_results = {
        "Original": original_results,
        "Proposed": proposed_results,
        "PFG-EMOA": pfg_results,
    }

    name_store = f"Result/{type}_{number}_{index}_{seed}.json"
    with open(name_store, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results stored in {name_store}")


import argparse

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with different parameters.")
    
    # --types can accept one or more experiment types: e.g. --types LRC1 LRC2
    parser.add_argument(
        "--type", 
        nargs="+", 
        default=["LC1", "LC2", "LR1", "LR2", "LRC1", "LRC2"],
        help="Experiment type(s). Default includes all possible types."
    )
    
    # --max_gen allows overriding the default of 100 from main; we set default to 150 here
    parser.add_argument(
        "--maxgen", 
        type=int, 
        default=150, 
        help="Maximum generation value. Default: 150."
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed. Default: 0."
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Loop to run main with the chosen arguments
    for num in [2, 4, 8]:
        for t in args.type:
            for i in range(1, 11):
                # We pass args.max_gen instead of the hard-coded 150
                main(number=num, type=t, index=i, seed=args.seed, num=100, max_gen=args.maxgen)
                print(f"Done {num}_{t}_{i}")