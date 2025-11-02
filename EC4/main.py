import logging
import random
import time
import sys
from typing import List
import algorithms
import data
import utils
import visualisation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
)


def process_instance(instance_name: str, nodes: List[data.Node]):
    """
    Runs the local search experiment for a given instance and reports results.
    """
    logging.info(f"Processing instance {instance_name} with {len(nodes)} nodes")
    print(f"Instance {instance_name} Statistics:")

    # Assumes data.calculate_distance_matrix exists
    D = data.calculate_distance_matrix(nodes)

    # Assumes Node is a dataclass/object with a .cost attribute
    costs = [node.cost for node in nodes]

    num_solutions = 200

    # Assumes algorithms.MethodSpec is a dataclass or class
    methods = [
        algorithms.MethodSpec(
            name="Baseline_Steepest_2opt_Random",
            use_cand=False,
            cand_k=0,
        ),
        algorithms.MethodSpec(
            name="Candidates_Steepest_2opt_Random_K10",
            use_cand=True,
            cand_k=10,
        ),
        algorithms.MethodSpec(
            name="Candidates_Steepest_2opt_Random_K20",
            use_cand=True,
            cand_k=20,
        ),
        algorithms.MethodSpec(
            name="Candidates_Steepest_2opt_Random_K30",
            use_cand=True,
            cand_k=30,
        ),
    ]

    rows: List[utils.Row] = []

    for m in methods:
        logging.info(f"Starting method: {m.name} for instance {instance_name}")
        # time.perf_counter() is more precise for measuring execution time
        start_time = time.perf_counter()

        # Assumes algorithms.run_local_search_batch exists
        solutions = algorithms.run_local_search_batch(D, costs, m, num_solutions)

        batch_time_sec = time.perf_counter() - start_time

        if not solutions:
            logging.warning(f"No solutions found for method {m.name}")
            continue

        # Assumes utils.calculate_statistics exists
        min_val, max_val, avg_val = utils.calculate_statistics(solutions)

        # Calculate avg time in milliseconds
        avg_time_ms = (batch_time_sec * 1000) / num_solutions

        # Assumes algorithms.find_best_solution exists
        best = algorithms.find_best_solution(solutions)

        # Assumes utils.Row is a dataclass or class
        rows.append(utils.Row(
            name=m.name,
            avg_v=avg_val,
            min_v=min_val,
            max_v=max_val,
            avg_t_ms=avg_time_ms,
            best_path=best.path,
            best_value=best.objective,
        ))

        logging.info(f"Completed method {m.name}: best value {best.objective}, avg time {avg_time_ms:.2f} ms")

        # Plotting
        try:
            title = f"Best {m.name} Solution for Instance {instance_name}"
            # Assumes utils.sanitize_file_name exists
            file_name = utils.sanitize_file_name(f"Best_{m.name}_Solution_{instance_name}")
            # Assumes visualisation.plot_solution exists
            visualisation.plot_solution(nodes, best.path, title, file_name, 0, 4000, 0, 2000)
        except Exception as e:
            logging.error(f"Plot error for {instance_name}/{m.name}: {e}")

    # 5) Results — Console (using f-strings for formatting)
    print("\nObjective value: av (min, max)")
    for r in rows:
        # {r.name:<34} pads the string to 34 characters (left-aligned)
        print(f"{r.name:<34}  {r.avg_v:.2f} ({r.min_v}, {r.max_v})")
        print(f"Best path: {r.best_path}")
    print()

    print("Average time per run [ms]:")
    for r in rows:
        print(f"{r.name:<34}  {r.avg_t_ms:.4f}")

    # 6) CSV
    try:
        # Assumes utils.write_results_csv exists
        utils.write_results_csv(instance_name, rows)
        logging.info(f"CSV results saved for instance {instance_name}")
    except Exception as e:
        logging.error(f"CSV write error for instance {instance_name}: {e}")


def main():
    """
    Main entry point for the program.
    """
    random.seed(time.time())
    logging.info("Starting evolutionary computation local search program")

    try:
        # Assumes data.read_nodes exists
        nodes_a = data.read_nodes("TSPA.csv")
    except Exception as e:
        logging.fatal(f"Error reading TSPA.csv: {e}")
        sys.exit(1)

    try:
        nodes_b = data.read_nodes("TSPB.csv")
    except Exception as e:
        logging.fatal(f"Error reading TSPB.csv: {e}")
        sys.exit(1)

    process_instance("A", nodes_a)
    print()
    process_instance("B", nodes_b)

    logging.info("Program execution completed")


if __name__ == "__main__":
    main()
