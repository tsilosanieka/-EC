import logging
import random
import time
import sys
from typing import List

# --- Imports ---
# These now point to the Python files translated from your Go code
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
    Runs the full experimental pipeline for a single instance.
    (Translated from Go's processInstance)
    """
    logging.info(f"Processing instance {instance_name} with {len(nodes)} nodes")
    print(f"Instance {instance_name} Statistics:")

    D = data.calculate_distance_matrix(nodes)
    costs = [node.cost for node in nodes]
    num_solutions = 200

    # This list of methods is translated directly from your main.go
    methods = [
        # BASELINE (no candidate moves)
        algorithms.MethodSpec(
            Name="Baseline_Steepest_2opt_Random",
            UseCand=False,
            CandK=0,
            UseLM=False,
        ),
        # LM-based steepest local search (no candidate moves)
        algorithms.MethodSpec(
            Name="LM_Steepest_2opt_Random",
            UseCand=False,
            CandK=0,
            UseLM=True,
        ),
        # CANDIDATE MOVES - K = 10
        algorithms.MethodSpec(
            Name="Candidates_Steepest_2opt_Random_K10",
            UseCand=True,
            CandK=10,
            UseLM=False,
        ),
        # CANDIDATE MOVES - K = 20
        algorithms.MethodSpec(
            Name="Candidates_Steepest_2opt_Random_K20",
            UseCand=True,
            CandK=20,
            UseLM=False,
        ),
        # CANDIDATE MOVES - K = 30
        algorithms.MethodSpec(
            Name="Candidates_Steepest_2opt_Random_K30",
            UseCand=True,
            CandK=30,
            UseLM=False,
        ),
    ]

    rows: List[utils.Row] = []

    for m in methods:
        logging.info(f"Starting method: {m.Name} for instance {instance_name}")

        # Run the batch, which now returns durations
        solutions, durations_ms = algorithms.run_local_search_batch(D, costs, m, num_solutions)

        if not solutions:
            logging.warning(f"No solutions found for method {m.Name}")
            continue

        # Calculate stats (min, max, avg)
        # Use the NEW utils.calculate_statistics
        min_val, max_val, avg_val = utils.calculate_statistics(solutions)

        # Calculate average time from the returned durations
        avg_time_ms = 0.0
        min_time_ms = 0.0
        max_time_ms = 0.0

        if durations_ms:
            avg_time_ms = sum(durations_ms) / len(durations_ms)
            min_time_ms = min(durations_ms)
            max_time_ms = max(durations_ms)

        best = algorithms.find_best_solution(solutions)

        # Use the NEW utils.Row (with Go-style capital letters)
        rows.append(utils.Row(
            Name=m.Name,
            AvgV=avg_val,
            MinV=min_val,
            MaxV=max_val,
            AvgTms=avg_time_ms,
            BestPath=best.Path,
            BestValue=best.Objective,
        ))

        # Log timings (matches Go logic)
        if m.UseLM and durations_ms:
            logging.info(
                f"Completed method {m.Name}: best value {best.Objective}, avg time {avg_time_ms:.2f} ms (min: {min_time_ms:.2f}, max: {max_time_ms:.2f})")
        else:
            logging.info(f"Completed method {m.Name}: best value {best.Objective}, avg time {avg_time_ms:.2f} ms")

        # Plotting
        try:
            title = f"Best {m.Name} Solution for Instance {instance_name} (Objective: {best.Objective})"
            # Use the NEW utils.sanitize_file_name
            file_name = utils.sanitize_file_name(f"Best_{m.Name}_Solution_{instance_name}")
            visualisation.plot_solution(nodes, best.Path, title, file_name, 0, 4000, 0, 2000)
        except Exception as e:
            logging.error(f"Plot error for {instance_name}/{m.Name}: {e}")

    # 5) Results — Console (matches Go formatting)
    print("\nObjective value: av (min, max)")
    for r in rows:
        # Use Go-style field names
        print(f"{r.Name:<34}  {r.AvgV:.2f} ({r.MinV}, {r.MaxV})")
        # Format path with spaces, not commas (using the new utils)
        path_str = utils._ints_to_dash_string(r.BestPath)
        print(f"Best path: {path_str}")
    print()

    print("Average time per run [ms]:")
    for r in rows:
        print(f"{r.Name:<34}  {r.AvgTms:.4f}")

    # 6) CSV
    try:
        # Use the NEW utils.write_results_csv
        utils.write_results_csv(instance_name, rows)
    except Exception as e:
        logging.error(f"CSV write error for instance {instance_name}: {e}")


def main():
    """
    Main entry point for the program.
    (Translated from Go's main)
    """
    random.seed(time.time())
    logging.info("Starting evolutionary computation local search program (Go-logic version)")

    # NOTE: You need to have TSPA.csv and TSPB.csv in the execution directory
    # or update the path here. Your Go code used "./instances/TSPA.csv"
    try:
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