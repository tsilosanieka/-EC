import time
import logging
import os
import random
import multiprocessing
from functools import partial
import data
import algorithms
import utils
import visualisation
from algorithms import PerturbationType

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')


def run_msls_worker(args):
    """
    Worker function to run a single instance of MSLS.
    Args tuple: (D, costs, num_starts)
    """
    D, costs, num_starts = args
    random.seed(os.urandom(4))
    return algorithms.run_msls(D, costs, num_starts)


def run_ils_worker(args):
    """
    Worker function to run a single instance of ILS.
    Args tuple: (D, costs, time_limit, perturb_type)
    """
    D, costs, time_limit, perturb_type = args
    random.seed(os.urandom(4))
    return algorithms.run_ils(D, costs, time_limit, perturb_type)


# ==========================================
# MAIN LOGIC
# ==========================================

def process_instance(instance_name: str, nodes: list[data.Node]):
    logging.info(f"Processing instance {instance_name} with {len(nodes)} nodes")
    print(f"\n=== Instance {instance_name} Statistics ===")

    # Prepare Data
    dist_matrix = data.calculate_distance_matrix(nodes)
    costs = [node.cost for node in nodes]

    NUM_MSLS_RUNS = 20 
    NUM_MSLS_STARTS = 200 
    NUM_ILS_RUNS = 20    

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Parallelizing with {num_cores} cores...")

    rows = []

    logging.info(f"Starting MSLS for instance {instance_name}")
    start_time = time.time()

    msls_args = [(dist_matrix, costs, NUM_MSLS_STARTS) for _ in range(NUM_MSLS_RUNS)]

    with multiprocessing.Pool(processes=num_cores) as pool:
        msls_results = pool.map(run_msls_worker, msls_args)

    total_msls_time = time.time() - start_time
    avg_msls_time_seconds = total_msls_time / NUM_MSLS_RUNS

    msls_solutions = [r.best_solution for r in msls_results]
    msls_min, msls_max, msls_avg = utils.calculate_statistics(msls_solutions)

    avg_msls_time_ms = avg_msls_time_seconds * 1000
    best_msls = algorithms.find_best_solution(msls_solutions)

    rows.append(utils.Row(
        name="MSLS",
        avg_v=msls_avg,
        min_v=msls_min,
        max_v=msls_max,
        avg_tms=avg_msls_time_ms,
        best_path=best_msls.path,
        best_value=best_msls.objective
    ))

    logging.info(f"Completed MSLS: best value {best_msls.objective}, avg time {avg_msls_time_ms:.2f} ms")

    logging.info(f"Starting ILS for instance {instance_name} with time limit {avg_msls_time_seconds:.4f}s")

    perturb_type = PerturbationType.RANDOM_4_OPT

    ils_args = [(dist_matrix, costs, avg_msls_time_seconds, perturb_type) for _ in range(NUM_ILS_RUNS)]

    with multiprocessing.Pool(processes=num_cores) as pool:
        ils_results = pool.map(run_ils_worker, ils_args)

    total_ils_iterations = sum(r.num_ls_iterations for r in ils_results)
    ils_solutions = [r.best_solution for r in ils_results]
    ils_min, ils_max, ils_avg = utils.calculate_statistics(ils_solutions)

    avg_ils_time_ms = avg_msls_time_seconds * 1000
    avg_ls_iterations = total_ils_iterations / NUM_ILS_RUNS
    best_ils = algorithms.find_best_solution(ils_solutions)

    rows.append(utils.Row(
        name="ILS",
        avg_v=ils_avg,
        min_v=ils_min,
        max_v=ils_max,
        avg_tms=avg_ils_time_ms,
        best_path=best_ils.path,
        best_value=best_ils.objective
    ))

    logging.info(f"Completed ILS: best value {best_ils.objective}, avg LS iterations {avg_ls_iterations:.1f}")

    print("\n--- Results Summary ---")
    print("Objective value: av (min, max)")
    for r in rows:
        print(f"{r.name:<34}  {r.avg_v:.2f} ({r.min_v}, {r.max_v})")

    print("\nAverage time per run [ms]:")
    for r in rows:
        print(f"{r.name:<34}  {r.avg_tms:.4f}")
    print(f"ILS - Average LS iterations per run: {avg_ls_iterations:.1f}")

    print("\n--- Best Solution Indices ---")
    for r in rows:
        path_str = str(r.best_path) if r.best_path else "[]"
        print(f"{r.name} Best Path ({len(r.best_path)} nodes): {path_str}")

    try:
        sanitized_name = utils.sanitize_file_name(instance_name)
        utils.write_results_csv(sanitized_name, rows)
    except Exception as e:
        logging.error(f"CSV write error for instance {instance_name}: {e}")

    X_MAX, Y_MAX = 4000, 2000

    title_msls = f"Best MSLS Solution for Instance {instance_name}"
    file_msls = utils.sanitize_file_name(f"Best_MSLS_Solution_{instance_name}")
    visualisation.plot_solution(
        nodes, best_msls.path, title_msls, file_msls,
        0, X_MAX, 0, Y_MAX
    )

    # Plot ILS Best
    title_ils = f"Best ILS Solution for Instance {instance_name}"
    file_ils = utils.sanitize_file_name(f"Best_ILS_Solution_{instance_name}")
    visualisation.plot_solution(
        nodes, best_ils.path, title_ils, file_ils,
        0, X_MAX, 0, Y_MAX
    )


def main():
    logging.info("Starting MSLS vs ILS local search experiments (Multiprocessing Enabled)")

  instance_path_a = "TSPA.csv"
    instance_path_b = "TSPB.csv"

    try:
        if os.path.exists(instance_path_a):
            nodes_a = data.read_nodes(instance_path_a)
            process_instance("A", nodes_a)
        else:
            logging.warning(f"File {instance_path_a} not found in current directory. Skipping Instance A.")

        print("-" * 50)

        if os.path.exists(instance_path_b):
            nodes_b = data.read_nodes(instance_path_b)
            process_instance("B", nodes_b)
        else:
            logging.warning(f"File {instance_path_b} not found in current directory. Skipping Instance B.")

    except Exception as e:
        logging.fatal(f"An error occurred during execution: {e}")

    logging.info("Program execution completed")


if __name__ == "__main__":
    main()
