import argparse
import time
import os
import sys

try:
   from data import load_data
   from node import Node
   from distance_matrix import calculate_distance_matrix
   from generate_start_node_indices import generate_start_node_indices
   from local_search import (
       MethodSpec,
       LSType,
       IntraType,
       StartType,
       run_local_search_batch,
   )
   from stats import calculate_stats
   from row import Row
   from results_to_csv import results_to_csv
   from plot import plot_solution
   from best_solution import find_best_solution
   from sanitizer import sanitize_path


except ImportError:
   print("Error: Could not import helper modules.")
   print("Ensure __init__.py exists in the directory if it's a package,")
   print("or adjust import paths if they are in the same directory.")
   print("Example: 'from data import load_data' instead of 'from .data import load_data'")
   sys.exit(1)




def main():
   """
   Main function to run the Evolutionary Computation algorithms.
   """


   # --- 1. Argument Parsing ---
   # Equivalent to Go's 'flag' package
   parser = argparse.ArgumentParser(
       description="Run Local Search algorithms for TSP."
   )


   # Define command-line arguments
   parser.add_argument(
       "-f", "--file",
       type=str,
       default="TSPA.csv",
       help="Input CSV file name (e.g., TSPA.csv or TSPB.csv)."
   )
   parser.add_argument(
       "-n", "--numSolutions",
       type=int,
       default=10,
       help="Number of solutions to generate for each method."
   )
   parser.add_argument(
       "-o", "--output",
       type=str,
       default="results.csv",
       help="Output CSV file name for results."
   )
   parser.add_argument(
       "-p", "--plot",
       type=str,
       default="plot",
       help="Base name for saving solution plots (e.g., 'plot' -> plot_method1.png)."
   )


   args = parser.parse_args()


   print(f"Running with options:")
   print(f"  Input file:   {args.file}")
   print(f"  Solutions/Method: {args.numSolutions}")
   print(f"  Results file: {args.output}")
   print(f"  Plot basename:  {args.plot}")
   print("-" * 30)


   # --- 2. Load Data ---
   nodes, header = load_data(args.file)
   if not nodes:
       print(f"Failed to load data from {args.file}. Exiting.")
       return


   print(f"Loaded {len(nodes)} nodes from {args.file}.")


   # --- 3. Pre-computation ---
   distance_matrix = calculate_distance_matrix(nodes)
   node_costs = [node.cost for node in nodes]
   n = len(nodes)
   start_node_indices = generate_start_node_indices(n)


   print(f"Calculated distance matrix and node costs for {n} nodes.")


   # --- 4. Define Methods ---
   # Define the list of methods to run
   methods = [
       # --- Random Start ---
       MethodSpec(LSType.LS_Steepest, IntraType.IntraSwap, StartType.StartRandom, "steepest-swap-random"),
       MethodSpec(LSType.LS_Steepest, IntraType.Intra2Opt, StartType.StartRandom, "steepest-2opt-random"),
       MethodSpec(LSType.LS_Greedy, IntraType.IntraSwap, StartType.StartRandom, "greedy-swap-random"),
       MethodSpec(LSType.LS_Greedy, IntraType.Intra2Opt, StartType.StartRandom, "greedy-2opt-random"),


       # --- Greedy Regret Start ---
       MethodSpec(LSType.LS_Steepest, IntraType.IntraSwap, StartType.StartGreedyRegret, "steepest-swap-regret"),
       MethodSpec(LSType.LS_Steepest, IntraType.Intra2Opt, StartType.StartGreedyRegret, "steepest-2opt-regret"),
       MethodSpec(LSType.LS_Greedy, IntraType.IntraSwap, StartType.StartGreedyRegret, "greedy-swap-regret"),
       MethodSpec(LSType.LS_Greedy, IntraType.Intra2Opt, StartType.StartGreedyRegret, "greedy-2opt-regret"),
   ]


   all_results: list[Row] = []


   # --- 5. Run All Methods ---
   for method in methods:
       print(f"\nRunning method: {method.Name}...")


       start_time = time.perf_counter()


       solutions = run_local_search_batch(
           distance_matrix,
           node_costs,
           start_node_indices,
           method,
           args.numSolutions
       )


       end_time = time.perf_counter()
       elapsed_time = end_time - start_time


       if not solutions:
           print(f"Method {method.Name} returned no solutions.")
           continue


       print(f"  Finished in {elapsed_time:.4f} seconds.")


       # --- 6. Calculate Stats ---
       min_obj, avg_obj, max_obj, best_path = calculate_stats(solutions)


       print(f"  Stats: Min={min_obj}, Avg={avg_obj:.2f}, Max={max_obj}")
       print(f"  Best tour: {sanitize_path(best_path)}")


       # Store results
       all_results.append(
           Row(
               MethodName=method.Name,
               Min=min_obj,
               Avg=avg_obj,
               Max=max_obj,
               Time=elapsed_time,
               BestPath=best_path
           )
       )


       # --- 7. Save Plot for this method ---
       if args.plot:
           best_sol = find_best_solution(solutions)


           # Get the base name of the input file (e.g., "TSPA" from "TSPA.csv")
           file_basename = os.path.splitext(os.path.basename(args.file))[0]


           # Create a unique plot filename
           plot_filename = f"{args.plot}_{file_basename}_{method.Name}.png"


           plot_solution(nodes, best_sol, plot_filename)


   # --- 8. Save All Results to CSV ---
   if all_results:
       results_to_csv(all_results, args.output)
       print(f"\nSuccessfully saved all results to {args.output}")
   else:
       print("\nNo results were generated to save.")




# Standard Python entry point
if __name__ == "__main__":
   main()
