import math
import random
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_instance(filename):
   """Loads node data from a file, splitting by semicolon."""
   coords = []
   costs = []
   with open(filename, 'r') as f:
       for line in f:
           parts = line.strip().split(';')
           if len(parts) == 3:
               coords.append((int(parts[0]), int(parts[1])))
               costs.append(int(parts[2]))
   return np.array(coords), np.array(costs)




def calculate_final_objective(tour, distance_matrix, node_costs):
   """Calculates the total objective: tour length + selected node costs."""
   selected_nodes = tour[:-1]
   tour_length = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
   total_node_cost = sum(node_costs[node] for node in selected_nodes)
   return tour_length + total_node_cost




def find_best_insertions(tour, city, distance_matrix, node_costs, k=2):
   """
   Efficiently finds the k-best insertion costs and positions for a city
   without sorting the entire list. Runs in O(K) time.
   """
   best_costs = [{'cost': float('inf'), 'pos': -1} for _ in range(k)]
   for i in range(1, len(tour)):
       u, v = tour[i - 1], tour[i]
       delta_length = distance_matrix[u][city] + distance_matrix[city][v] - distance_matrix[u][v]
       current_cost = delta_length + node_costs[city]
       if current_cost < best_costs[-1]['cost']:
           best_costs[-1] = {'cost': current_cost, 'pos': i}
           best_costs.sort(key=lambda x: x['cost'])
   return best_costs




def regretful_insert(distance_matrix, node_costs, start_node, num_to_select, k=2, method='2-regret', w1=0.5, w2=0.5):
   """Constructs a tour by iteratively inserting nodes based on a specified heuristic."""
   best_partner = -1
   min_initial_obj = float('inf')
   for partner in range(len(distance_matrix)):
       if partner == start_node: continue
       obj = 2 * distance_matrix[start_node][partner] + node_costs[start_node] + node_costs[partner]
       if obj < min_initial_obj:
           min_initial_obj = obj
           best_partner = partner
   tour = [start_node, best_partner, start_node]
   unvisited_cities = set(range(len(distance_matrix))) - {start_node, best_partner}
   while len(tour) - 1 < num_to_select:
       if not unvisited_cities: break
       candidates = []
       for city in unvisited_cities:
           best_insert_data = find_best_insertions(tour, city, distance_matrix, node_costs, k)
           best_change = best_insert_data[0]['cost']
           best_i = best_insert_data[0]['pos']
           if len(best_insert_data) >= k and best_insert_data[k - 1]['pos'] != -1:
               second_best_change = best_insert_data[k - 1]['cost']
               regret = second_best_change - best_change
           else:
               regret = -float('inf')
           if method == '2-regret':
               criterion = regret
           elif method == 'weighted-sum':
               criterion = w1 * regret - w2 * best_change
           else:
               raise ValueError(f"Unknown method: {method}")
           candidates.append((criterion, city, best_i))
       _, best_city, best_i = max(candidates, key=lambda x: x[0])
       tour = tour[:best_i] + [best_city] + tour[best_i:]
       unvisited_cities.remove(best_city)
   final_objective = calculate_final_objective(tour, distance_matrix, node_costs)
   return {"tour": tour, "objective": final_objective}




def solve_optimized(distance_matrix, node_costs, num_to_select, k, method, w1=0.5, w2=0.5, num_starts=10):
   """Runs the heuristic from a limited, random sample of starting nodes for speed."""
   num_nodes = len(distance_matrix)
   if num_starts >= num_nodes:
       start_nodes = range(num_nodes)
   else:
       start_nodes = random.sample(range(num_nodes), num_starts)
   best_result = {"objective": float('inf')}
   for start_node in start_nodes:
       result = regretful_insert(distance_matrix, node_costs, start_node, num_to_select, k, method, w1, w2)
       if result["objective"] < best_result["objective"]:
           best_result = result
   return best_result




def plot_solution(result, cities, node_costs, instance_name, method_name):
   """Generates an advanced plot of the solution, showing node costs."""
   plt.figure(figsize=(10, 10))
   all_coords = np.array(cities)
   selected_nodes = result['tour'][:-1]
   selected_costs = node_costs[selected_nodes]
   unselected_nodes = list(set(range(len(cities))) - set(selected_nodes))
   if unselected_nodes:
       plt.scatter(all_coords[unselected_nodes, 0], all_coords[unselected_nodes, 1],
                   c='gray', s=20, alpha=0.5, label='Unused Nodes')
   scatter = plt.scatter(all_coords[selected_nodes, 0], all_coords[selected_nodes, 1],
                         s=selected_costs * 0.5, c=selected_costs,
                         cmap='viridis', label='Selected Nodes', zorder=2)
   cbar = plt.colorbar(scatter)
   cbar.set_label('Node Cost')
   tour_coords = np.array([cities[i] for i in result["tour"]])
   plt.plot(tour_coords[:, 0], tour_coords[:, 1], c="k", linewidth=1.5, zorder=1)
   plt.title(f"Best Solution for '{instance_name}'\nMethod: {method_name} (Objective = {result['objective']:.0f})")
   plt.legend()
   plt.axis("off")
   plt.tight_layout()
   plt.show()



# --- Main Computational Experiment Block ---
if __name__ == '__main__':
   random.seed(123)   #We can comment this part to get truly random numbers for each run!
   plt.style.use("fivethirtyeight")


   instance_files = ["tspa.csv", "tspb.csv"]
   experiment_results = []


   # --- CHANGE 1: Define number of repetitions for statistics ---
   NUM_REPETITIONS = 10
   NUM_RANDOM_STARTS_PER_RUN = 10  # Each repetition will try 10 random starts


   for filename in instance_files:
       print(f"\n--- Processing Instance: {filename} ---")
       try:
           cities, node_costs = load_instance(filename)
       except FileNotFoundError:
           print(f"Error: The file '{filename}' was not found.")
           continue


       num_to_select = math.ceil(len(cities) * 0.5)
       print(f"Total nodes: {len(cities)}. Nodes to select: {num_to_select}. Repetitions: {NUM_REPETITIONS}")


       distance_matrix = np.array(
           [[round(math.sqrt((sc[0] - ec[0]) ** 2 + (sc[1] - ec[1]) ** 2)) for ec in cities] for sc in cities],
           dtype=int)


       # --- CHANGE 2: Run experiment for each method with repetitions ---
       for method_name in ['Greedy 2-Regret', 'Greedy Weighted Sum']:
           print(f"Running method: {method_name}...")


           objectives = []
           best_solution_for_method = {'objective': float('inf')}


           for i in range(NUM_REPETITIONS):
               # Call the solver
               if method_name == 'Greedy 2-Regret':
                   current_result = solve_optimized(distance_matrix, node_costs, num_to_select, k=2, method='2-regret',
                                                    num_starts=NUM_RANDOM_STARTS_PER_RUN)
               else:  # Greedy Weighted Sum
                   current_result = solve_optimized(distance_matrix, node_costs, num_to_select, k=2,
                                                    method='weighted-sum', num_starts=NUM_RANDOM_STARTS_PER_RUN)


               # Collect objective for stats
               objectives.append(current_result['objective'])


               # Keep track of the best solution found across all repetitions
               if current_result['objective'] < best_solution_for_method['objective']:
                   best_solution_for_method = current_result


           # Calculate stats
           min_obj = np.min(objectives)
           max_obj = np.max(objectives)
           avg_obj = np.mean(objectives)


           # Store the comprehensive results
           experiment_results.append({
               'instance': filename,
               'method': method_name,
               'min_obj': min_obj,
               'max_obj': max_obj,
               'avg_obj': avg_obj,
               'best_solution': best_solution_for_method  # Store the best tour for plotting
           })


   if not experiment_results:
       print("\nNo experiments were run. Please check file names.")
   else:
       # --- CHANGE 3: Update reporting to show the new stats ---
       print("\n\n--- FINAL REPORT ---")
       print("\n1. Computational Experiment Results:")
       print(f"{'Instance':<15} {'Method':<25} {'Min Obj':<12} {'Max Obj':<12} {'Avg Obj':<15}")
       print("-" * 80)
       for res in experiment_results:
           print(
               f"{res['instance']:<15} {res['method']:<25} {res['min_obj']:<12.0f} {res['max_obj']:<12.0f} {res['avg_obj']:<15.2f}")


       print("\n2. Best Solution Tours (for the overall best run of each method):")
       for res in experiment_results:
           # The best solution's objective is the 'min_obj' for that set of runs
           best_obj = res['best_solution']['objective']
           best_tour = res['best_solution']['tour']
           print(f"  Instance: {res['instance']}, Method: {res['method']}")
           print(f"    - Best Objective Found: {best_obj}")
           print(f"    - Tour: {best_tour[:-1]}")


       print("\n3. Generating Visualizations for Best Solutions...")
       for res in experiment_results:
           print(f"Plotting best result for '{res['instance']}' (Method: {res['method']})")
           cities, node_costs = load_instance(res['instance'])
           # We plot the best solution we saved
           plot_solution(res['best_solution'], cities, node_costs, res['instance'], res['method'])


       print("\nExperiment complete.")
