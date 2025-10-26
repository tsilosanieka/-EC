import numpy as np
import random
import time
import os
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


############################################################################
# 1. PROBLEM SETUP AND HELPER FUNCTIONS
############################################################################

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


def calculate_objective(tour, distance_matrix, node_costs):
   selected_nodes = tour[:-1]
   tour_length = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
   total_node_cost = sum(node_costs[node] for node in selected_nodes)
   return tour_length + total_node_cost


############################################################################
# 2. PLOTTING FUNCTION
############################################################################

def plot_solution(all_coords, all_costs, tour, title, filename):
    """Generates and saves a plot of the solution."""
    plt.figure(figsize=(12, 10))

    selected_indices = tour[:-1]
    # Use all_coords directly for indexing
    tour_coords = all_coords[tour]

    unselected_indices = list(set(range(len(all_coords))) - set(selected_indices))
    unselected_coords = all_coords[unselected_indices]

    plt.scatter(unselected_coords[:, 0], unselected_coords[:, 1], c='lightgray', s=20, label='Unused Nodes', zorder=1)
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'r-', zorder=2, color='darkslateblue', linewidth=1.5)
    scatter = plt.scatter(
        all_coords[selected_indices, 0],
        all_coords[selected_indices, 1],
        c=all_costs[selected_indices], # Use all_costs directly
        cmap='viridis', s=100, edgecolors='black', label='Selected Nodes', zorder=3
    )

    plt.title(title, fontsize=16); plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    cbar = plt.colorbar(scatter); cbar.set_label('Node Cost')

    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', filename)); plt.close()



############################################################################
# 3. STARTING SOLUTION GENERATORS (CONSTRUCTION HEURISTICS)
############################################################################

def generate_random_solution(num_all_nodes, num_to_select):
    """Generates a single random solution."""
    selected_nodes = random.sample(range(num_all_nodes), num_to_select)
    random.shuffle(selected_nodes)
    return selected_nodes + [selected_nodes[0]]


def find_best_insertions(tour, city, distance_matrix, node_costs, k):
    """Helper function to find the k best insertion positions for a city."""
    insertions = []
    for i in range(1, len(tour)):
        prev_node, curr_node = tour[i - 1], tour[i]
        delta_dist = distance_matrix[prev_node, city] + distance_matrix[city, curr_node] - distance_matrix[
            prev_node, curr_node]
        total_cost_change = delta_dist + node_costs[city]
        insertions.append({'cost': total_cost_change, 'pos': i})

    insertions.sort(key=lambda x: x['cost'])

    # Pad with dummy entries if fewer than k insertions are found
    while len(insertions) < k:
        insertions.append({'cost': float('inf'), 'pos': -1})

    return insertions[:k]


def generate_greedy_weighted_sum_solutions(dist_matrix, node_costs, num_to_select):
    """
    INTEGRATED VERSION using your regretful_insert logic, but adapted to generate 200 solutions.
    """
    print("Generating 200 greedy starting solutions (using your regretful_insert logic)...")
    all_solutions = []
    num_all_nodes = len(node_costs)
    k, w1, w2 = 2, 0.5, 0.5

    # Main loop to satisfy assignment requirement: one run for each starting node
    for start_node in range(num_all_nodes):
        # Step 1: Initialize tour with start_node and its best partner
        best_partner, min_initial_obj = -1, float('inf')
        for partner in range(num_all_nodes):
            if partner == start_node: continue
            obj = 2 * dist_matrix[start_node, partner] + node_costs[start_node] + node_costs[partner]
            if obj < min_initial_obj:
                min_initial_obj, best_partner = obj, partner

        # Tour representation: [start, partner, start] for insertion logic
        tour = [start_node, best_partner, start_node]
        unvisited = set(range(num_all_nodes)) - {start_node, best_partner}

        # Step 2: Iteratively insert the best nodes
        while len(tour) - 1 < num_to_select:
            if not unvisited: break

            candidates = []
            for city in unvisited:
                best_insert_data = find_best_insertions(tour, city, dist_matrix, node_costs, k)
                best_change = best_insert_data[0]['cost']
                best_pos = best_insert_data[0]['pos']

                regret = -float('inf')
                if len(best_insert_data) >= k and best_insert_data[k - 1]['pos'] != -1:
                    second_best_change = best_insert_data[k - 1]['cost']
                    regret = second_best_change - best_change

                criterion = w1 * regret - w2 * best_change
                candidates.append((criterion, city, best_pos))

            _, best_city, best_pos = max(candidates, key=lambda x: x[0])

            tour.insert(best_pos, best_city)
            unvisited.remove(best_city)

        # Add the final tour (in [n1, n2, ..., n_k, n1] format) to the list of solutions
        all_solutions.append(tour)

    return all_solutions


############################################################################
# 4. LOCAL SEARCH CORE LOGIC (DELTAS AND MOVES)
############################################################################

def calculate_delta_inter_route_swap(tour, u_idx, v_node, dist_matrix, node_costs):
    tour_nodes = tour[:-1]
    u_node, p_node, s_node = tour_nodes[u_idx], tour_nodes[u_idx - 1], tour_nodes[(u_idx + 1) % len(tour_nodes)]
    return (node_costs[v_node] - node_costs[u_node]) + \
           (dist_matrix[p_node, v_node] + dist_matrix[v_node, s_node]) - \
           (dist_matrix[p_node, u_node] + dist_matrix[u_node, s_node])

def calculate_delta_intra_route_node_swap(tour, i, j, dist_matrix):
    tour_nodes = tour[:-1]; n = len(tour_nodes)
    if i > j: i, j = j, i
    node_i, node_j = tour_nodes[i], tour_nodes[j]
    p_i, s_i = tour_nodes[i - 1], tour_nodes[(i + 1) % n]
    p_j, s_j = tour_nodes[j - 1], tour_nodes[(j + 1) % n]
    if (j - i) % n == 1:
        return (dist_matrix[p_i, node_j] + dist_matrix[node_i, s_j]) - (dist_matrix[p_i, node_i] + dist_matrix[node_j, s_j])
    return (dist_matrix[p_i, node_j] + dist_matrix[node_j, s_i] + dist_matrix[p_j, node_i] + dist_matrix[node_i, s_j]) - \
           (dist_matrix[p_i, node_i] + dist_matrix[node_i, s_i] + dist_matrix[p_j, node_j] + dist_matrix[node_j, s_j])

def calculate_delta_intra_route_edge_swap(tour, i, j, dist_matrix):
    tour_nodes = tour[:-1]; n = len(tour_nodes)
    node1, node2, node3, node4 = tour_nodes[i], tour_nodes[(i + 1) % n], tour_nodes[j], tour_nodes[(j + 1) % n]
    return (dist_matrix[node1, node3] + dist_matrix[node2, node4]) - (dist_matrix[node1, node2] + dist_matrix[node3, node4])

def apply_inter_route_swap(tour, u_idx, v_node):
    new_tour = tour[:-1]; new_tour[u_idx] = v_node; return new_tour + [new_tour[0]]
def apply_intra_route_node_swap(tour, i, j):
    new_tour = tour[:-1]; new_tour[i], new_tour[j] = new_tour[j], new_tour[i]; return new_tour + [new_tour[0]]
def apply_intra_route_edge_swap(tour, i, j):
    new_tour = tour[:-1]
    if i > j: i, j = j, i
    segment = new_tour[i+1 : j+1]; segment.reverse()
    return new_tour[:i+1] + segment + new_tour[j+1:] + [new_tour[0]]

def local_search(start_tour, dist_matrix, node_costs, search_type, intra_route_type):
    current_tour, current_obj = list(start_tour), calculate_objective(start_tour, dist_matrix, node_costs)
    num_all_nodes = len(node_costs)
    while True:
        tour_nodes, n = current_tour[:-1], len(current_tour) - 1
        if search_type == 'steepest':
            best_move, best_delta = None, 0
            non_tour_nodes = list(set(range(num_all_nodes)) - set(tour_nodes))
            for i, v_node in itertools.product(range(n), non_tour_nodes):
                delta = calculate_delta_inter_route_swap(current_tour, i, v_node, dist_matrix, node_costs)
                if delta < best_delta: best_delta, best_move = delta, ('inter_swap', i, v_node)
            if intra_route_type == 'nodes':
                for i, j in itertools.combinations(range(n), 2):
                    delta = calculate_delta_intra_route_node_swap(current_tour, i, j, dist_matrix)
                    if delta < best_delta: best_delta, best_move = delta, ('intra_node_swap', i, j)
            elif intra_route_type == 'edges':
                for i, j in itertools.combinations(range(n), 2):
                    if abs(i - j) <= 1 or (i==0 and j==n-1): continue
                    delta = calculate_delta_intra_route_edge_swap(current_tour, i, j, dist_matrix)
                    if delta < best_delta: best_delta, best_move = delta, ('intra_edge_swap', i, j)
            if best_move:
                current_obj += best_delta
                move_type, arg1, arg2 = best_move
                if move_type == 'inter_swap': current_tour = apply_inter_route_swap(current_tour, arg1, arg2)
                elif move_type == 'intra_node_swap': current_tour = apply_intra_route_node_swap(current_tour, arg1, arg2)
                elif move_type == 'intra_edge_swap': current_tour = apply_intra_route_edge_swap(current_tour, arg1, arg2)
            else: break
        elif search_type == 'greedy':
            all_moves, improvement_found = [], False
            non_tour_nodes = list(set(range(num_all_nodes)) - set(tour_nodes))
            for i, v_node in itertools.product(range(n), non_tour_nodes): all_moves.append(('inter_swap', i, v_node))
            if intra_route_type == 'nodes':
                for i, j in itertools.combinations(range(n), 2): all_moves.append(('intra_node_swap', i, j))
            elif intra_route_type == 'edges':
                for i, j in itertools.combinations(range(n), 2):
                    if abs(i-j) <= 1 or (i==0 and j==n-1): continue
                    all_moves.append(('intra_edge_swap', i, j))
            random.shuffle(all_moves)
            for move in all_moves:
                move_type, arg1, arg2 = move
                delta = 0
                if move_type == 'inter_swap': delta = calculate_delta_inter_route_swap(current_tour, arg1, arg2, dist_matrix, node_costs)
                elif move_type == 'intra_node_swap': delta = calculate_delta_intra_route_node_swap(current_tour, arg1, arg2, dist_matrix)
                elif move_type == 'intra_edge_swap': delta = calculate_delta_intra_route_edge_swap(current_tour, arg1, arg2, dist_matrix)
                if delta < 0:
                    current_obj += delta
                    if move_type == 'inter_swap': current_tour = apply_inter_route_swap(current_tour, arg1, arg2)
                    elif move_type == 'intra_node_swap': current_tour = apply_intra_route_node_swap(current_tour, arg1, arg2)
                    elif move_type == 'intra_edge_swap': current_tour = apply_intra_route_edge_swap(current_tour, arg1, arg2)
                    improvement_found = True
                    break
            if not improvement_found: break
    return current_tour, current_obj


############################################################################
# 5. COMPUTATIONAL EXPERIMENT
############################################################################

def main():
    random.seed(123)
    np.random.seed(123)
    instance_files = ['tspa.csv', 'tspb.csv']  # Make sure these files are in your data folder
    num_runs = 200
    configurations = [
        {'search': 'steepest', 'intra': 'nodes', 'start': 'random'},
        {'search': 'greedy', 'intra': 'nodes', 'start': 'random'},
        {'search': 'steepest', 'intra': 'edges', 'start': 'random'},
        {'search': 'greedy', 'intra': 'edges', 'start': 'random'},
        {'search': 'steepest', 'intra': 'nodes', 'start': 'greedy'},
        {'search': 'greedy', 'intra': 'nodes', 'start': 'greedy'},
        {'search': 'steepest', 'intra': 'edges', 'start': 'greedy'},
        {'search': 'greedy', 'intra': 'edges', 'start': 'greedy'},
    ]
    results_summary = {}

    for instance_file in instance_files:
        print(f"\n{'=' * 20} PROCESSING INSTANCE: {instance_file} {'=' * 20}")
        filepath = instance_file
        if not os.path.exists(filepath):
            print(f"ERROR: Instance file not found at {filepath}. Please check the path.")
            continue

        try:
            cities, node_costs = load_instance(instance_file)
        except FileNotFoundError:
            print(f"Error: The file '{instance_file}' was not found.")
            continue

        distance_matrix = np.array(
            [[round(math.sqrt((sc[0] - ec[0]) ** 2 + (sc[1] - ec[1]) ** 2)) for ec in cities] for sc in cities],
            dtype=int)

        num_all_nodes, num_to_select = len(node_costs), math.ceil(len(cities) * 0.5)

        print(f"Generating {num_runs} random starting solutions...")
        random_starts = [generate_random_solution(num_all_nodes, num_to_select) for _ in range(num_runs)]
        greedy_starts = generate_greedy_weighted_sum_solutions(distance_matrix, node_costs, num_to_select)
        results_summary[instance_file] = []

        for config in configurations:
            search_type, intra_type, start_type = config['search'], config['intra'], config['start']
            method_name = f"LS {search_type.capitalize():<8} | Intra: {intra_type.capitalize():<5} | Start: {start_type.capitalize()}"
            print(f"\n--- Running Method: {method_name} ---")

            starting_solutions = random_starts if start_type == 'random' else greedy_starts
            run_objectives, run_times, best_tour_for_method, min_obj_for_method = [], [], None, float('inf')

            for i in tqdm(range(num_runs), desc=f"{start_type.capitalize()} Starts"):
                start_time = time.time()
                final_tour, final_obj = local_search(starting_solutions[i], distance_matrix, node_costs, search_type,
                                                     intra_type)
                run_times.append(time.time() - start_time)
                run_objectives.append(final_obj)
                if final_obj < min_obj_for_method: min_obj_for_method, best_tour_for_method = final_obj, final_tour

            results_summary[instance_file].append({
                'name': method_name, 'avg_obj': np.mean(run_objectives), 'min_obj': np.min(run_objectives),
                'max_obj': np.max(run_objectives), 'avg_time': np.mean(run_times),
                'min_time': np.min(run_times), 'max_time': np.max(run_times)
            })

            # FIX #4: This call now works because it passes the correct, separate variables
            plot_title = f"Best Solution for '{instance_file}'\nMethod: {method_name}\nObjective: {min_obj_for_method:.0f}"
            plot_filename = f"{instance_file.split('.')[0]}_{search_type}_{intra_type}_{start_type}.png"
            plot_solution(cities, node_costs, best_tour_for_method, plot_title, plot_filename)
            print(f"    -> Best plot saved to plots/{plot_filename}")

    print("\n\n" + "=" * 80);
    print(" " * 25 + "FINAL COMPUTATIONAL RESULTS");
    print("=" * 80)
    for instance, results in results_summary.items():
        print(f"\nINSTANCE: {instance}\n");
        print("--- Objective Function Values ---")
        print(f"{'Method':<60} {'Avg (Min - Max)':<25}");
        print("-" * 85)
        for res in results: print(
            f"{res['name']:<60} {f'{res['avg_obj']:.2f} ({res['min_obj']:.0f} - {res['max_obj']:.0f})':<25}")
        print("\n--- Running Times (seconds) ---")
        print(f"{'Method':<60} {'Avg (Min - Max)':<25}");
        print("-" * 85)
        for res in results: print(
            f"{res['name']:<60} {f'{res['avg_time']:.4f}s ({res['min_time']:.4f}s - {res['max_time']:.4f}s)':<25}")


if __name__ == "__main__":
    main()
