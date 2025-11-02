import math
import random
import time
import logging
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

@dataclass
class Solution:
    path: List[int]
    objective: int = 0

    def copy(self) -> 'Solution':
        return Solution(path=self.path.copy(), objective=self.objective)


def calculate_objective(path: List[int], D: List[List[int]], costs: List[int]) -> int:
    objective = 0
    total_distance = 0

    if not path:
        return 0

    for node_idx in path:
        objective += costs[node_idx]

    if not path:
        return 0

    for i in range(len(path)):
        total_distance += D[path[i - 1]][path[i]]

    return objective - total_distance

def find_best_solution(solutions: List[Solution]) -> Solution:
    if not solutions:
        return Solution(path=[], objective=0)
    return max(solutions, key=lambda s: s.objective)

@dataclass
class MethodSpec:
    name: str
    use_cand: bool
    cand_k: int


def generate_random_path(n: int) -> List[int]:
    """
    Generates a random solution for the Selective TSP.
    Picks exactly 100 nodes from the n (200) total nodes.
    """
    if n == 0:
        return []

    k = 100

    if k > n:
        logging.warning(f"k={k} is larger than n={n}. Using n.")
        k = n

    path = random.sample(range(n), k)
    return path


def _build_candidate_list(n: int, D: List[List[int]], costs: List[int], K: int) -> Dict[int, Set[int]]:
    """
    Builds a candidate list for each node.
    Metric: (Distance + Cost of neighbor)
    """
    candidate_list = {}
    for i in range(n):
        distances = []
        for j in range(n):
            if i == j:
                continue
            metric = D[i][j] + costs[j]
            distances.append((metric, j))

        distances.sort()
        candidate_list[i] = {neighbor_idx for _, neighbor_idx in distances[:K]}
    return candidate_list

def local_search_steepest(
        initial_path: List[int],
        D: List[List[int]],
        costs: List[int],
        use_candidates: bool,
        K: int
) -> Solution:
    """
    Performs a steepest-ascent local search for PCTSP.

    It evaluates two neighborhoods:
    1. Inter-route (Exchange): Swap a node IN the path with one OUT.
    2. Intra-route (2-opt): Re-order nodes IN the path.

    It implements the candidate-list logic as described in Assignment 4.
    """

    N_problem_size = len(costs)
    current_path = initial_path.copy()
    current_objective = calculate_objective(current_path, D, costs)

    candidate_list = {}
    if use_candidates:
        candidate_list = _build_candidate_list(N_problem_size, D, costs, K)

    while True:
        best_delta = 0
        best_move = None  # e.g., ('exchange', in_idx, out_node) or ('2-opt', idx1, idx2)

        path_indices = {node: idx for idx, node in enumerate(current_path)}
        nodes_out_path = set(range(N_problem_size)) - set(path_indices.keys())
        n_path = len(current_path)

        if n_path < 2:
            break  # Can't do moves on a path this small

        if use_candidates:
            # --- EFFICIENT CANDIDATE SEARCH (O(n_path * K)) ---
            # "first loop over all selected nodes and then second loop over all nodes nearest nodes"
            for i_idx, node_i in enumerate(current_path):
                i_prev_node = current_path[i_idx - 1]
                i_next_node = current_path[(i_idx + 1) % n_path]

                for node_j in candidate_list[node_i]:

                    if node_j in path_indices:
                        # CASE 1: Both nodes IN path. Evaluate 2-opt move.
                        j_idx = path_indices[node_j]

                        # Ensure (i, i+1) and (j, j+1) are not adjacent
                        if j_idx == i_idx or j_idx == (i_idx + 1) % n_path or (j_idx + 1) % n_path == i_idx:
                            continue

                        j_next_node = current_path[(j_idx + 1) % n_path]

                        # Use i+1 and j+1 as the nodes *after* i and j
                        i_next_idx = (i_idx + 1) % n_path
                        j_next_idx = (j_idx + 1) % n_path

                        node_i_next = current_path[i_next_idx]
                        node_j_next = current_path[j_next_idx]

                        # Ensure i < j for simplicity
                        idx1, idx2 = sorted([i_idx, j_idx])
                        node1, node1_next = current_path[idx1], current_path[(idx1 + 1) % n_path]
                        node2, node2_next = current_path[idx2], current_path[(idx2 + 1) % n_path]

                        delta = (D[node1][node1_next] + D[node2][node2_next]) - \
                                (D[node1][node2] + D[node1_next][node2_next])

                        if delta > best_delta:
                            best_delta = delta
                            best_move = ('2-opt', (idx1 + 1) % n_path, idx2)

                    else:
                        # CASE 2: node_i IN path, node_j OUT. Evaluate exchange move.
                        # This is a candidate "inter-route" move.
                        delta_cost = costs[node_j] - costs[node_i]
                        delta_dist = (D[i_prev_node][node_j] + D[node_j][i_next_node]) - \
                                     (D[i_prev_node][node_i] + D[node_i][i_next_node])

                        delta = delta_cost - delta_dist

                        if delta > best_delta:
                            best_delta = delta
                            best_move = ('exchange', i_idx, node_j)

        else:
            # --- INEFFICIENT BASELINE SEARCH ---

            # 1. Find best Inter-route (Exchange) move (O(n_path * n_out))
            for i_idx, node_i in enumerate(current_path):
                i_prev_node = current_path[i_idx - 1]
                i_next_node = current_path[(i_idx + 1) % n_path]

                for node_j in nodes_out_path:
                    delta_cost = costs[node_j] - costs[node_i]
                    delta_dist = (D[i_prev_node][node_j] + D[node_j][i_next_node]) - \
                                 (D[i_prev_node][node_i] + D[node_i][i_next_node])
                    delta = delta_cost - delta_dist

                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('exchange', i_idx, node_j)

            # 2. Find best Intra-route (2-opt) move (O(n_path^2))
            for i in range(n_path):
                for j in range(i + 2, n_path):  # j starts at i+2
                    node_i, node_i_next = current_path[i], current_path[(i + 1) % n_path]
                    node_j, node_j_next = current_path[j], current_path[(j + 1) % n_path]

                    if (i == 0 and j == n_path - 1):  # Avoid swapping first and last edge
                        continue

                    delta = (D[node_i][node_i_next] + D[node_j][node_j_next]) - \
                            (D[node_i][node_j] + D[node_i_next][node_j_next])

                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('2-opt', (i + 1) % n_path, j)

        # --- APPLY THE BEST MOVE FOUND ---
        if best_move:
            move_type = best_move[0]
            if move_type == 'exchange':
                i_idx, node_j = best_move[1], best_move[2]
                current_path[i_idx] = node_j

            elif move_type == '2-opt':
                i, j = best_move[1], best_move[2]

                # Ensure i < j
                if i > j:
                    i, j = j, i

                # Reverse the slice
                current_path[i: j + 1] = current_path[i: j + 1][::-1]

            # Recalculate objective after move
            current_objective = calculate_objective(current_path, D, costs)
        else:
            # No improving move found, stop.
            break

    return Solution(path=current_path, objective=current_objective)


# --- Worker function for parallel execution ---

def _local_search_worker(
        initial_path: List[int],
        D: List[List[int]],
        costs: List[int],
        method: MethodSpec
) -> Solution:
    """
    A top-level wrapper function for ProcessPoolExecutor.
    """
    try:
        # Call the new, correct local search function
        return local_search_steepest(
            initial_path, D, costs, method.use_cand, method.cand_k
        )
    except Exception as e:
        logging.error(f"Error in worker process: {e}")
        return Solution(path=[], objective=0)


def run_local_search_batch(
        D: List[List[int]],
        costs: List[int],
        method: MethodSpec,
        num_solutions: int
) -> List[Solution]:
    """
    Runs a batch of local searches in parallel.
    """
    n = len(costs)
    solutions = []

    initial_paths = [generate_random_path(n) for _ in range(num_solutions)]

    with ProcessPoolExecutor() as executor:
        worker_func = partial(
            _local_search_worker,
            D=D,
            costs=costs,
            method=method
        )

        futures = [executor.submit(worker_func, path) for path in initial_paths]

        for future in as_completed(futures):
            result = future.result()
            if result.path:
                solutions.append(result)

    return solutions
