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


# ---
# ⚠️ 1. OBJECTIVE IS NOW MINIMIZED (Length + Cost)
# ---
def calculate_objective(path: List[int], D: List[List[int]], costs: List[int]) -> int:
    total_cost = 0
    total_distance = 0

    if not path:
        return 0

    for node_idx in path:
        total_cost += costs[node_idx]

    for i in range(len(path)):
        total_distance += D[path[i - 1]][path[i]]

    # Return the sum, which we want to MINIMIZE
    return total_cost + total_distance


# ---
# ⚠️ 2. FIND_BEST_SOLUTION NOW USES MIN()
# ---
def find_best_solution(solutions: List[Solution]) -> Solution:
    if not solutions:
        return Solution(path=[], objective=0)
    # Find the solution with the MINIMUM objective
    return min(solutions, key=lambda s: s.objective)


@dataclass
class MethodSpec:
    name: str
    use_cand: bool
    cand_k: int


def generate_random_path(n: int) -> List[int]:
    if n == 0:
        return []

    k = 100

    if k > n:
        logging.warning(f"k={k} is larger than n={n}. Using n.")
        k = n

    path = random.sample(range(n), k)
    return path


# This metric is still correct, as we want to find a neighbor 'j'
# that is close (low D[i][j]) and has a low cost (low costs[j]).
def _build_candidate_list(n: int, D: List[List[int]], costs: List[int], K: int) -> Dict[int, Set[int]]:
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
    N_problem_size = len(costs)
    current_path = initial_path.copy()
    current_objective = calculate_objective(current_path, D, costs)

    candidate_list = {}
    if use_candidates:
        candidate_list = _build_candidate_list(N_problem_size, D, costs, K)

    while True:
        # ---
        # ⚠️ 3. SEARCH IS INVERTED (Steepest DESCENT)
        # ---
        best_delta = 0  # We are now looking for the most NEGATIVE delta
        best_move = None

        path_indices = {node: idx for idx, node in enumerate(current_path)}
        nodes_out_path = set(range(N_problem_size)) - set(path_indices.keys())
        n_path = len(current_path)

        if n_path < 2:
            break

        if use_candidates:
            for i_idx, node_i in enumerate(current_path):
                i_prev_node = current_path[i_idx - 1]
                i_next_node = current_path[(i_idx + 1) % n_path]

                for node_j in candidate_list.get(node_i, set()):

                    if node_j in path_indices:
                        # CASE 1: 2-opt move
                        j_idx = path_indices[node_j]
                        node_i_next = current_path[(i_idx + 1) % n_path]
                        node_j_next = current_path[(j_idx + 1) % n_path]

                        if node_i_next == node_j or node_j_next == node_i:
                            continue

                        # ---
                        # ⚠️ 4. DELTA LOGIC IS INVERTED
                        # ---
                        # delta = (new_dist) - (old_dist)
                        delta = (D[node_i][node_j] + D[node_i_next][node_j_next]) - \
                                (D[node_i][node_i_next] + D[node_j][node_j_next])

                        if delta < best_delta:  # Looking for negative delta
                            best_delta = delta
                            idx1, idx2 = sorted([i_idx, j_idx])
                            best_move = ('2-opt', (idx1 + 1) % n_path, idx2)

                    else:
                        # CASE 2: Exchange move
                        # delta = (new_score) - (old_score)
                        delta_cost = costs[node_j] - costs[node_i]
                        delta_dist = (D[i_prev_node][node_j] + D[node_j][i_next_node]) - \
                                     (D[i_prev_node][node_i] + D[node_i][i_next_node])

                        delta = delta_cost + delta_dist  # Both are added to score

                        if delta < best_delta:  # Looking for negative delta
                            best_delta = delta
                            best_move = ('exchange', i_idx, node_j)

        else:
            # --- BASELINE SEARCH ---

            # 1. Exchange move
            for i_idx, node_i in enumerate(current_path):
                i_prev_node = current_path[i_idx - 1]
                i_next_node = current_path[(i_idx + 1) % n_path]

                for node_j in nodes_out_path:
                    # delta = (new_score) - (old_score)
                    delta_cost = costs[node_j] - costs[node_i]
                    delta_dist = (D[i_prev_node][node_j] + D[node_j][i_next_node]) - \
                                 (D[i_prev_node][node_i] + D[node_i][i_next_node])
                    delta = delta_cost + delta_dist

                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('exchange', i_idx, node_j)

            # 2. 2-opt move
            if n_path > 2:
                for i in range(n_path):
                    for j in range(i + 2, n_path):
                        node_i, node_i_next = current_path[i], current_path[(i + 1) % n_path]
                        node_j, node_j_next = current_path[j], current_path[(j + 1) % n_path]

                        if (i == 0 and j == n_path - 1):
                            continue

                        # delta = (new_dist) - (old_dist)
                        delta = (D[node_i][node_j] + D[node_i_next][node_j_next]) - \
                                (D[node_i][node_i_next] + D[node_j][node_j_next])

                        if delta < best_delta:
                            best_delta = delta
                            best_move = ('2-opt', (i + 1) % n_path, j)

        # Apply the best move if it's an improvement (negative delta)
        if best_delta < 0:
            move_type = best_move[0]
            if move_type == 'exchange':
                i_idx, node_j = best_move[1], best_move[2]
                current_path[i_idx] = node_j

            elif move_type == '2-opt':
                i, j = best_move[1], best_move[2]

                if i > j: i, j = j, i

                current_path[i: j + 1] = current_path[i: j + 1][::-1]

            current_objective = calculate_objective(current_path, D, costs)
        else:
            # No improving move found
            break

    return Solution(path=current_path, objective=current_objective)


def _local_search_worker(
        initial_path: List[int],
        D: List[List[int]],
        costs: List[int],
        method: MethodSpec
) -> Solution:
    try:
        return local_search_steepest(
            initial_path, D, costs, method.use_cand, method.cand_k
        )
    except Exception as e:
        logging.error(f"Error in worker process: {e}", exc_info=True)
        return Solution(path=[], objective=0)


def run_local_search_batch(
        D: List[List[int]],
        costs: List[int],
        method: MethodSpec,
        num_solutions: int
) -> List[Solution]:
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
