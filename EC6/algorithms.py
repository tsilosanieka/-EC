import random
import time
import numpy as np
from numba import njit
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

@dataclass
class Solution:
    path: List[int] = field(default_factory=list)
    objective: int = 0

@njit(fastmath=True)
def calculate_objective_numba(D, costs, path):
    total_sum = 0
    n = len(path)
    for i in range(n - 1):
        total_sum += D[path[i], path[i + 1]]
    total_sum += D[path[n - 1], path[0]] 
    for i in range(n):
        total_sum += costs[path[i]]

    return total_sum


@njit(fastmath=True)
def delta_two_opt_numba(D, path, i, j):
    n = len(path)
    idx_i = i
    idx_i_next = (i + 1) % n
    idx_j = j
    idx_j_next = (j + 1) % n

    a = path[idx_i]
    b = path[idx_i_next]
    c = path[idx_j]
    d = path[idx_j_next]

    if b == c or a == d:
        return 0

    current_dist = D[a, b] + D[c, d]
    new_dist = D[a, c] + D[b, d]

    return new_dist - current_dist


@njit(fastmath=True)
def delta_exchange_numba(D, costs, path, i, u, prev_idx_val, next_idx_val):
    a = path[prev_idx_val]
    v = path[i]
    b = path[next_idx_val]

    before = D[a, v] + D[v, b] + costs[v]
    after = D[a, u] + D[u, b] + costs[u]

    return after - before


@njit(fastmath=True)
def apply_two_opt_numba(path, i, j):
    p1 = i + 1
    p2 = j
    while p1 < p2:
        path[p1], path[p2] = path[p2], path[p1]
        p1 += 1
        p2 -= 1


@njit(fastmath=True)
def solve_steepest_numba(D, costs, initial_path, full_node_count):
    path = initial_path.copy()
    n = len(path)

    is_selected = np.zeros(full_node_count, dtype=np.bool_)
    for p in path:
        is_selected[p] = True

    while True:
        best_delta = 0
        best_move_type = 0  
        best_args_1 = -1
        best_args_2 = -1

        for i in range(n):
            for j in range(i + 2, n + (1 if i > 0 else 0)):
          
                if j >= n: continue 

                idx_i = i
                idx_i_next = i + 1
                if idx_i_next == n: idx_i_next = 0

                idx_j = j
                idx_j_next = j + 1
                if idx_j_next == n: idx_j_next = 0

                a = path[idx_i]
                b = path[idx_i_next]
                c = path[idx_j]
                d = path[idx_j_next]

                change = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])

                if change < best_delta:
                    best_delta = change
                    best_move_type = 1
                    best_args_1 = i
                    best_args_2 = j

        for i in range(n):
            # Pre-calc neighbors
            idx_prev = n - 1 if i == 0 else i - 1
            idx_next = 0 if i == n - 1 else i + 1

            a = path[idx_prev]
            v = path[i]
            b = path[idx_next]

            base_cost = D[a, v] + D[v, b] + costs[v]

            for u in range(full_node_count):
                if is_selected[u]:
                    continue

                # Calculate delta
                change = (D[a, u] + D[u, b] + costs[u]) - base_cost

                if change < best_delta:
                    best_delta = change
                    best_move_type = 2
                    best_args_1 = i
                    best_args_2 = u

        if best_delta < 0:
            if best_move_type == 1:
                # Apply 2-opt
                apply_two_opt_numba(path, best_args_1, best_args_2)
            elif best_move_type == 2:
                # Apply Exchange
                idx_in_path = best_args_1
                new_node = best_args_2
                old_node = path[idx_in_path]

                path[idx_in_path] = new_node
                is_selected[old_node] = False
                is_selected[new_node] = True
        else:
            break

    return path

def calculate_objective(D: List[List[int]], costs: List[int], path: List[int]) -> int:
    D_np = np.array(D, dtype=np.int32) if not isinstance(D, np.ndarray) else D
    c_np = np.array(costs, dtype=np.int32) if not isinstance(costs, np.ndarray) else costs
    p_np = np.array(path, dtype=np.int32) if not isinstance(path, np.ndarray) else path
    return int(calculate_objective_numba(D_np, c_np, p_np))


def start_random(D, costs) -> Solution:
    n = len(D)
    k = (n + 1) // 2
    idx = list(range(n))
    random.shuffle(idx)
    path = idx[:k]
    random.shuffle(path)

    obj = calculate_objective(D, costs, path)
    return Solution(path, obj)


def local_search_steepest_baseline(D, costs, init_sol: Solution) -> Solution:
    D_np = np.array(D, dtype=np.int32) if not isinstance(D, np.ndarray) else D
    costs_np = np.array(costs, dtype=np.int32) if not isinstance(costs, np.ndarray) else costs
    path_np = np.array(init_sol.path, dtype=np.int32)

    final_path_np = solve_steepest_numba(D_np, costs_np, path_np, len(D))

    final_path = final_path_np.tolist()
    final_obj = calculate_objective_numba(D_np, costs_np, final_path_np)

    return Solution(final_path, int(final_obj))

class PerturbationType(Enum):
    DOUBLE_EXCHANGE = auto()
    RANDOM_4_OPT = auto()
    PATH_DESTROY = auto()  

@dataclass
class ILSResult:
    best_solution: Solution
    num_ls_iterations: int
    elapsed_seconds: float
    all_solutions: List[Solution]


@dataclass
class MSLSResult:
    best_solution: Solution
    num_ls_iterations: int
    elapsed_seconds: float
    all_solutions: List[Solution]


def run_msls(D: List[List[int]], costs: List[int], iterations: int) -> MSLSResult:
    # Pre-convert to numpy to save time
    D_np = np.array(D, dtype=np.int32)
    costs_np = np.array(costs, dtype=np.int32)

    start_time = time.time()

    # Initial
    current_sol = start_random(D, costs)
    # Pass numpy arrays to avoid re-conversion
    current_sol = local_search_steepest_baseline(D_np, costs_np, current_sol)

    best_sol = current_sol
    all_sols = [current_sol]

    for _ in range(1, iterations):
        # Start Random
        rnd_sol = start_random(D, costs)
        # Local Search
        optimized = local_search_steepest_baseline(D_np, costs_np, rnd_sol)

        all_sols.append(optimized)
        if optimized.objective < best_sol.objective:
            best_sol = optimized

    return MSLSResult(best_sol, iterations, time.time() - start_time, all_sols)

def perturb_random_4_opt(sol: Solution, n_full: int) -> Solution:
    path = list(sol.path)
    n = len(path)
    if n < 4: return sol

  num_moves = 2 + random.randint(0, 1)
    for _ in range(num_moves):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i > j: i, j = j, i
        if j - i > 1 and j - i < n - 1:
            # Python swap implementation
            path[i + 1:j + 1] = reversed(path[i + 1:j + 1])

    return Solution(path, 0) 


def run_ils(D: List[List[int]], costs: List[int], time_limit: float, p_type: PerturbationType) -> ILSResult:
    D_np = np.array(D, dtype=np.int32)
    costs_np = np.array(costs, dtype=np.int32)

    start_time = time.time()

    current = start_random(D, costs)
    current = local_search_steepest_baseline(D_np, costs_np, current)

    best_sol = current
    iterations = 1
    all_sols = [current]

    while (time.time() - start_time) < time_limit:
        # Perturb (Pure Python logic for simplicity)
        perturbed = perturb_random_4_opt(current, len(D))

        # Local Search (Numba Optimized)
        local_opt = local_search_steepest_baseline(D_np, costs_np, perturbed)

        iterations += 1
        all_sols.append(local_opt)

        if local_opt.objective < current.objective:
            current = local_opt

        if local_opt.objective < best_sol.objective:
            best_sol = local_opt

    return ILSResult(best_sol, iterations, time.time() - start_time, all_sols)


def find_best_solution(solutions):
    if not solutions: return Solution()
    return min(solutions, key=lambda s: s.objective)
