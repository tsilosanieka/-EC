import math
import random
import time
import logging
import sys  # For printing to stderr
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from enum import Enum


# --- From solution.go ---
@dataclass
class Solution:
    Path: List[int]  # Use Go's capitalization for consistency
    Objective: int = 0

    def copy(self) -> 'Solution':
        return Solution(Path=self.Path.copy(), Objective=self.Objective)


# --- From objective.go ---
@dataclass
class MethodSpec:
    Name: str
    UseCand: bool = False  # should use candidate moves?
    CandK: int = 0  # how many nearest to include in candidate list
    UseLM: bool = False  # should use list-of-moves (LM) delta reuse?


# --- From best_solution.go ---
def find_best_solution(solutions: List[Solution]) -> Solution:
    """ Finds the solution with the MINIMUM objective value. """
    if not solutions:
        return Solution(Path=[], Objective=float('inf'))
    return min(solutions, key=lambda s: s.Objective)


# --- From objective.go ---
def calculate_objective(path: List[int], D: List[List[int]], costs: List[int]) -> int:
    """ Calculates the objective function to be MINIMIZED (Cost + Distance). """
    if not path:
        return sys.maxsize // 4

    total_cost = 0
    total_distance = 0

    for node_idx in path:
        total_cost += costs[node_idx]

    n_path = len(path)
    if n_path > 1:
        for i in range(n_path):
            total_distance += D[path[i - 1]][path[i]]

    return total_cost + total_distance


def prev_idx(i: int, n: int) -> int:
    """ Returns the previous index in a cyclic path of length n. """
    return n - 1 if i == 0 else i - 1


def next_idx(i: int, n: int) -> int:
    """ Returns the next index in a cyclic path of length n. """
    return (i + 1) % n


def select_count(n: int) -> int:
    """ Returns the number of nodes to select (50% rounded up). """
    # Use integer division
    return (n + 1) // 2


def start_random(D: List[List[int]], costs: List[int]) -> Solution:
    """ Builds an initial solution by selecting 50% rounded up nodes at random. """
    n = len(D)
    k = select_count(n)

    nodes = list(range(n))
    random.shuffle(nodes)

    path = nodes[:k]
    random.shuffle(path)  # Shuffle the selected nodes

    return Solution(Path=path, Objective=calculate_objective(path, D, costs))


# --- From neighborhood.go ---
def delta_two_opt(D: List[List[int]], path: List[int], i: int, j: int) -> int:
    """ intra-route move - two edges exchange: 2-opt between path[i] and path[j] """
    if i == j:
        return 0
    n = len(path)
    if next_idx(i, n) == j or next_idx(j, n) == i:
        return 0  # Adjacent edges

    a = path[i]
    b = path[next_idx(i, n)]
    c = path[j]
    d = path[next_idx(j, n)]

    before = D[a][b] + D[c][d]
    after = D[a][c] + D[b][d]
    return after - before


def delta_exchange_selected(D: List[List[int]], costs: List[int], path: List[int], i: int, u: int) -> int:
    """ inter-route move - two-nodes exchange - path[i] with u (u outside path) """
    n = len(path)
    a = path[prev_idx(i, n)]
    v = path[i]
    b = path[next_idx(i, n)]

    before = D[a][v] + D[v][b] + costs[v]
    after = D[a][u] + D[u][b] + costs[u]
    return after - before


def apply_two_opt(path: List[int], i: int, j: int):
    """ applyTwoOpt performs a 2-opt move on the path between indices i and j. """
    n = len(path)
    if i == j or next_idx(i, n) == j or next_idx(j, n) == i:
        return
    if i > j:
        i, j = j, i
    # Reverse segment [i+1...j]
    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]


def apply_two_opt_and_update_pos(path: List[int], pos_of: List[int], i: int, j: int):
    """ Applies 2-opt and keeps the position index array `pos_of` in sync. """
    n = len(path)
    if i == j or next_idx(i, n) == j or next_idx(j, n) == i:
        return
    if i > j:
        i, j = j, i
    # reverse segment [i+1..j]
    segment_to_reverse = path[i + 1:j + 1]
    path[i + 1:j + 1] = segment_to_reverse[::-1]

    # Update positions for the reversed segment
    for k, node in enumerate(path[i + 1:j + 1]):
        pos_of[node] = i + 1 + k


def apply_exchange_selected(path: List[int], i: int, u: int):
    """ applyExchangeSelected replaces the selected vertex at position i with u. """
    path[i] = u


# --- From runner.go (Baseline) ---
def local_search_steepest_baseline(D: List[List[int]], costs: List[int], init: Solution) -> Solution:
    """ Performs steepest local search on the full neighborhood. """
    path = init.Path[:]  # Copy
    n = len(path)

    while True:
        best_delta = 0
        best_move: Optional[Callable[[], None]] = None

        # 1. Intra-route move - 2-opt
        for i in range(n):
            for j in range(i + 1, n):  # Go: j := i + 1
                dl = delta_two_opt(D, path, i, j)
                if dl < best_delta:
                    ii, jj = i, j
                    best_delta = dl
                    best_move = lambda ii=ii, jj=jj: apply_two_opt(path, ii, jj)

        # 2. Inter-route moves - node exchange
        in_sel = [False] * len(D)
        for v in path:
            in_sel[v] = True

        non_sel = [u for u in range(len(D)) if not in_sel[u]]

        for i in range(n):
            for u in non_sel:
                dl = delta_exchange_selected(D, costs, path, i, u)
                if dl < best_delta:
                    ii, uu = i, u
                    best_delta = dl
                    best_move = lambda ii=ii, uu=uu: apply_exchange_selected(path, ii, uu)

        if best_delta < 0 and best_move is not None:
            best_move()
            # Note: The Go baseline has a bug: it doesn't update `non_sel` after an exchange.
            # This Python version replicates that logic.
        else:
            break

    return Solution(Path=path, Objective=calculate_objective(path, D, costs))


# --- From candidates.go ---
@dataclass
class CandData:
    cand_list: List[List[int]]
    is_cand: Set[Tuple[int, int]]


def pack_edge(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def build_candidates(D: List[List[int]], costs: List[int], K: int) -> CandData:
    n = len(D)
    if K <= 0:
        K = 10

    cand_list = [[] for _ in range(n)]
    is_cand = set()

    for u in range(n):
        nbs = []
        for v in range(n):
            if v == u:
                continue
            w = D[u][v] + costs[v]
            nbs.append((v, w))

        # Sort by weight
        nbs.sort(key=lambda x: x[1])

        m = min(K, len(nbs))
        list_for_u = [nbs[i][0] for i in range(m)]
        cand_list[u] = list_for_u

        for v in list_for_u:
            is_cand.add(pack_edge(u, v))

    return CandData(cand_list=cand_list, is_cand=is_cand)


def is_candidate_edge(cd: CandData, a: int, b: int) -> bool:
    return pack_edge(a, b) in cd.is_cand


def local_search_steepest_candidates(D: List[List[int]], costs: List[int], init: Solution, cd: CandData) -> Solution:
    path = init.Path[:]
    n = len(path)
    dim = len(D)

    # Use standard lists for pos_of/in_sel, they are O(1) for index access
    pos_of = [-1] * dim
    in_sel = [False] * dim
    for i, v in enumerate(path):
        pos_of[v] = i
        in_sel[v] = True

    visit_mark = [-1] * dim
    epoch = 0

    while True:
        best_delta = 0
        best_move: Optional[Callable[[], None]] = None

        # 1. Intra-route (2-opt)
        for i in range(n):
            n1 = path[i]
            ip1 = next_idx(i, n)
            im1 = prev_idx(i, n)

            for n2 in cd.cand_list[n1]:
                j = pos_of[n2]
                if j == -1:  # n2 is not selected
                    continue
                if j == i or j == ip1 or j == im1:  # neighbors/degenerate
                    continue

                # Prune symmetric
                if j > i:
                    # MOVE A: 2-opt(i, j)
                    dlA = delta_two_opt(D, path, i, j)
                    if dlA < best_delta:
                        ii, jj = i, j
                        best_delta = dlA
                        best_move = lambda ii=ii, jj=jj: apply_two_opt_and_update_pos(path, pos_of, ii, jj)

                # MOVE B: 2-opt(prev(i), prev(j))
                ii = prev_idx(i, n)
                jj = prev_idx(j, n)
                # Need to check for neighbor/degenerate again for (i-1, j-1)
                if ii == jj or next_idx(ii, n) == jj or next_idx(jj, n) == ii:
                    continue

                dlB = delta_two_opt(D, path, ii, jj)
                if dlB < best_delta:
                    iii, jjj = ii, jj
                    best_delta = dlB
                    best_move = lambda iii=iii, jjj=jjj: apply_two_opt_and_update_pos(path, pos_of, iii, jjj)

        # 2. Inter-route (Exchange)
        for i in range(n):
            a = path[prev_idx(i, n)]
            b = path[next_idx(i, n)]

            epoch += 1
            # Go code iterates cand[a] and cand[b]
            nodes_to_check = set(cd.cand_list[a]) | set(cd.cand_list[b])

            for u in nodes_to_check:
                if visit_mark[u] == epoch:
                    continue
                visit_mark[u] = epoch
                if in_sel[u]:
                    continue

                # Check if move introduces a candidate edge
                if not (is_candidate_edge(cd, a, u) or is_candidate_edge(cd, b, u)):
                    continue

                dl = delta_exchange_selected(D, costs, path, i, u)
                if dl < best_delta:
                    ii, uu = i, u
                    best_delta = dl
                    # Need to capture v_old in the closure
                    v_to_remove = path[ii]
                    best_move = (lambda ii=ii, uu=uu, v_old_cap=v_to_remove: (
                        apply_exchange_selected(path, ii, uu),
                        set_in_sel(in_sel, v_old_cap, False), set_in_sel(in_sel, uu, True),
                        set_pos_of(pos_of, v_old_cap, -1), set_pos_of(pos_of, uu, ii)
                    ))

        if best_delta < 0 and best_move is not None:
            best_move()
        else:
            break

    return Solution(Path=path, Objective=calculate_objective(path, D, costs))


# Helper functions for lambda closures in candidates search
def set_in_sel(in_sel_list: List[bool], index: int, value: bool):
    if 0 <= index < len(in_sel_list):
        in_sel_list[index] = value


def set_pos_of(pos_of_list: List[int], index: int, value: int):
    if 0 <= index < len(pos_of_list):
        pos_of_list[index] = value


# --- From lm.go ---
class MoveType(Enum):
    MoveTwoOpt = 0
    MoveExchangeSelected = 1


# Use tuples for hashable keys
EdgeKey = Tuple[int, int]
MoveKey = Tuple[EdgeKey, EdgeKey]


@dataclass
class MoveRecord:
    kind: MoveType
    a: int
    b: int
    c: int
    d: int
    v: int
    u: int
    delta: int
    key: Optional[MoveKey] = None
    # For 2-opt, store cut indices in temp fields
    temp_cut1: int = -1
    temp_cut2: int = -1


class LmState:
    def __init__(self, n: int, dim: int):
        self.moves: List[MoveRecord] = []
        self.index: Dict[MoveKey, int] = {}  # Maps key to index in self.moves

        # Quick lookup structures
        self.pos_of = [-1] * dim
        self.in_sel = [False] * dim
        self.non_sel: List[int] = []

    def _canonical_edge(self, x: int, y: int) -> EdgeKey:
        return (x, y) if x < y else (y, x)

    def _canonical_move_key(self, e1: EdgeKey, e2: EdgeKey) -> MoveKey:
        # Sort keys to make the move key canonical
        return (e1, e2) if e1 < e2 else (e2, e1)

    def add_move(self, rec: MoveRecord):
        e1 = self._canonical_edge(rec.a, rec.b)
        e2 = self._canonical_edge(rec.c, rec.d)
        key = self._canonical_move_key(e1, e2)

        if key in self.index:
            return  # Move already exists

        rec.key = key
        self.moves.append(rec)
        self.index[key] = len(self.moves) - 1

    def remove_move(self, rec: MoveRecord):
        if not self.moves or rec.key is None:
            return

        key = rec.key
        idx = self.index.get(key)

        if idx is None:
            return  # Move not in index

        last_idx = len(self.moves) - 1

        if idx != last_idx:
            # Swap with last and update index
            last_rec = self.moves[last_idx]
            self.moves[idx] = last_rec
            if last_rec.key:
                self.index[last_rec.key] = idx

        # Pop the last element
        self.moves.pop()
        del self.index[key]

    def find_edge_cut(self, path: List[int], x: int, y: int) -> Tuple[bool, bool, int]:
        n = len(path)
        px = self.pos_of[x]
        py = self.pos_of[y]

        if px < 0 and py < 0:
            return False, False, 0

        if px >= 0:
            if path[next_idx(px, n)] == y:
                return True, True, px  # edge x->y, cut at x
            if path[prev_idx(px, n)] == y and py >= 0:
                return True, False, py  # edge y->x, cut at y

        if py >= 0:
            if path[next_idx(py, n)] == x:
                return True, False, py  # edge y->x, cut at y
            if path[prev_idx(py, n)] == x and px >= 0:
                return True, True, px  # edge x->y, cut at x

        return False, False, 0


def build_full_neighborhood_LM(D: List[List[int]], costs: List[int], path: List[int], non_sel: List[int], lm: LmState):
    n = len(path)

    # 1. Intra: 2-opt
    for i in range(n):
        for j in range(i + 1, n):  # Go: j := i + 1
            dl = delta_two_opt(D, path, i, j)
            if dl >= 0:
                continue

            a = path[i]
            b = path[next_idx(i, n)]
            c = path[j]
            d = path[next_idx(j, n)]
            rec = MoveRecord(
                kind=MoveType.MoveTwoOpt, a=a, b=b, c=c, d=d, v=-1, u=-1, delta=dl
            )
            lm.add_move(rec)

    # 2. Inter: exchange
    for i in range(n):
        for u in non_sel:
            dl = delta_exchange_selected(D, costs, path, i, u)
            if dl >= 0:
                continue

            a = path[prev_idx(i, n)]
            v = path[i]
            b = path[next_idx(i, n)]
            rec = MoveRecord(
                kind=MoveType.MoveExchangeSelected, a=a, b=v, c=v, d=b, v=v, u=u, delta=dl
            )
            lm.add_move(rec)


def update_LMAfter_move(D: List[List[int]], costs: List[int], path: List[int], lm: LmState, best_move: MoveRecord):
    n = len(path)
    if n == 0:
        return

    edge_starts: Set[int] = set()

    if best_move.kind == MoveType.MoveTwoOpt:
        i = best_move.temp_cut1
        j = best_move.temp_cut2
        if i < 0 or j < 0: return  # Should not happen
        if i > j: i, j = j, i
        # Go: for k := i; k <= j; k++
        # The segment reversed is from i+1 to j
        for k_idx in range(i, j + 1):
            k = k_idx % n
            edge_starts.add(k)  # edge (k, k+1)
            edge_starts.add(prev_idx(k, n))  # edge (k-1, k)

    elif best_move.kind == MoveType.MoveExchangeSelected:
        u_new = best_move.u
        if u_new < 0: return
        pos = lm.pos_of[u_new]
        if pos < 0 or pos >= n: return
        edge_starts.add(pos)  # edge (pos, pos+1)
        edge_starts.add(prev_idx(pos, n))  # edge (pos-1, pos)

    if not edge_starts:
        return

    # 1. 2-opt moves touching affected edges
    for i in edge_starts:
        for j in range(n):
            if j == i or j == next_idx(i, n) or j == prev_idx(i, n):
                continue

            dl = delta_two_opt(D, path, i, j)
            if dl >= 0:
                continue

            a = path[i]
            b = path[next_idx(i, n)]
            c = path[j]
            d = path[next_idx(j, n)]
            rec = MoveRecord(
                kind=MoveType.MoveTwoOpt, a=a, b=b, c=c, d=d, v=-1, u=-1, delta=dl
            )
            lm.add_move(rec)

    # 2. Exchange moves for positions adjacent to affected edges
    for i in edge_starts:
        for u in lm.non_sel:
            dl = delta_exchange_selected(D, costs, path, i, u)
            if dl >= 0:
                continue

            a = path[prev_idx(i, n)]
            v = path[i]
            b = path[next_idx(i, n)]
            rec = MoveRecord(
                kind=MoveType.MoveExchangeSelected, a=a, b=v, c=v, d=b, v=v, u=u, delta=dl
            )
            lm.add_move(rec)


def local_search_steepest_LM(D: List[List[int]], costs: List[int], init: Solution) -> Solution:
    path = init.Path[:]
    n = len(path)
    if n == 0:
        return init

    dim = len(D)
    lm = LmState(n, dim)

    # Init lookup structures
    for i, v in enumerate(path):
        lm.pos_of[v] = i
        lm.in_sel[v] = True
    for u in range(dim):
        if not lm.in_sel[u]:
            lm.non_sel.append(u)

    # Build full neighborhood once for initial solution
    build_full_neighborhood_LM(D, costs, path, lm.non_sel, lm)

    while True:
        best_delta = 0
        best_move_record: Optional[MoveRecord] = None
        has_best = False

        # 1. Browse LM
        idx = 0
        while idx < len(lm.moves):
            rec = lm.moves[idx]
            removed = False

            if rec.kind == MoveType.MoveTwoOpt:
                ok1, fwd1, cut1 = lm.find_edge_cut(path, rec.a, rec.b)
                ok2, fwd2, cut2 = lm.find_edge_cut(path, rec.c, rec.d)

                if not ok1 or not ok2:
                    lm.remove_move(rec)
                    removed = True
                elif fwd1 != fwd2:
                    # Stale delta, remove
                    lm.remove_move(rec)
                    removed = True
                else:
                    # Applicable
                    if rec.delta < best_delta or not has_best:
                        has_best = True
                        best_delta = rec.delta
                        best_move_record = rec
                        rec.temp_cut1 = cut1  # Store current indices
                        rec.temp_cut2 = cut2

            elif rec.kind == MoveType.MoveExchangeSelected:
                if rec.v < 0 or rec.v >= dim or lm.pos_of[rec.v] == -1:
                    lm.remove_move(rec)
                    removed = True
                elif rec.u >= 0 and rec.u < dim and lm.in_sel[rec.u]:
                    lm.remove_move(rec)
                    removed = True
                else:
                    ok1, fwd1, _ = lm.find_edge_cut(path, rec.a, rec.b)
                    ok2, fwd2, _ = lm.find_edge_cut(path, rec.c, rec.d)

                    if not ok1 or not ok2:
                        lm.remove_move(rec)
                        removed = True
                    elif fwd1 != fwd2:
                        lm.remove_move(rec)
                        removed = True
                    elif rec.delta < best_delta or not has_best:
                        has_best = True
                        best_delta = rec.delta
                        best_move_record = rec

            if not removed:
                idx += 1

        # 2. Check for termination
        if not has_best or best_delta >= 0:
            break

        # 3. Apply the best move
        if best_move_record is None:  # Should not happen if has_best is True
            print("Error: has_best is True but best_move_record is None", file=sys.stderr)
            break

        best_move = best_move_record  # type: MoveRecord

        if best_move.kind == MoveType.MoveTwoOpt:
            i = best_move.temp_cut1
            j = best_move.temp_cut2
            apply_two_opt_and_update_pos(path, lm.pos_of, i, j)

        elif best_move.kind == MoveType.MoveExchangeSelected:
            pos_v = lm.pos_of[best_move.v]
            if pos_v >= 0:
                v_old = best_move.v
                u_new = best_move.u
                apply_exchange_selected(path, pos_v, u_new)

                lm.in_sel[v_old], lm.in_sel[u_new] = False, True
                lm.pos_of[v_old], lm.pos_of[u_new] = -1, pos_v

                # Update non_sel list
                try:
                    lm.non_sel.remove(u_new)
                except ValueError:
                    # This can happen if u_new wasn't in non_sel, ignore
                    pass
                lm.non_sel.append(v_old)

        # Remove applied move
        lm.remove_move(best_move)

        # Incrementally update LM
        update_LMAfter_move(D, costs, path, lm, best_move)

    return Solution(Path=path, Objective=calculate_objective(path, D, costs))


# --- Worker function for parallel execution (Adapted from your project) ---

def _local_search_worker(
        D: List[List[int]],
        costs: List[int],
        method: MethodSpec
) -> Tuple[Solution, float]:  # Return solution and duration
    """
    A top-level wrapper function for ProcessPoolExecutor.
    This is the Python equivalent of Go's `RunLocalSearchBatch`'s inner loop.
    It runs ONE local search and times it.
    """

    try:
        # 1. Get a random initial solution
        init = start_random(D, costs)

        start_time = time.perf_counter()
        sol: Solution

        # 2. Run the specified local search method
        if method.UseCand:
            K = method.CandK if method.CandK > 0 else 10
            cd = build_candidates(D, costs, K)
            sol = local_search_steepest_candidates(D, costs, init, cd)
        elif method.UseLM:
            sol = local_search_steepest_LM(D, costs, init)
        else:
            sol = local_search_steepest_baseline(D, costs, init)

        duration_sec = time.perf_counter() - start_time
        return sol, duration_sec

    except Exception as e:
        # CRITICAL FIX: Removed logging.error, use print to stderr
        # This prevents the deadlock
        print(f"Error in worker process for {method.Name}: {e}", file=sys.stderr)
        return Solution(Path=[], Objective=float('inf')), 0.0


def run_local_search_batch(
        D: List[List[int]],
        costs: List[int],
        method: MethodSpec,
        num_solutions: int
) -> Tuple[List[Solution], List[float]]:
    """
    Runs a batch of local searches in parallel using ProcessPoolExecutor.
    This is your project's original (and superior) parallel runner.
    """
    solutions = []
    durations_ms = []

    with ProcessPoolExecutor() as executor:
        worker_func = partial(
            _local_search_worker,
            D=D,
            costs=costs,
            method=method
        )

        # Create a list of futures
        futures = [executor.submit(worker_func) for _ in range(num_solutions)]

        for future in as_completed(futures):
            result, duration_sec = future.result()
            if result.Path:
                solutions.append(result)
                durations_ms.append(duration_sec * 1000)

    return solutions, durations_ms