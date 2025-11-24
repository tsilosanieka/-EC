import os
import csv
import logging
from dataclasses import dataclass
from typing import List, Tuple
from algorithms import Solution 

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

@dataclass
class Row:
    """
    Represents a row of data to be written to the CSV.
    """
    name: str
    avg_v: float
    min_v: int
    max_v: int
    avg_tms: float
    best_path: List[int]
    best_value: int

def sanitize_file_name(name: str) -> str:
    """
    Replaces spaces with underscores and removes parentheses and commas.
    Equivalent to Go's strings.NewReplacer.
    """
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

def calculate_statistics(solutions: List[Solution]) -> Tuple[int, int, float]:
    """
    Calculates min, max, and average objective values from a list of solutions.
    Returns: (min_obj, max_obj, avg_obj)
    """
    if not solutions:
        return 0, 0, 0.0

    # Extract all objective values
    objectives = [sol.objective for sol in solutions]

    min_obj = min(objectives)
    max_obj = max(objectives)
    avg_obj = sum(objectives) / len(objectives)

    return min_obj, max_obj, avg_obj

def generate_start_node_indices(n: int) -> List[int]:
    """
    Creates a list of starting node indices [0, 1, ..., n-1].
    """
    return list(range(n))

OUTPUT_DIR = os.path.join("output", "results")

def _ints_to_dash_string(nums: List[int]) -> str:
    """
    Helper to format list as string like "[1 2 3]".
    Note: Go implementation used spaces inside brackets, not commas.
    """
    if not nums:
        return ""
    # Join numbers with a space
    content = " ".join(map(str, nums))
    return f"[{content}]"

def write_results_csv(instance_name: str, rows: List[Row]) -> None:
    """
    Writes the processing results to a CSV file.
    Creates directories if they don't exist.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {OUTPUT_DIR}: {e}")
        raise

    filename = os.path.join(OUTPUT_DIR, f"results_instance_{instance_name}.csv")

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write Header
            header = [
                "instance",
                "method",
                "avg_objective",
                "av(min,max)",
                "min_objective",
                "max_objective",
                "avg_time_ms",
                "best_objective",
                "best_path",
            ]
            writer.writerow(header)

            # Write Rows
            for r in rows:
                avg_4 = f"{r.avg_v:.4f}"
                avg_summary = f"{r.avg_v:.4f} ({r.min_v}, {r.max_v})"
                avg_time = f"{r.avg_tms:.2f}"
                best_path_str = _ints_to_dash_string(r.best_path)

                rec = [
                    instance_name,
                    r.name,
                    avg_4,          # avg_objective
                    avg_summary,    # av(min,max)
                    str(r.min_v),
                    str(r.max_v),
                    avg_time,
                    str(r.best_value),
                    best_path_str,
                ]
                writer.writerow(rec)

        logging.info(f"CSV saved: {filename}")

    except IOError as e:
        logging.error(f"Failed to write CSV: {e}")
        raise
