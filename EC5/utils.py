import csv
import math
import sys
import os
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

# We need the Solution class for type hinting
# This assumes it's in a file named algorithms.py
try:
    # Use the Go-style capitalization
    from algorithms import Solution
except ImportError:
    # Create a placeholder if algorithms.py isn't available yet
    @dataclass
    class Solution:
        Path: List[int]
        Objective: int


# --- From row.go ---
@dataclass
class Row:
    """ Represents a single row of aggregated experiment results. """
    Name: str
    AvgV: float
    MinV: int
    MaxV: int
    AvgTms: float
    BestPath: List[int]
    BestValue: int


# --- From sanitizer.go ---
def sanitize_file_name(name: str) -> str:
    """
    Normalises a string so it can be safely used as a filename.
    Matches the Go code's replacement logic.
    """
    # Replaces " ", "(", ")", and ","
    name = name.replace(" ", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(",", "")
    return name


# --- From stats.go ---
def calculate_statistics(solutions: List[Solution]) -> Tuple[int, int, float]:
    """
    Calculates the minimum, maximum and average objective
    value across all provided solutions.
    """
    if not solutions:
        return 0, 0, 0.0

    # Use sys.maxsize for MaxInt32
    min_obj = sys.maxsize
    max_obj = -sys.maxsize  # Go init to 0, but -inf is safer for minimization
    sum_val = 0.0

    for sol in solutions:
        obj = sol.Objective
        if obj < min_obj:
            min_obj = obj
        if obj > max_obj:
            max_obj = obj
        sum_val += obj

    avg = sum_val / len(solutions)
    return min_obj, max_obj, avg


# --- From generate_start_node_indices.go ---
def generate_start_node_indices(n: int) -> List[int]:
    """
    Creates a list of starting node indices [0, 1, ..., n-1].
    Note: This is not used by the 'start_random' function.
    """
    return list(range(n))


# --- From results_to_csv.go ---
OUTPUT_DIR = "output/results"


def _ints_to_dash_string(nums: List[int]) -> str:
    """
    Formats a slice of ints as a space-separated list enclosed
    in square brackets, e.g. [1 2 3].
    """
    if not nums:
        return "[]"  # Match Go's behavior for empty
    # More Pythonic way to build the string
    content = " ".join(map(str, nums))
    return f"[{content}]"


def write_results_csv(instance_name: str, rows: List[Row]):
    """
    Writes aggregated experiment results for a single instance
    to a CSV file under output/results.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {OUTPUT_DIR}: {e}", file=sys.stderr)
        return

    filename = os.path.join(
        OUTPUT_DIR,
        f"results_instance_{instance_name}.csv"
    )

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)

            # Write header (matches the Go file exactly)
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
            w.writerow(header)

            # Write data rows
            for r in rows:
                avg_str = f"{r.AvgV:.4f}"
                avg_summary = f"{r.AvgV:.4f} ({r.MinV}, {r.MaxV})"

                rec = [
                    instance_name,
                    r.Name,
                    avg_str,
                    avg_summary,
                    str(r.MinV),
                    str(r.MaxV),
                    f"{r.AvgTms:.2f}",
                    str(r.BestValue),
                    _ints_to_dash_string(r.BestPath),
                ]
                w.writerow(rec)

        # Use logging for info, as this runs in the main process (it's safe)
        logging.info(f"CSV saved: {filename}")

    except IOError as e:
        print(f"Error writing CSV file {filename}: {e}", file=sys.stderr)
    except csv.Error as e:
        print(f"Error writing CSV data: {e}", file=sys.stderr)