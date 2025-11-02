import csv
import math
import re
import random
from dataclasses import dataclass
from typing import List, Tuple

# We need the Solution class for type hinting in calculate_statistics
# This assumes it's in a file named algorithms.py
try:
    from algorithms import Solution
except ImportError:
    # Create a placeholder if algorithms.py isn't available yet
    # This allows the file to be read, but it will fail at runtime
    # if the real Solution class isn't defined.
    @dataclass
    class Solution:
        path: List[int]
        objective: int


@dataclass
class Row:
    """
    Data container for a single row of results in the final report.
    Equivalent to the Row struct in row.go.
    """
    name: str
    avg_v: float
    min_v: int
    max_v: int
    avg_t_ms: float
    best_path: List[int]
    best_value: int


_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?* ]')


def sanitize_file_name(filename: str) -> str:
    """
    Replaces unsafe filename characters with an underscore.
    Equivalent to SanitizeFileName in sanitizer.go.
    """
    return _UNSAFE_CHARS_RE.sub('_', filename)


# --- From stats.go ---

def calculate_statistics(solutions: List[Solution]) -> Tuple[int, int, float]:
    """
    Calculates the min, max, and average objective values from a list of solutions.
    Equivalent to CalculateStatistics in stats.go.
    """
    if not solutions:
        return 0, 0, 0.0

    min_val = solutions[0].objective
    max_val = solutions[0].objective
    total_val = 0.0

    for s in solutions:
        obj = s.objective
        if obj < min_val:
            min_val = obj
        if obj > max_val:
            max_val = obj
        total_val += obj

    avg_val = total_val / len(solutions)
    return min_val, max_val, avg_val


def generate_start_node_indices(n: int) -> List[int]:
    """
    Generates a list of n random start node indices (from 0 to n-1).
    Equivalent to GenerateStartNodeIndices in generate_start_node_indices.go.

    Note: This function is not used by the provided main.go,
    which uses generateRandomPath instead.
    """
    return [random.randint(0, n - 1) for _ in range(n)]


def write_results_csv(instance_name: str, rows: List[Row]):
    """
    Writes the final results to a CSV file.
    Equivalent to WriteResultsCSV in results_to_csv.go.
    """
    filename = f"results_{instance_name}.csv"

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write header row
            writer.writerow([
                "Name", "AvgValue", "MinValue", "MaxValue",
                "AvgTimeMs", "BestPath", "BestValue"
            ])

            # Write data rows
            for r in rows:
                # Format the path list as a string, e.g., "[1,2,3]"
                path_str = str(r.best_path)

                writer.writerow([
                    r.name,
                    f"{r.avg_v:.2f}",  # Format float to 2 decimal places
                    r.min_v,
                    r.max_v,
                    f"{r.avg_t_ms:.4f}",  # Format float to 4 decimal places
                    path_str,
                    r.best_value
                ])

    except IOError as e:
        # Re-raise as an IOError with more context
        raise IOError(f"Error writing CSV file {filename}: {e}")