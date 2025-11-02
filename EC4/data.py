import csv
import math
import logging
from dataclasses import dataclass
from typing import List

@dataclass
class Node:
    """
    Represents a node in the graph, with coordinates and a cost/prize.
    MODIFIED: x and y are now integers.
    """
    id: int
    x: int  # Changed from float
    y: int  # Changed from float
    cost: int


def euclidean_distance(n1: Node, n2: Node) -> float:
    """
    Calculates the Euclidean distance between two nodes.
    """
    return math.hypot(n1.x - n2.x, n1.y - n2.y)


def calculate_distance_matrix(nodes: List[Node]) -> List[List[int]]:
    """
    Creates an N_x_N distance matrix, rounding Euclidean distances
    to the nearest integer.
    """
    n = len(nodes)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(nodes[i], nodes[j])
            rounded_dist = int(round(dist))
            D[i][j] = rounded_dist
            D[j][i] = rounded_dist

    return D

def read_nodes(file_path: str) -> List[Node]:
    """
    Reads node data from a CSV file.

    MODIFIED:
    - Assumes NO header row.
    - Assumes 3-column format: X;Y;Cost
    - X and Y are parsed as integers.
    - ID is generated dynamically (0, 1, 2...).
    """
    nodes: List[Node] = []

    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            # Specify the semicolon delimiter
            reader = csv.reader(file, delimiter=';')

            # Removed the line that skips the header

            # Read data rows
            # 'i' will be the 0-based index of the data row
            for i, row in enumerate(reader):
                # row_num is now i + 1 (since there's no header)
                row_num = i + 1

                # Skip empty lines
                if not row:
                    continue

                # Expecting 3 columns
                if len(row) < 3:
                    logging.warning(f"Skipping malformed row {row_num}: {row}")
                    continue

                try:
                    #Map 3 columns and generate ID
                    node = Node(
                        id=i,  # Dynamically assign ID (0, 1, 2...)
                        x=int(row[0]),  # Column 0 is X (as int)
                        y=int(row[1]),  # Column 1 is Y (as int)
                        cost=int(row[2])  # Column 2 is Cost
                    )
                    nodes.append(node)
                except ValueError as e:
                    # Handle parsing errors (e.g., "abc" instead of a number)
                    raise ValueError(f"Error parsing row {row_num}: {e}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at path: {file_path}")
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {e}")

    return nodes
