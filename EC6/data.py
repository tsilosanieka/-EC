import math
import csv
import logging
import os
from dataclasses import dataclass
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')


@dataclass
class Node:
    """
    Data class representing a Node with X, Y coordinates and a Cost.
    """
    x: int
    y: int
    cost: int


def calculate_distance_matrix(nodes: List[Node]) -> List[List[int]]:
    """
    Calculates a Euclidean distance matrix for the provided nodes.
    """
    n = len(nodes)
    # Initialize n x n matrix with zeros
    distance_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = nodes[i].x, nodes[i].y
                x2, y2 = nodes[j].x, nodes[j].y

                # Calculate Euclidean distance
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Round to nearest integer and cast to int
                distance_matrix[i][j] = int(round(distance))

    logging.info(f"Calculated distance matrix for {n} nodes")
    return distance_matrix


def read_nodes(filename: str) -> List[Node]:
    """
    Reads nodes from a CSV file using ';' as a delimiter.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    nodes = []

    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')

        for row_index, row in enumerate(reader):
            # Skip empty lines if any
            if not row:
                continue

            try:
                x = int(row[0])
                y = int(row[1])
                cost = int(row[2])
                nodes.append(Node(x, y, cost))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing line {row_index + 1}: {e}")

    logging.info(f"Read {len(nodes)} nodes from {filename}")
    return nodes
