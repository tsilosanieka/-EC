import os
import logging
from typing import List

import matplotlib.pyplot as plt

try:
    from data import Node
except ImportError:
    from dataclasses import dataclass


    @dataclass
    class Node:
        id: int
        x: int
        y: int
        cost: int


def plot_solution(
        nodes: List[Node],
        path: List[int],
        title: str,
        file_name: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float
):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title(title, fontsize=16)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    path_nodes = [nodes[i] for i in path]
    path_x = [n.x for n in path_nodes]
    path_y = [n.y for n in path_nodes]

    if path_x:
        path_x.append(path_nodes[0].x)
        path_y.append(path_nodes[0].y)

    path_set = set(path)
    used_x, used_y = [], []
    unused_x, unused_y = [], []

    for node in nodes:
        if node.id in path_set:
            used_x.append(node.x)
            used_y.append(node.y)
        else:
            unused_x.append(node.x)
            unused_y.append(node.y)

    ax.scatter(unused_x, unused_y, c='gray', s=15, label='Unused Nodes', alpha=0.7)
    ax.scatter(used_x, used_y, c='green', s=30, label='Used Nodes', zorder=5)
    ax.plot(path_x, path_y, 'b-', label='Solution Path', zorder=4)

    for node in nodes:
        ax.text(
            node.x + 0.5,
            node.y + 0.5,
            str(node.id),
            fontsize=8,
            ha='left'
        )

    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    output_dir = "./charts"
    output_filename = f"{output_dir}/{file_name}.png"

    try:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')

    except IOError as e:
        logging.error(f"Failed to save plot {output_filename}: {e}")
    finally:
        plt.close(fig)
