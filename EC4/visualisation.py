import os
import logging
from typing import List

#standard plotting library for Python
import matplotlib.pyplot as plt

# We need the Node dataclass from our data.py file
try:
    from data import Node
except ImportError:
    # Placeholder if data.py is not in the same directory
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
    """
    Plots a solution path, highlighting used and unused nodes.
    Saves the result to a PNG file.
    """

    # 1. Create a new plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # 2. Set title and axis limits
    ax.set_title(title, fontsize=16)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --- 3. Prepare data ---

    # a) Data for the solution path (for the blue line)
    # Get the actual Node objects for the path
    # This works because data.py assigns id=i, so node.id == its index
    path_nodes = [nodes[i] for i in path]
    path_x = [n.x for n in path_nodes]
    path_y = [n.y for n in path_nodes]

    # Add the starting node to the end to close the loop
    if path_x:  # Only if path is not empty
        path_x.append(path_nodes[0].x)
        path_y.append(path_nodes[0].y)

    # b) Data for scatter plots (Used vs. Unused)
    path_set = set(path)  # Use a set for fast O(1) lookups
    used_x, used_y = [], []
    unused_x, unused_y = [], []

    # Loop over the FULL list of nodes
    for node in nodes:
        if node.id in path_set:
            # This node is in the solution
            used_x.append(node.x)
            used_y.append(node.y)
        else:
            # This node is NOT in the solution
            unused_x.append(node.x)
            unused_y.append(node.y)

    # --- 4. Create plot layers ---

    # a) Plot Unused Nodes (gray, small, transparent)
    ax.scatter(unused_x, unused_y, c='gray', s=15, label='Unused Nodes', alpha=0.7)

    # b) Plot Used Nodes (green, larger, on top)
    ax.scatter(used_x, used_y, c='red', s=30, label='Used Nodes', zorder=5)

    # c) Plot the solution path as a blue line
    ax.plot(path_x, path_y, 'b-', label='Solution Path', zorder=4)

    # d) Add labels for each node
    for node in nodes:
        ax.text(
            node.x + 0.5,
            node.y + 0.5,
            str(node.id),
            fontsize=8,
            ha='left'
        )

    # Add a grid and set aspect ratio to 'equal'
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    # --- 5. Save the file ---

    output_dir = "./charts"
    output_filename = f"{output_dir}/{file_name}.png"

    try:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')

    except IOError as e:
        logging.error(f"Failed to save plot {output_filename}: {e}")
    finally:
        # IMPORTANT: Close the plot to free memory
        plt.close(fig)
