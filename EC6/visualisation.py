import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from typing import List
from data import Node

OUTPUT_DIR = os.path.join("output", "plots")


def plot_solution(nodes: List[Node], path: List[int], title: str, filename: str,
                  x_min: float, x_max: float, y_min: float, y_max: float):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {OUTPUT_DIR}: {e}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---------------------------------------------------------
    # 1. Prepare Data
    # ---------------------------------------------------------
    costs = [n.cost for n in nodes]
    min_cost, max_cost = min(costs), max(costs)

    x_coords = [n.x for n in nodes]
    y_coords = [n.y for n in nodes]

    # ---------------------------------------------------------
    # 2. Node Styling (Match Go: Light Blue -> Dark Blue)
    # ---------------------------------------------------------
    cmap = mcolors.LinearSegmentedColormap.from_list("cost_blue", ["#C6DBEF", "#084594"])

    norm = plt.Normalize(min_cost, max_cost)

    range_cost = max_cost - min_cost if max_cost > min_cost else 1
    sizes = [(3.5 + (n.cost - min_cost) / range_cost * 3.0) ** 2 * 3 for n in nodes]

    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------

    # A. Draw the Path Line
    path_x = []
    path_y = []
    for idx in path:
        path_x.append(nodes[idx].x)
        path_y.append(nodes[idx].y)

    # Close the loop if path exists
    if path:
        path_x.append(nodes[path[0]].x)
        path_y.append(nodes[path[0]].y)

    ax.plot(path_x, path_y, color='blue', linewidth=1.5, alpha=1.0, zorder=1)

    ax.scatter(x_coords, y_coords, c=costs, cmap=cmap, norm=norm, s=sizes, zorder=2, edgecolors='none')

    if path:
        path_node_x = [nodes[i].x for i in path]
        path_node_y = [nodes[i].y for i in path]
        ax.scatter(path_node_x, path_node_y, c='black', marker='+', s=40, linewidth=1, zorder=3)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Add arrow-like look to axes ends
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # Set Limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Enforce square aspect ratio
    ax.set_aspect('equal')

    # Title slightly above
    ax.set_title(title, y=1.02, fontsize=12)

    # Save
    full_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {full_path}")
