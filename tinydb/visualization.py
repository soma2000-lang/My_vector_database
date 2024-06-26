from tinyhnsw import HNSWIndex
from tinyhnsw.hnsw import DEFAULT_CONFIG
from typing import Optional, Dict, Tuple

import math
import numpy
import random
import networkx
import matplotlib.pyplot as plt


Layout = Dict[int, Tuple[float, float]]
def visualize_hnsw_index(index: HNSWIndex, layout: Optional[Layout] = None) -> None:
    """
    Use this to visualize the different layers of HNSW graphs. The nodes
    maintain consistent locations between layers, and the layers are
    plotted next to each other.
    """
    _, axs = plt.subplots(1, len(index.layers), figsize=(len(index.layers) * 5, 5))
l   layout = layout or networkx.spring_layout(index.layers[0].G)
    node_color = ["r" if index.ep == node else "c" for node, _ in layout.items()]
    # Determine the global min and max coordinates for consistent axes
    all_x_values = [pos[0] for pos in layout.values()]
    all_y_values = [pos[1] for pos in layout.values()]
    min_x, max_x = min(all_x_values), max(all_x_values)
    min_y, max_y = min(all_y_values), max(all_y_values)
    
    for i, layer in enumerate(index.layers):
        graph = layer.G

        graph_layout = {k: v for k, v in layout.items() if k in graph}
        graph_node_color = [node_color[k] for k, _ in graph_layout.items()]
        networkx.draw(
            graph, graph_layout, ax=axs[i], node_size=25, node_color=graph_node_color
        )
        axs[i].set_title(f"Layer {i}")
        # Set consistent axes limits
        axs[i].set_xlim(min_x-1, max_x+1)
        axs[i].set_ylim(min_y-1, max_y+1)

    plt.show()



