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
