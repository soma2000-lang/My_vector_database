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