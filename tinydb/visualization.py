from tinyhnsw import HNSWIndex
from tinyhnsw.hnsw import DEFAULT_CONFIG
from typing import Optional, Dict, Tuple

import math
import numpy
import random
import networkx
import matplotlib.pyplot as plt


Layout = Dict[int, Tuple[float, float]]
