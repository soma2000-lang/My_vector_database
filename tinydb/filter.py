from __future__ import annotations
from tinyhnsw.hnsw import HNSWIndex, HNSWLayer
from heapq import heappop, heappush, nlargest, nsmallest

import numpy
import random