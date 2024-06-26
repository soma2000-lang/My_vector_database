from __future__ import annotations
from tinyhnsw.index import Index
from tinyhnsw.utils import load_sift, evaluate
from dataclasses import dataclass
from heapq import nlargest, nsmallest, heappop, heappush, heapify
from tqdm import tqdm


