from __future__ import annotations
from tinyhnsw.index import Index
from tinyhnsw.utils import load_sift, evaluate
from dataclasses import dataclass
from heapq import nlargest, nsmallest, heappop, heappush, heapify
from tqdm import tqdm


@dataclass
class HNSWConfig:
    M: int
    M_max: int
    M_max0: int
    m_L: float
    ef_construction: int
    ef_search: int

    neighbors: str = "simple"
    extend_candidates: bool = False
    keep_pruned_connections: bool = True


DEFAULT_CONFIG = HNSWConfig(
    M=16,
    M_max=16,
    M_max0=32,
    m_L=(1.0 / math.log(16)),
    ef_construction=32,
    ef_search=32,
)


class HNSWIndex(Index):
    def __init__(
        self, d: int, distance: str = "cosine", config: HNSWConfig = DEFAULT_CONFIG
    ) -> None:
        super().__init__(d, distance)

        self.config = config
        self.vectors = None

        self.ep = 0
        self.L = 0
        self.ix = 0
        self.layers = [self.layer_factory(0, self.ep)]
