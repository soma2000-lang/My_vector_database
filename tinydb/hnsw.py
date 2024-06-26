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
        
        
        for layer in range(L):
            _,W= self.layers[layer].search(q, ep, ef=1)
            ep = W[0]
        for layer in range(min(L, l), -1, -1):
            self.layers[layer].insert(q, ix, ep)
        if l > self.L:
            for l_new in range(L + 1, l + 1):
                self.layers.append(self.layer_factory(l_new, self.ix))
                self.L = l
                self.ep = ix
        def distance(self, q: numpy.ndarray, v: numpy.ndarray) -> numpy.ndarray:
            if len(q.shape) == 1:
            q = numpy.expand_dims(q, axis=0)

        if len(v.shape) == 1:
            v = numpy.expand_dims(v, axis=0)

        return self.f_distance(q, v)
    
        def search(self, q: numpy.ndarray, k: int) -> tuple[numpy.ndarray, numpy.ndarray]:
