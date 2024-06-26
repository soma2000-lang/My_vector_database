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
            ep = self.ep
            ef = max(k, self.config.ef_search)
            for lc in range(self.L, 0, -1):
                ep = self.layers[lc].search(q, ep, 1)[1][0]

            W = list(zip(*self.layers[0].search(q, ep, ef)))
            neighbors = nsmallest(k, W, lambda x: x[0])
            return list(zip(*neighbors))
        class HNSWLayer:
            def __init__(self, index: HNSWIndex, lc: int, ep: int | None = None) -> None:
                self.G = networkx.Graph()
                self.index = index
                self.config = self.index.config

                if ep is not None:
                    self.G.add_node(ep)

                if lc == 0:
                    self.M_max = self.config.M_max0
                else:
                    self.M_max = self.config.M_max

                if self.config.neighbors == "simple":
                    self.f_neighbors = self.select_neighbors
                else:
                    self.f_neighbors = self.select_neighbors_heuristic
            def search(
                self, q: numpy.ndarray, ep: int, ef: int
            ) -> tuple[list[float], list[int]]:
                ep_dist = self.distance_to_node(q, ep)
                v = {ep}
                C = [(ep_dist, ep)]
                W = [(ep_dist, ep)]

                while len(C) > 0:
                    d_c, c = heappop(C)
                    d_f, f = nlargest(1, W, key=lambda x: x[0])[0]

                    if d_c > d_f:
                        break

                    for e in self.G[c]:
                        if e in v:
                            continue

                        v.add(e)
                        d_f, f = nlargest(1, W, key=lambda x: x[0])[0]
                        d_e = self.distance_to_node(q, e)

                        if d_e < d_f or len(W) < ef:
                            heappush(C, (d_e, e))
                            heappush(W, (d_e, e))

                            if len(W) > ef:
                                W = nsmallest(ef, W, key=lambda x: x[0])

                return tuple(zip(*W))