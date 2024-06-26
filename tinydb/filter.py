from __future__ import annotations
from tinyhnsw.hnsw import HNSWIndex, HNSWLayer
from heapq import heappop, heappush, nlargest, nsmallest

import numpy
import random

class FilterableHNSWIndex(HNSWIndex):
    """
    A FilterableHNSWIndex implements the "missing WHERE clause" that Pinecone talks about
    in their blog. It allows you to filter which nodes get added to the nearest neighbor
    list by passing an allow-list to the index.

    If no allow-list is passed, then the behavior defaults to the standard HNSWIndex.
    We can make this change pretty non-invasively through the use of a layer_factory
    function.
    """
    def layer_factory(self, lc: int, ep: int | None = None) -> FilterableHNSWLayer:
        ep = ep or self.ep
        return FilterableHNSWLayer(self, lc, ep)

    def search(
        self, q: numpy.ndarray, k: int, valid: list[int] | None = None
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ep = self.ep
        for lc in range(self.L, 0, -1):
            ep = self.layers[lc].search(q, ep, 1)[1][0]

        return self.layers[0].search(q, ep, k, valid=valid)
class FilterableHNSWLayer(HNSWLayer):
        """
    The allow-list works by ensuring that only valid neighbors are added to the
    candidate neighbor list (W).
    """
    def search(
            self, q: numpy.ndarray, ep: int, ef: int, valid: list[int] | None = None
        ) -> tuple[list[float], list[int]]:
            ep_dist = self.distance_to_node(q, ep)
            valid_set = set(valid)
            
        v = {ep}
        C = [(ep_dist, ep)]
        # this addresses the issue of not considering the ep if it's not a valid node:
        W = [] if (valid and ep not in valid_set) else [(ep_dist, ep)]