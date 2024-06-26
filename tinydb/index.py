from __future__ import annotations

import numpy
import pickle

class Index:
    def __init__(self, d: int, distance: str = "cosine") -> None:
        self.ntotal = 0
        self.vectors = None
        self.is_trained = False
        self.d = d

        assert distance in ["cosine", "l2", "inner_product"]

        if distance == "cosine":
            self.f_distance = cosine_distance
        elif distance == "l2":
            self.f_distance = l2_distance
        elif distance == "inner_product":
            self.f_distance = inner_product_distance
