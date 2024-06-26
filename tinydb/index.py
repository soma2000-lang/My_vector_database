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
    def add(self, vectors: numpy.ndarray) -> None:
            assert vectors.shape[1] == self.d

        if self.vectors is None:
            self.vectors = vectors
            self.is_trained = True
        else:
            self.vectors = numpy.append(self.vectors, vectors, axis=0)

        self.ntotal = self.vectors.shape[0]
    def search(
        self, query: numpy.ndarray, k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError()

    def save(self, file: str) -> None:
        with open(file, "wb") as f:
            pickle.dump(self, f)
        @classmethod
        def from_file(cls, file: str) -> Index:
            with open(file, "rb") as f:
                return pickle.load(f)
def _normalize(X: numpy.ndarray) -> numpy.ndarray:
    return X / numpy.expand_dims(numpy.linalg.norm(X, axis=1), axis=1)

