import numpy as np

from tinydb.utils import evaluate, load_sift
from tinydb.knn import cosine_similarity


class NSWIndex:
    def __init__(self):
        self.graph = {}  
        self.data = []  
        
    def add_item(self, item, f=10, w=5):
        
        self.data.append(item)
        idx = len(self.data) - 1
        if idx == 0:
            self.graph[idx] = set()
            return

        # Perform k-NN search to find nearest neighbors
        neighbors = self.k_nn_search(item, w, f)

        # Connect the new item with its neighbors
        self.graph[idx] = set(neighbors)
        for neighbor in neighbors:
            self.graph[neighbor].add(idx)
        if idx == 0:
                self.graph[idx] = set()
            return

        # Perform k-NN search to find nearest neighbors
        neighbors = self.k_nn_search(item, w, f)

        # Connect the new item with its neighbors
        self.graph[idx] = set(neighbors)
        for neighbor in neighbors:
            self.graph[neighbor].add(idx)
            
        def k_nn_search(self, query, m, k):
            temp_res = set()
            candidates = set()
            visited_set = set()
            result = set()
            for i in range(m):
                candidates.add(self.get_random_entry_point())
            temp_res.clear()

            while candidates:
                c = min(candidates, key=lambda x: self.distance(query, self.data[x]))
                candidates.remove(c)

                if c in visited_set or (
                    result
                    and self.distance(query, self.data[c])
                    > self.distance(
                        query,
                        self.data[
                            max(
                                result, key=lambda x: self.distance(query, self.data[x])
                            )
                        ],
                    )
                ):

                 break

                visited_set.add(c)
                temp_res.add(c)
                for e in self.graph.get(c, []):
                    if e not in visited_set:
                        visited_set.add(e)
                        candidates.add(e)
                        temp_res.add(e)

            result.update(temp_res)

        return sorted(result, key=lambda x: self.distance(query, self.data[x]))[:k]
