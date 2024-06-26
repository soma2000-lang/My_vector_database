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
