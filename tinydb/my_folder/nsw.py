import numpy as np

from tinydb.utils import evaluate, load_sift
from tinydb.knn import cosine_similarity


class NSWIndex:
    def __init__(self):
        self.graph = {}  # Dictionary to store the graph
        self.data = []  # List to store the actual data items

    def add_item(self, item, f=10, w=5):
        # Add item to data list
        self.data.append(item)
        idx = len(self.data) - 1

