"""
Tests for the HNSWIndex (Hierarchical Navigable Small World).

HNSWIndex(d) implements approximate nearest neighbor search via a
multi-layer navigable small world graph. It exposes the same interface
as FullNNIndex but trades exact correctness for speed.

API:
    index = HNSWIndex(d)         # d = vector dimensionality
    index.add(vectors)           # vectors: np.ndarray (n, d)
    D, I = index.search(q, k)   # q: (nq, d) → D: distances, I: indices
    index.ntotal                 # number of indexed vectors

Run: pytest tests/test_hnsw.py -v
"""

import pytest
import numpy as np

from tinyhnsw import HNSWIndex, FullNNIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_hnsw():
    """50 vectors, dim=8, seeded for reproducibility."""
    np.random.seed(42)
    index = HNSWIndex(d=8)
    data = np.random.randn(50, 8).astype(np.float32)
    index.add(data)
    return index, data


@pytest.fixture
def medium_hnsw():
    """500 vectors, dim=32."""
    np.random.seed(0)
    index = HNSWIndex(d=32)
    data = np.random.randn(500, 32).astype(np.float32)
    index.add(data)
    return index, data


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestHNSWConstruction:

    def test_empty_index(self):
        index = HNSWIndex(d=16)
        assert index.ntotal == 0

    def test_add_sets_ntotal(self, small_hnsw):
        index, _ = small_hnsw
        assert index.ntotal == 50

    def test_add_single_vector(self):
        index = HNSWIndex(d=4)
        vec = np.random.randn(1, 4).astype(np.float32)
        index.add(vec)
        assert index.ntotal == 1

    def test_add_batch(self):
        index = HNSWIndex(d=10)
        data = np.random.randn(100, 10).astype(np.float32)
        index.add(data)
        assert index.ntotal == 100

    def test_dimension_preserved(self):
        index = HNSWIndex(d=64)
        data = np.random.randn(20, 64).astype(np.float32)
        index.add(data)
        assert index.ntotal == 20


# ---------------------------------------------------------------------------
# Search — Shape & Validity
# ---------------------------------------------------------------------------

class TestHNSWSearchShape:

    def test_search_returns_correct_shapes(self, small_hnsw):
        index, _ = small_hnsw
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=10)
        assert D.shape == (5, 10)
        assert I.shape == (5, 10)

    def test_search_k1(self, small_hnsw):
        index, data = small_hnsw
        D, I = index.search(data[:3], k=1)
        assert I.shape == (3, 1)

    def test_search_indices_valid(self, small_hnsw):
        index, _ = small_hnsw
        queries = np.random.randn(10, 8).astype(np.float32)
        D, I = index.search(queries, k=5)
        assert np.all(I >= 0)
        assert np.all(I < index.ntotal)

    def test_search_distances_nonnegative(self, small_hnsw):
        index, _ = small_hnsw
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=5)
        assert np.all(D >= 0)

    def test_search_distances_sorted(self, small_hnsw):
        """Distances should be returned in ascending order."""
        index, _ = small_hnsw
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=10)
        for i in range(D.shape[0]):
            dists = D[i].tolist()
            assert dists == sorted(dists)

    def test_search_indices_unique_per_query(self, small_hnsw):
        index, _ = small_hnsw
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=10)
        for i in range(I.shape[0]):
            assert len(set(I[i])) == I.shape[1]


# ---------------------------------------------------------------------------
# Search — Self-Retrieval & Recall
# ---------------------------------------------------------------------------

class TestHNSWSelfRetrieval:

    def test_self_retrieval_k1(self, small_hnsw):
        """Each indexed vector should find itself as the nearest neighbor."""
        index, data = small_hnsw
        D, I = index.search(data, k=1)
        recall = np.mean(I[:, 0] == np.arange(len(data)))
        # HNSW is approximate, but self-retrieval should be very high
        assert recall >= 0.90

    def test_self_distance_near_zero(self, small_hnsw):
        """Distance to self should be near zero."""
        index, data = small_hnsw
        D, I = index.search(data, k=1)
        # Only check rows where self was actually retrieved
        self_mask = I[:, 0] == np.arange(len(data))
        if self_mask.any():
            assert np.all(D[self_mask, 0] < 1e-4)


# ---------------------------------------------------------------------------
# Recall Against Exact NN
# ---------------------------------------------------------------------------

class TestHNSWRecall:

    def test_recall_at_1_on_random_data(self, medium_hnsw):
        """HNSW recall@1 should be high compared to exact search."""
        index, data = medium_hnsw
        np.random.seed(99)
        queries = np.random.randn(50, 32).astype(np.float32)

        # Exact search
        exact = FullNNIndex(32)
        exact.add(data)
        D_exact, I_exact = exact.search(queries, k=1)

        # HNSW search
        D_hnsw, I_hnsw = index.search(queries, k=1)

        recall = np.mean(I_hnsw[:, 0] == I_exact[:, 0])
        assert recall >= 0.80, f"Recall@1 too low: {recall:.2f}"

    def test_recall_at_10_on_random_data(self, medium_hnsw):
        """HNSW recall@10 (intersection of top-10 results)."""
        index, data = medium_hnsw
        np.random.seed(99)
        queries = np.random.randn(20, 32).astype(np.float32)

        exact = FullNNIndex(32)
        exact.add(data)
        D_exact, I_exact = exact.search(queries, k=10)
        D_hnsw, I_hnsw = index.search(queries, k=10)

        recalls = []
        for i in range(len(queries)):
            overlap = len(set(I_exact[i]) & set(I_hnsw[i]))
            recalls.append(overlap / 10.0)
        avg_recall = np.mean(recalls)
        assert avg_recall >= 0.70, f"Recall@10 too low: {avg_recall:.2f}"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestHNSWEdgeCases:

    def test_two_vectors(self):
        """Minimal index with 2 vectors."""
        index = HNSWIndex(d=3)
        data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        index.add(data)
        assert index.ntotal == 2

        query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        D, I = index.search(query, k=1)
        assert I[0, 0] == 0

    def test_identical_vectors(self):
        """All identical vectors — should not crash."""
        index = HNSWIndex(d=4)
        data = np.ones((20, 4), dtype=np.float32)
        index.add(data)
        assert index.ntotal == 20

        D, I = index.search(data[:1], k=5)
        assert D.shape == (1, 5)
        # All distances should be 0 (or near 0)
        assert np.allclose(D[0], 0.0, atol=1e-5)

    def test_high_dimensional(self):
        """128-dim vectors (SIFT-like dimensionality)."""
        np.random.seed(7)
        index = HNSWIndex(d=128)
        data = np.random.randn(100, 128).astype(np.float32)
        index.add(data)
        assert index.ntotal == 100

        D, I = index.search(data[:5], k=1)
        # At least some should retrieve themselves
        self_hits = np.sum(I[:, 0] == np.arange(5))
        assert self_hits >= 3

    def test_single_query_vector(self, small_hnsw):
        index, data = small_hnsw
        query = data[0:1]
        D, I = index.search(query, k=5)
        assert D.shape == (1, 5)
        assert I.shape == (1, 5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestHNSWDeterminism:

    def test_same_seed_same_results(self):
        """Building with the same data and searching should give consistent results."""
        np.random.seed(42)
        data = np.random.randn(100, 16).astype(np.float32)
        query = np.random.randn(5, 16).astype(np.float32)

        index1 = HNSWIndex(d=16)
        index1.add(data)
        D1, I1 = index1.search(query, k=5)

        # The same search on the same built index should be deterministic
        D2, I2 = index1.search(query, k=5)
        np.testing.assert_array_equal(I1, I2)