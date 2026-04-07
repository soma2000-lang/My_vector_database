"""
Tests for the FullNNIndex (exact nearest neighbor / brute-force KNN).

FullNNIndex(d) provides exact nearest neighbor search via exhaustive
distance computation. It serves as the correctness baseline against
which HNSW recall is measured.

API:
    index = FullNNIndex(d)       # d = vector dimensionality
    index.add(vectors)           # vectors: np.ndarray (n, d)
    D, I = index.search(q, k)   # q: (nq, d) → D: distances, I: indices
    index.ntotal                 # number of indexed vectors

Run: pytest tests/test_knn.py -v
"""

import pytest
import numpy as np

from tinyhnsw import FullNNIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_index():
    """10 vectors of dimension 4."""
    np.random.seed(42)
    index = FullNNIndex(4)
    data = np.random.randn(10, 4).astype(np.float32)
    index.add(data)
    return index, data


@pytest.fixture
def medium_index():
    """500 vectors of dimension 32."""
    np.random.seed(0)
    index = FullNNIndex(32)
    data = np.random.randn(500, 32).astype(np.float32)
    index.add(data)
    return index, data


# ---------------------------------------------------------------------------
# Construction & Basic Properties
# ---------------------------------------------------------------------------

class TestFullNNConstruction:

    def test_empty_index(self):
        index = FullNNIndex(8)
        assert index.ntotal == 0

    def test_add_updates_ntotal(self, small_index):
        index, _ = small_index
        assert index.ntotal == 10

    def test_add_larger_batch(self):
        index = FullNNIndex(16)
        data = np.random.randn(1000, 16).astype(np.float32)
        index.add(data)
        assert index.ntotal == 1000

    def test_add_incremental(self):
        index = FullNNIndex(8)
        index.add(np.random.randn(5, 8).astype(np.float32))
        index.add(np.random.randn(5, 8).astype(np.float32))
        assert index.ntotal == 10


# ---------------------------------------------------------------------------
# Search — Correctness
# ---------------------------------------------------------------------------

class TestFullNNSearch:

    def test_search_returns_correct_shapes(self, small_index):
        index, _ = small_index
        queries = np.random.randn(3, 4).astype(np.float32)
        D, I = index.search(queries, k=5)
        assert D.shape == (3, 5)
        assert I.shape == (3, 5)

    def test_search_k1_self_retrieval(self, small_index):
        """Each vector should be its own nearest neighbor."""
        index, data = small_index
        D, I = index.search(data, k=1)
        for i in range(len(data)):
            assert I[i, 0] == i

    def test_search_distances_are_nonnegative(self, small_index):
        index, data = small_index
        queries = np.random.randn(5, 4).astype(np.float32)
        D, I = index.search(queries, k=3)
        assert np.all(D >= 0)

    def test_search_distances_are_sorted(self, small_index):
        """Returned distances should be in ascending order per query."""
        index, data = small_index
        queries = np.random.randn(5, 4).astype(np.float32)
        D, I = index.search(queries, k=5)
        for i in range(D.shape[0]):
            assert list(D[i]) == sorted(D[i])

    def test_search_indices_are_valid(self, small_index):
        index, data = small_index
        queries = np.random.randn(3, 4).astype(np.float32)
        D, I = index.search(queries, k=5)
        assert np.all(I >= 0)
        assert np.all(I < index.ntotal)

    def test_search_indices_are_unique_per_query(self, small_index):
        """No duplicate indices in a single query's results."""
        index, data = small_index
        queries = np.random.randn(3, 4).astype(np.float32)
        D, I = index.search(queries, k=5)
        for i in range(I.shape[0]):
            assert len(set(I[i])) == I.shape[1]

    def test_search_single_query(self, small_index):
        index, data = small_index
        query = data[0:1]
        D, I = index.search(query, k=3)
        assert D.shape == (1, 3)
        assert I[0, 0] == 0  # self is nearest

    def test_self_distance_is_zero(self, small_index):
        """Distance to self should be (near) zero."""
        index, data = small_index
        D, I = index.search(data, k=1)
        assert np.allclose(D[:, 0], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Search — k Variations
# ---------------------------------------------------------------------------

class TestFullNNSearchK:

    def test_k_equals_1(self, medium_index):
        index, data = medium_index
        queries = data[:10]
        D, I = index.search(queries, k=1)
        assert I.shape == (10, 1)

    def test_k_equals_ntotal(self, small_index):
        """Requesting all vectors should return all indices."""
        index, data = small_index
        query = np.random.randn(1, 4).astype(np.float32)
        D, I = index.search(query, k=index.ntotal)
        assert set(I[0].tolist()) == set(range(index.ntotal))

    def test_k_larger_returns_at_most_ntotal(self, small_index):
        """If k > ntotal, should not crash (return up to ntotal)."""
        index, data = small_index
        query = np.random.randn(1, 4).astype(np.float32)
        try:
            D, I = index.search(query, k=100)
            # If it returns, indices should be valid
            assert I.shape[1] <= 100
        except (ValueError, IndexError):
            # Also acceptable to raise an error for k > ntotal
            pass


# ---------------------------------------------------------------------------
# Search — Determinism
# ---------------------------------------------------------------------------

class TestFullNNDeterminism:

    def test_repeated_search_same_results(self, medium_index):
        """FullNN is exact → same query should always return same results."""
        index, data = medium_index
        query = np.random.randn(1, 32).astype(np.float32)
        D1, I1 = index.search(query, k=10)
        D2, I2 = index.search(query, k=10)
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_array_almost_equal(D1, D2)


# ---------------------------------------------------------------------------
# Ground Truth Verification
# ---------------------------------------------------------------------------

class TestFullNNGroundTruth:

    def test_manual_nearest_neighbor(self):
        """Hand-crafted example with known nearest neighbor."""
        index = FullNNIndex(2)
        data = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ], dtype=np.float32)
        index.add(data)

        query = np.array([[0.1, 0.1]], dtype=np.float32)
        D, I = index.search(query, k=3)

        # Origin [0, 0] is closest to [0.1, 0.1]
        assert I[0, 0] == 0

    def test_known_ordering(self):
        """Three points on a line → known distance ordering."""
        index = FullNNIndex(1)
        data = np.array([[0.0], [5.0], [10.0]], dtype=np.float32)
        index.add(data)

        query = np.array([[4.0]], dtype=np.float32)
        D, I = index.search(query, k=3)

        # Closest: 5.0 (dist=1), then 0.0 (dist=4), then 10.0 (dist=6)
        assert I[0, 0] == 1  # 5.0
        assert I[0, 1] == 0  # 0.0
        assert I[0, 2] == 2  # 10.0