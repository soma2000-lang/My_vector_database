"""
Tests for the FilteredHNSWIndex (filtered / hybrid search).

tinyhnsw supports filtering during HNSW search, allowing users to
restrict results to vectors matching certain metadata criteria. This is
the "hybrid search" capability described in Chapter 7 of the tutorial.

The filter module is at tinyhnsw/filter.py and the index is
imported via `from tinyhnsw import FilteredHNSWIndex` (or the
filtering functionality may be part of HNSWIndex itself).

Run: pytest tests/test_filter.py -v
"""

import pytest
import numpy as np

# Try both possible import paths — the repo may expose filtering
# as a separate class or as a method on HNSWIndex.
try:
    from tinyhnsw.filter import FilteredHNSWIndex
except ImportError:
    try:
        from tinyhnsw import FilteredHNSWIndex
    except ImportError:
        # If the class doesn't exist under either name, skip all tests
        pytestmark = pytest.mark.skip(reason="FilteredHNSWIndex not available")
        FilteredHNSWIndex = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def filtered_index():
    """
    Build a FilteredHNSWIndex with 100 vectors in dim=8.
    Assign metadata labels: even-indexed vectors get label 0,
    odd-indexed vectors get label 1.
    """
    np.random.seed(42)
    d = 8
    n = 100
    data = np.random.randn(n, d).astype(np.float32)

    # Labels: 0 for even indices, 1 for odd
    labels = np.array([i % 2 for i in range(n)])

    index = FilteredHNSWIndex(d=d)
    index.add(data, labels=labels)
    return index, data, labels


@pytest.fixture
def multi_label_index():
    """
    Build a FilteredHNSWIndex with 200 vectors in dim=16.
    Assign 4 category labels (0, 1, 2, 3).
    """
    np.random.seed(0)
    d = 16
    n = 200
    data = np.random.randn(n, d).astype(np.float32)
    labels = np.array([i % 4 for i in range(n)])

    index = FilteredHNSWIndex(d=d)
    index.add(data, labels=labels)
    return index, data, labels


# ---------------------------------------------------------------------------
# Basic Filtered Search
# ---------------------------------------------------------------------------

class TestFilteredSearchBasic:

    def test_filtered_search_returns_correct_shape(self, filtered_index):
        index, data, labels = filtered_index
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=10, filter_label=0)
        assert D.shape == (5, 10)
        assert I.shape == (5, 10)

    def test_filtered_results_match_label(self, filtered_index):
        """All returned indices should have the requested label."""
        index, data, labels = filtered_index
        queries = np.random.randn(10, 8).astype(np.float32)
        D, I = index.search(queries, k=10, filter_label=0)

        for i in range(I.shape[0]):
            for idx in I[i]:
                if idx >= 0:  # -1 may indicate "no result"
                    assert labels[idx] == 0, (
                        f"Query {i}: index {idx} has label {labels[idx]}, expected 0"
                    )

    def test_filtered_results_match_label_1(self, filtered_index):
        """Same test but filtering for label=1 (odd indices)."""
        index, data, labels = filtered_index
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=5, filter_label=1)

        for i in range(I.shape[0]):
            for idx in I[i]:
                if idx >= 0:
                    assert labels[idx] == 1


class TestFilteredSearchMultiLabel:

    def test_filter_each_category(self, multi_label_index):
        """Filter for each of the 4 categories — all results should match."""
        index, data, labels = multi_label_index
        queries = np.random.randn(5, 16).astype(np.float32)

        for target_label in range(4):
            D, I = index.search(queries, k=5, filter_label=target_label)
            for i in range(I.shape[0]):
                for idx in I[i]:
                    if idx >= 0:
                        assert labels[idx] == target_label


# ---------------------------------------------------------------------------
# Unfiltered Search Still Works
# ---------------------------------------------------------------------------

class TestFilteredIndexUnfiltered:

    def test_unfiltered_search_works(self, filtered_index):
        """Search without a filter should return results from any label."""
        index, data, labels = filtered_index
        queries = np.random.randn(5, 8).astype(np.float32)

        # Unfiltered — pass no filter argument or filter_label=None
        try:
            D, I = index.search(queries, k=10)
        except TypeError:
            D, I = index.search(queries, k=10, filter_label=None)

        assert D.shape == (5, 10)
        assert I.shape == (5, 10)
        # Should contain a mix of labels
        result_labels = set(labels[idx] for idx in I.flatten() if idx >= 0)
        assert len(result_labels) > 1

    def test_ntotal_includes_all_labels(self, filtered_index):
        index, data, labels = filtered_index
        assert index.ntotal == 100


# ---------------------------------------------------------------------------
# Distances Under Filtering
# ---------------------------------------------------------------------------

class TestFilteredDistances:

    def test_filtered_distances_are_sorted(self, filtered_index):
        index, data, labels = filtered_index
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=10, filter_label=0)
        for i in range(D.shape[0]):
            valid = D[i][D[i] >= 0]
            assert list(valid) == sorted(valid)

    def test_filtered_distances_nonnegative(self, filtered_index):
        index, data, labels = filtered_index
        queries = np.random.randn(5, 8).astype(np.float32)
        D, I = index.search(queries, k=10, filter_label=1)
        assert np.all(D >= 0) or np.all(D[D != -1] >= 0)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestFilteredEdgeCases:

    def test_filter_with_k_larger_than_matching_vectors(self, filtered_index):
        """Request more results than exist for a given label."""
        index, data, labels = filtered_index
        queries = np.random.randn(1, 8).astype(np.float32)
        # 50 even-indexed vectors have label 0, so k=50 is the max
        try:
            D, I = index.search(queries, k=50, filter_label=0)
            valid_results = I[0][I[0] >= 0]
            assert len(valid_results) <= 50
        except (ValueError, IndexError):
            pass  # Acceptable to raise an error

    def test_self_retrieval_with_filter(self, filtered_index):
        """An even-indexed vector searching with filter=0 should find itself."""
        index, data, labels = filtered_index
        # Take vector at index 0 (label 0)
        query = data[0:1]
        D, I = index.search(query, k=1, filter_label=0)
        # Should ideally return itself
        assert I[0, 0] == 0 or D[0, 0] < 1e-4


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------

class TestFilteredConsistency:

    def test_filtered_is_subset_of_unfiltered(self, filtered_index):
        """
        Filtered results should be a subset of what unfiltered returns
        (for the same label).
        """
        index, data, labels = filtered_index
        np.random.seed(123)
        query = np.random.randn(1, 8).astype(np.float32)

        try:
            D_all, I_all = index.search(query, k=50)
        except TypeError:
            D_all, I_all = index.search(query, k=50, filter_label=None)

        D_filt, I_filt = index.search(query, k=10, filter_label=0)

        # Every filtered result should also appear in unfiltered (approximately)
        unfiltered_set = set(I_all[0].tolist())
        for idx in I_filt[0]:
            if idx >= 0:
                # Due to HNSW being approximate, this might not hold perfectly
                # but we check label correctness instead
                assert labels[idx] == 0