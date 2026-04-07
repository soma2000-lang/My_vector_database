"""
Tests for the SkipList implementation (tinyhnsw.teaching.skip_list).

SkipList is a probabilistic data structure used as a teaching tool to build
intuition for HNSW's hierarchical layer structure. It supports integer keys
with find, insert, delete, and tolist operations.

Run: pytest tests/test_skip_list.py -v
"""

import pytest
import random

from tinyhnsw.teaching.skip_list import SkipList


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSkipListConstruction:
    """Test building skip lists from various inputs."""

    def test_construct_from_list(self):
        s = SkipList([3, 1, 4, 1, 5, 9, 2, 6])
        result = s.tolist()
        # Should contain unique sorted elements
        assert sorted(set([3, 1, 4, 5, 9, 2, 6])) == result or set(result) == set([1, 2, 3, 4, 5, 6, 9])

    def test_construct_empty(self):
        s = SkipList([])
        assert s.tolist() == []

    def test_construct_single_element(self):
        s = SkipList([42])
        assert s.tolist() == [42]

    def test_construct_sorted_input(self):
        s = SkipList([1, 2, 3, 4, 5])
        assert s.tolist() == [1, 2, 3, 4, 5]

    def test_construct_reverse_sorted(self):
        s = SkipList([5, 4, 3, 2, 1])
        result = s.tolist()
        assert result == [1, 2, 3, 4, 5]

    def test_construct_large_list(self):
        data = list(range(100))
        random.shuffle(data)
        s = SkipList(data)
        assert s.tolist() == list(range(100))


# ---------------------------------------------------------------------------
# Find
# ---------------------------------------------------------------------------

class TestSkipListFind:
    """Test the find operation."""

    def test_find_existing_element(self):
        s = SkipList([3, 1, 7, 9, 14, 6, 2])
        node = s.find(3)
        assert node is not None
        assert node.value == 3

    def test_find_first_element(self):
        s = SkipList([5, 2, 8, 1])
        node = s.find(1)
        assert node is not None
        assert node.value == 1

    def test_find_last_element(self):
        s = SkipList([5, 2, 8, 1])
        node = s.find(8)
        assert node is not None
        assert node.value == 8

    def test_find_nonexistent_element(self):
        s = SkipList([3, 1, 7, 9, 14, 6, 2])
        node = s.find(100)
        assert node is None

    def test_find_in_empty_list(self):
        s = SkipList([])
        node = s.find(5)
        assert node is None

    def test_find_all_elements(self):
        values = [3, 1, 7, 9, 14, 6, 2]
        s = SkipList(values)
        for v in values:
            node = s.find(v)
            assert node is not None
            assert node.value == v


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

class TestSkipListInsert:
    """Test the insert operation."""

    def test_insert_single(self):
        s = SkipList([1, 3, 5])
        s.insert(4)
        assert 4 in s.tolist()

    def test_insert_maintains_order(self):
        s = SkipList([1, 3, 5, 7])
        s.insert(4)
        result = s.tolist()
        assert result == sorted(result)

    def test_insert_at_beginning(self):
        s = SkipList([5, 10, 15])
        s.insert(1)
        result = s.tolist()
        assert result[0] == 1

    def test_insert_at_end(self):
        s = SkipList([5, 10, 15])
        s.insert(20)
        result = s.tolist()
        assert result[-1] == 20

    def test_insert_into_empty(self):
        s = SkipList([])
        s.insert(42)
        assert s.tolist() == [42]

    def test_insert_multiple(self):
        s = SkipList([])
        for i in [5, 3, 8, 1, 9, 2]:
            s.insert(i)
        assert s.tolist() == [1, 2, 3, 5, 8, 9]

    def test_insert_preserves_existing(self):
        s = SkipList([1, 3, 5])
        s.insert(4)
        for v in [1, 3, 5]:
            assert s.find(v) is not None

    def test_insert_then_find(self):
        s = SkipList([10, 20, 30])
        s.insert(25)
        node = s.find(25)
        assert node is not None
        assert node.value == 25


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestSkipListDelete:
    """Test the delete operation."""

    def test_delete_middle_element(self):
        s = SkipList([1, 3, 5, 7, 9])
        s.delete(5)
        assert 5 not in s.tolist()

    def test_delete_first_element(self):
        s = SkipList([1, 3, 5])
        s.delete(1)
        assert 1 not in s.tolist()
        assert 3 in s.tolist()

    def test_delete_last_element(self):
        s = SkipList([1, 3, 5])
        s.delete(5)
        assert 5 not in s.tolist()
        assert 3 in s.tolist()

    def test_delete_preserves_order(self):
        s = SkipList([1, 2, 3, 4, 5])
        s.delete(3)
        result = s.tolist()
        assert result == sorted(result)

    def test_delete_then_find_returns_none(self):
        s = SkipList([1, 3, 5, 7])
        s.delete(3)
        assert s.find(3) is None

    def test_delete_does_not_affect_others(self):
        s = SkipList([1, 3, 5, 7, 9])
        s.delete(5)
        for v in [1, 3, 7, 9]:
            assert s.find(v) is not None

    def test_delete_all_elements(self):
        values = [1, 2, 3, 4, 5]
        s = SkipList(values)
        for v in values:
            s.delete(v)
        assert s.tolist() == []


# ---------------------------------------------------------------------------
# tolist
# ---------------------------------------------------------------------------

class TestSkipListToList:
    """Test the tolist method."""

    def test_tolist_returns_sorted(self):
        s = SkipList([9, 3, 7, 1, 5])
        assert s.tolist() == [1, 3, 5, 7, 9]

    def test_tolist_after_modifications(self):
        s = SkipList([1, 2, 3, 4, 5])
        s.delete(3)
        s.insert(6)
        result = s.tolist()
        assert result == sorted(result)
        assert 3 not in result
        assert 6 in result


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------

class TestSkipListRepr:
    """Test that the skip list can be printed (has levels)."""

    def test_str_does_not_crash(self):
        s = SkipList([3, 1, 7, 9, 14, 6, 2])
        output = str(s)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_has_multiple_levels(self):
        """A skip list with enough elements should have at least 2 levels."""
        random.seed(42)
        s = SkipList(list(range(50)))
        output = str(s)
        # The string representation shows numbered levels like "0 |", "1 |", etc.
        lines = output.strip().split("\n")
        assert len(lines) >= 2


# ---------------------------------------------------------------------------
# Stress / Randomized
# ---------------------------------------------------------------------------

class TestSkipListStress:
    """Randomized stress tests for correctness."""

    def test_random_insert_delete_cycle(self):
        random.seed(123)
        s = SkipList([])
        active = set()

        for _ in range(200):
            if random.random() < 0.6 or len(active) == 0:
                val = random.randint(0, 500)
                s.insert(val)
                active.add(val)
            else:
                val = random.choice(list(active))
                s.delete(val)
                active.discard(val)

        assert set(s.tolist()) == active

    def test_insert_order_invariance(self):
        """Same elements inserted in different orders produce same sorted list."""
        data = list(range(20))
        s1 = SkipList(data)

        random.seed(99)
        shuffled = data.copy()
        random.shuffle(shuffled)
        s2 = SkipList(shuffled)

        assert s1.tolist() == s2.tolist()