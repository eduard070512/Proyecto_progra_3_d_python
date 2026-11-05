import pytest

from src.partitioners import chunk, hash_partition, range_partition, stable_hash


def test_chunk_balanced_and_complete():
    parts = chunk(range(10), 3)
    assert parts == ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert sum(len(p) for p in parts) == 10


def test_chunk_more_parts_than_items():
    assert chunk([1, 2], 4) == ((1,), (2,), (), ())


def test_chunk_invalid():
    with pytest.raises(ValueError):
        chunk([1], 0)


def test_hash_partition_same_key_same_partition():
    pairs = [("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5)]
    parts = hash_partition(pairs, 3)
    for key in "abc":
        holders = {i for i, p in enumerate(parts) for k, _ in p if k == key}
        assert len(holders) == 1
    assert sorted(x for p in parts for x in p) == sorted(pairs)


def test_stable_hash_is_deterministic():
    assert stable_hash("hola") == stable_hash("hola")


def test_range_partition():
    parts = range_partition([1, 15, 25, 10, 5, 20], lambda x: x, [10, 20])
    assert parts == ((1, 5), (15, 10), (25, 20))


def test_range_partition_requires_sorted_bounds():
    with pytest.raises(ValueError):
        range_partition([1], lambda x: x, [20, 10])


def test_partitions_do_not_modify_input():
    data = [3, 1, 2]
    chunk(data, 2)
    assert data == [3, 1, 2]
