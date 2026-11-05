from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from src.mapreduce import MapReduce
from src.monoid import AvgMonoid, MaxMonoid, Semigroup, SumMonoid, product_monoid


# Funciones de nivel de módulo: se pueden serializar y ejecutar en paralelo.
def split_words(line):
    return line.lower().split()


def word_one(word):
    return (word, 1)


def is_even(x):
    return x % 2 == 0


def square(x):
    return x * x


LINES = ["a b a", "b c", "c c d"]


def word_count(mr):
    return mr.flat_map(split_words).map(word_one).reduce_by_key(SumMonoid).to_dict()


# --------------------------------------------------------------------- básico

def test_of_and_collect():
    assert MapReduce.of([1, 2, 3], workers=1).collect() == [1, 2, 3]


def test_map_filter():
    mr = MapReduce.of(range(6), workers=2)
    assert mr.map(square).filter(is_even).collect() == [0, 4, 16]


def test_word_count():
    assert word_count(MapReduce.of(LINES, workers=2)) == {"a": 2, "b": 2, "c": 3, "d": 1}


def test_lambdas_work_sequentially():
    mr = MapReduce.of(LINES, workers=4)
    result = (
        mr.flat_map(lambda l: l.split())
        .map(lambda w: (w, 1))
        .reduce_by_key(SumMonoid)
        .sort_by_value(descending=True)
        .collect()
    )
    assert result[0] == ("c", 3)


def test_empty_data():
    mr = MapReduce.of([], workers=3)
    assert mr.map(square).collect() == []
    assert mr.reduce_by_key(SumMonoid).collect() == []
    assert mr.aggregate(SumMonoid) == 0


# ------------------------------------------------------------- inmutabilidad

def test_operations_return_new_objects():
    original = MapReduce.of([1, 2, 3], workers=1)
    mapped = original.map(square)
    assert original.collect() == [1, 2, 3]
    assert mapped.collect() == [1, 4, 9]
    assert mapped is not original


def test_frozen():
    mr = MapReduce.of([1])
    with pytest.raises(FrozenInstanceError):
        mr.data = (2,)  # type: ignore[misc]


def test_input_list_not_modified():
    data = [3, 1, 2]
    MapReduce.of(data).map(square)
    assert data == [3, 1, 2]


# --------------------------------------------------- corrección por tipos

def test_reduce_by_key_requires_semigroup():
    mr = MapReduce.of([("a", 1)])
    with pytest.raises(TypeError):
        mr.reduce_by_key(lambda a, b: a + b)  # type: ignore[arg-type]


def test_aggregate_requires_monoid():
    with pytest.raises(TypeError):
        MapReduce.of([1]).aggregate(Semigroup(combine=max))  # type: ignore[arg-type]


# ---------------------------------------------------------------- agregación

def test_aggregate_sum_and_max():
    mr = MapReduce.of(range(101), workers=4)
    assert mr.aggregate(SumMonoid) == 5050
    assert mr.aggregate(MaxMonoid) == 100


def test_average_by_key():
    sales = [("norte", 10), ("sur", 4), ("norte", 20), ("sur", 6)]
    result = (
        MapReduce.of(sales, workers=2)
        .map_values(lambda v: (v, 1))
        .reduce_by_key(AvgMonoid)
        .map_values(lambda p: p[0] / p[1])
        .to_dict()
    )
    assert result == {"norte": 15.0, "sur": 5.0}


def test_product_monoid_by_key():
    stats = product_monoid(total=SumMonoid, top=MaxMonoid)
    rows = [("x", 3), ("x", 7), ("y", 1)]
    result = (
        MapReduce.of(rows, workers=2)
        .map_values(lambda v: {"total": v, "top": v})
        .reduce_by_key(stats)
        .to_dict()
    )
    assert result == {"x": {"total": 10, "top": 7}, "y": {"total": 1, "top": 1}}


def test_group_by_key():
    pairs = [("a", 1), ("b", 2), ("a", 3)]
    assert MapReduce.of(pairs).group_by_key().to_dict() == {"a": (1, 3), "b": (2,)}


def test_key_by_and_sort():
    mr = MapReduce.of(["ccc", "a", "bb"]).key_by(len).sort_by_key()
    assert mr.collect() == [(1, "a"), (2, "bb"), (3, "ccc")]


# --------------------------------------------------------------- paralelismo

@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_equals_sequential(workers):
    lines = [f"w{i % 37} w{i % 11} w{i % 5}" for i in range(3000)]
    expected = Counter(w for l in lines for w in l.split())
    # parallel_threshold=0 fuerza el uso real de procesos.
    mr = MapReduce.of(lines, workers=workers, parallel_threshold=0)
    assert word_count(mr) == dict(expected)


def test_parallel_aggregate():
    mr = MapReduce.of(range(20_000), workers=4, parallel_threshold=0)
    assert mr.map(square).aggregate(SumMonoid) == sum(x * x for x in range(20_000))


def test_invalid_workers():
    with pytest.raises(ValueError):
        MapReduce.of([1], workers=0)


# ----------------------------------------------------------- evaluación lazy

def test_transformations_are_lazy():
    calls = []

    def spy(x):
        calls.append(x)
        return x

    plan = MapReduce.of([1, 2, 3], workers=1).map(spy).filter(is_even)
    assert calls == []            # nada se ejecutó todavía
    assert plan.collect() == [2]  # ahora sí
    assert calls == [1, 2, 3]


def test_cache_materializes_once():
    calls = []

    def spy(x):
        calls.append(x)
        return x

    cached = MapReduce.of([1, 2], workers=1).map(spy).cache()
    cached.collect()
    cached.collect()
    assert calls == [1, 2]
