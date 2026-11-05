import math
import random
import statistics

import pytest

from src.aggregators import (
    HistogramMonoid,
    MomentsMonoid,
    combine,
    count,
    describe,
    max_of,
    mean_of,
    min_of,
    quantiles_of,
    stddev_of,
    sum_of,
    top_k,
    variance_of,
)
from src.mapreduce import MapReduce
from src.monoid import check_associativity, check_identity

RNG = random.Random(1)
VALUES = [RNG.uniform(0, 100) for _ in range(500)]
ROWS = [{"cat": RNG.choice("abc"), "v": v} for v in VALUES]


def by_cat(rows, cat):
    return [r["v"] for r in rows if r["cat"] == cat]


def is_big(r):
    return r["v"] > 50


def test_basic_aggregators():
    assert count().aggregate(VALUES) == 500
    assert sum_of().aggregate(VALUES) == pytest.approx(sum(VALUES))
    assert mean_of().aggregate(VALUES) == pytest.approx(statistics.mean(VALUES))
    assert min_of().aggregate(VALUES) == min(VALUES)
    assert max_of().aggregate(VALUES) == max(VALUES)


def test_empty_inputs_return_none_not_crash():
    assert mean_of().aggregate([]) is None
    assert min_of().aggregate([]) is None
    assert variance_of().aggregate([1.0]) is None  # varianza muestral con n=1


def test_variance_matches_statistics():
    assert variance_of().aggregate(VALUES) == pytest.approx(statistics.variance(VALUES))
    assert stddev_of(sample=False).aggregate(VALUES) == pytest.approx(statistics.pstdev(VALUES))


def test_moments_monoid_laws():
    a, b, c = (1, 2.0, 0.0), (3, 5.0, 8.0), (2, -1.0, 2.0)
    m1 = MomentsMonoid.combine(MomentsMonoid.combine(a, b), c)
    m2 = MomentsMonoid.combine(a, MomentsMonoid.combine(b, c))
    assert all(math.isclose(x, y) for x, y in zip(m1, m2))
    assert check_identity(MomentsMonoid, a)


def test_histogram_monoid_laws_and_batch():
    a, b, c = ((1, 2), (5, 1)), ((1, 1), (3, 4)), ((0, 1), (5, 2))
    assert check_associativity(HistogramMonoid, a, b, c)
    assert check_identity(HistogramMonoid, a)
    # batch debe ser equivalente a combinar uno por uno
    from functools import reduce

    assert HistogramMonoid.fold([a, b, c]) == reduce(HistogramMonoid.combine, [a, b, c], ())


def test_quantiles_are_close_to_exact():
    q = quantiles_of(None, qs=(0.1, 0.5, 0.9), lo=0, hi=100, bins=1000).aggregate(VALUES)
    exact = statistics.quantiles(VALUES, n=10)
    assert q[0.5] == pytest.approx(statistics.median(VALUES), abs=0.5)
    assert q[0.1] == pytest.approx(exact[0], abs=0.5)
    assert q[0.9] == pytest.approx(exact[-1], abs=0.5)


def test_quantiles_validation():
    with pytest.raises(ValueError):
        quantiles_of("v", lo=1, hi=0)
    with pytest.raises(ValueError):
        quantiles_of("v", qs=(1.5,))


def test_top_k():
    result = top_k(3, "v", "cat").aggregate(ROWS)
    assert [s for s, _ in result] == sorted(VALUES, reverse=True)[:3]


def test_combine_single_pass():
    stats = combine(n=count(), total=sum_of("v"), top=max_of("v")).aggregate(ROWS)
    assert stats == {"n": 500, "total": pytest.approx(sum(VALUES)), "top": max(VALUES)}


def test_map_and_where():
    doubled = sum_of("v").map(lambda x: 2 * x).aggregate(ROWS)
    assert doubled == pytest.approx(2 * sum(VALUES))
    big = count().where(is_big).aggregate(ROWS)
    assert big == sum(1 for v in VALUES if v > 50)


def test_describe():
    d = describe("v").aggregate(ROWS)
    assert d["n"] == 500
    assert d["mean"] == pytest.approx(statistics.mean(VALUES))
    assert d["std"] == pytest.approx(statistics.stdev(VALUES))


# ------------------------------------------------- integración con MapReduce

@pytest.mark.parametrize("workers", [1, 3])
def test_aggregate_with_parallel_equals_sequential(workers):
    mr = MapReduce.of(ROWS, workers=workers, parallel_threshold=0)
    assert mr.aggregate_with(describe("v")) == pytest.approx(describe("v").aggregate(ROWS))


@pytest.mark.parametrize("workers", [1, 4])
def test_aggregate_by_key(workers):
    mr = MapReduce.of(ROWS, workers=workers, parallel_threshold=0)
    agg = combine(n=count(), mean=mean_of("v"), std=stddev_of("v"))
    result = mr.aggregate_by_key("cat", agg).to_dict()
    assert set(result) == {"a", "b", "c"}
    for cat, stats in result.items():
        vals = by_cat(ROWS, cat)
        assert stats["n"] == len(vals)
        assert stats["mean"] == pytest.approx(statistics.mean(vals))
        assert stats["std"] == pytest.approx(statistics.stdev(vals))


def test_quantiles_by_key_parallel():
    mr = MapReduce.of(ROWS, workers=4, parallel_threshold=0)
    q = mr.aggregate_by_key("cat", quantiles_of("v", qs=(0.5,), lo=0, hi=100, bins=1000)).to_dict()
    width = 100 / 1000
    for cat, res in q.items():
        vals = by_cat(ROWS, cat)
        # el resultado cae entre las dos muestras centrales (± un bin)
        assert statistics.median_low(vals) - width <= res[0.5] <= statistics.median_high(vals) + width
