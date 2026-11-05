import random

import pytest

from src.monoid import (
    AllMonoid,
    AnyMonoid,
    AvgMonoid,
    CountMonoid,
    FirstSemigroup,
    LastSemigroup,
    MaxMonoid,
    MinMonoid,
    Monoid,
    ProductMonoid,
    SumMonoid,
    check_associativity,
    check_identity,
    product_monoid,
)

RNG = random.Random(42)
INTS = [RNG.randint(-1000, 1000) for _ in range(60)]
BOOLS = [True, False]
PAIRS = [(RNG.randint(-100, 100), RNG.randint(0, 10)) for _ in range(30)]

NUMERIC = [SumMonoid, ProductMonoid, CountMonoid, MinMonoid, MaxMonoid]


@pytest.mark.parametrize("m", NUMERIC, ids=lambda m: m.name)
def test_numeric_laws(m):
    for a, b, c in zip(INTS, INTS[1:], INTS[2:]):
        assert check_associativity(m, a, b, c)
        assert check_identity(m, a)


@pytest.mark.parametrize("m", [AllMonoid, AnyMonoid], ids=lambda m: m.name)
def test_boolean_laws(m):
    for a in BOOLS:
        assert check_identity(m, a)
        for b in BOOLS:
            for c in BOOLS:
                assert check_associativity(m, a, b, c)


def test_avg_laws():
    for a, b, c in zip(PAIRS, PAIRS[1:], PAIRS[2:]):
        assert check_associativity(AvgMonoid, a, b, c)
        assert check_identity(AvgMonoid, a)


def test_first_last_semigroups():
    assert FirstSemigroup.combine_all([3, 1, 2]) == 3
    assert LastSemigroup.combine_all([3, 1, 2]) == 2
    assert check_associativity(FirstSemigroup, 1, 2, 3)
    with pytest.raises(ValueError):
        FirstSemigroup.combine_all([])


def test_fold_empty_returns_identity():
    assert SumMonoid.fold([]) == 0
    assert MaxMonoid.fold([]) == float("-inf")


def test_fold_map():
    assert SumMonoid.fold_map(len, ["ab", "cde", ""]) == 5


def test_fold_is_partition_independent():
    """La razón de usar monoids: reducir por partes = reducir todo."""
    whole = SumMonoid.fold(INTS)
    for cut in (0, 1, 17, len(INTS)):
        parts = [SumMonoid.fold(INTS[:cut]), SumMonoid.fold(INTS[cut:])]
        assert SumMonoid.fold(parts) == whole


def test_product_monoid():
    stats = product_monoid(total=SumMonoid, n=CountMonoid, top=MaxMonoid)
    rows = [{"total": x, "n": 1, "top": x} for x in [5, 9, 2]]
    assert stats.fold(rows) == {"total": 16, "n": 3, "top": 9}
    a, b, c = rows
    assert check_associativity(stats, a, b, c)
    assert check_identity(stats, a)


def test_monoid_is_immutable():
    with pytest.raises(Exception):
        SumMonoid.empty = 5  # type: ignore[misc]


def test_custom_monoid():
    concat = Monoid(combine=lambda a, b: a + b, empty="", name="concat")
    assert concat.fold(["a", "b", "c"]) == "abc"
