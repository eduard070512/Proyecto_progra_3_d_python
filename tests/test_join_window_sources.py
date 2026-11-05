import csv
import pickle

import pytest

from src.aggregators import count, sum_of
from src.mapreduce import MapReduce
from src.sources import Record, read_csv_sources

LEFT = [("a", 1), ("b", 2), ("a", 3), ("c", 4)]
RIGHT = [("a", "x"), ("b", "y"), ("d", "z")]


@pytest.mark.parametrize("workers", [1, 3])
def test_inner_join(workers):
    l = MapReduce.of(LEFT, workers=workers, parallel_threshold=0)
    result = l.join(MapReduce.of(RIGHT)).collect()
    assert sorted(result) == [("a", (1, "x")), ("a", (3, "x")), ("b", (2, "y"))]


def test_left_and_outer_join():
    l, r = MapReduce.of(LEFT, workers=2), MapReduce.of(RIGHT)
    assert ("c", (4, None)) in l.join(r, "left").collect()
    outer = l.join(r, "outer").collect()
    assert ("c", (4, None)) in outer and ("d", (None, "z")) in outer


def test_join_invalid_how():
    with pytest.raises(ValueError):
        MapReduce.of(LEFT).join(MapReduce.of(RIGHT), "cross")


def test_union():
    assert MapReduce.of([1, 2]).union(MapReduce.of([3])).collect() == [1, 2, 3]


def ts(e):
    return e["t"]


def test_window():
    events = [{"t": t} for t in (0, 5, 9, 10, 25)]
    result = (
        MapReduce.of(events)
        .window(ts, 10)
        .map_values(lambda _: 1)
        .reduce_by_key(__import__("src").SumMonoid)
        .to_dict()
    )
    assert result == {0: 3, 10: 1, 20: 1}
    with pytest.raises(ValueError):
        MapReduce.of(events).window(ts, 0)


# ------------------------------------------------------------------ Record

def test_record_is_immutable_and_picklable():
    r = Record({"a": 1})
    with pytest.raises(TypeError):
        r["a"] = 2
    with pytest.raises(TypeError):
        r.update(b=2)
    assert pickle.loads(pickle.dumps(r)) == r
    r2 = r.with_(a=5)
    assert r == {"a": 1} and r2 == {"a": 5}


# ------------------------------------------------------------------ fuentes

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.mark.parametrize("workers", [1, 2])
def test_read_csv_sources(tmp_path, workers):
    write_csv(tmp_path / "tienda1.csv", [{"monto": "10"}, {"monto": "5"}])
    write_csv(tmp_path / "tienda2.csv", [{"monto": "7"}, {"monto": "malo"}])
    loaded = read_csv_sources(
        [tmp_path / "tienda1.csv", tmp_path / "tienda2.csv"], {"monto": float}, workers=workers
    )
    assert len(loaded.errors) == 1 and loaded.errors[0].source == "tienda2"
    totals = loaded.data.aggregate_by_key("source", sum_of("monto")).to_dict()
    assert totals == {"tienda1": 15.0, "tienda2": 7.0}
    assert loaded.data.aggregate_with(count()) == 3


def test_read_csv_requires_files():
    with pytest.raises(ValueError):
        read_csv_sources([])
