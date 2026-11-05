"""Sistema de Agregación y Análisis de Datos Distribuidos (Proyecto 7)."""
from .mapreduce import MapReduce
from .monoid import (
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
    Semigroup,
    SumMonoid,
    product_monoid,
)
from .partitioners import chunk, hash_partition, range_partition

__all__ = [
    "MapReduce",
    "Monoid",
    "Semigroup",
    "SumMonoid",
    "ProductMonoid",
    "CountMonoid",
    "MinMonoid",
    "MaxMonoid",
    "AvgMonoid",
    "AllMonoid",
    "AnyMonoid",
    "FirstSemigroup",
    "LastSemigroup",
    "product_monoid",
    "chunk",
    "hash_partition",
    "range_partition",
]
