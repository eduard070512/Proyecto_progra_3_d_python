"""
Agregadores composables (Semana 2).

Un ``Aggregator`` separa una agregación en tres funciones puras:

    prepare : registro -> A      (se ejecuta en cada worker, fase map)
    monoid  : Monoid[A]          (se combina en paralelo, fase reduce)
    present : A -> resultado     (se ejecuta una sola vez, al final)

Ejemplo: el promedio NO es asociativo, pero si ``prepare`` produce
``(valor, 1)``, el monoid suma pares y ``present`` divide, el promedio se puede
calcular en paralelo de forma exacta.

Los agregadores se componen:

* ``combine(total=sum_of("monto"), n=count())`` -> un solo agregador que
  calcula ambos en UNA pasada (producto de monoids).
* ``agg.map(fn)`` -> transforma el resultado final.
* ``agg.where(pred)`` -> sólo cuenta los registros que cumplen ``pred``.

Todas las funciones auxiliares son clases inmutables o funciones de nivel de
módulo para que se puedan serializar con pickle y ejecutarse en otros procesos.
"""
from __future__ import annotations

import heapq
import math
import operator
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Optional, Tuple, TypeVar

from .monoid import (
    AvgMonoid,
    CountMonoid,
    MaxMonoid,
    MinMonoid,
    Monoid,
    SumMonoid,
    product_monoid,
)

T = TypeVar("T")
A = TypeVar("A")
R = TypeVar("R")

Getter = Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Funciones auxiliares serializables
# ---------------------------------------------------------------------------

def identity(x: Any) -> Any:
    return x


def _one(_: Any) -> int:
    return 1


def getter(field: Optional[str | int | Callable[[Any], Any]]) -> Getter:
    """Convierte un nombre de campo (o índice, o función) en una función."""
    if field is None:
        return identity
    if callable(field):
        return field
    return operator.itemgetter(field)


@dataclass(frozen=True)
class Compose:
    """``Compose(f, g)(x) == g(f(x))`` (aplica primero f)."""

    first: Callable[[Any], Any]
    second: Callable[[Any], Any]

    def __call__(self, x: Any) -> Any:
        return self.second(self.first(x))


@dataclass(frozen=True)
class _PairWithOne:
    get: Getter

    def __call__(self, x: Any) -> Tuple[float, int]:
        return (float(self.get(x)), 1)


@dataclass(frozen=True)
class _ToFloat:
    get: Getter

    def __call__(self, x: Any) -> float:
        return float(self.get(x))


@dataclass(frozen=True)
class _WhereCombine:
    """prepare condicionado: si no cumple el predicado produce el neutro."""

    predicate: Callable[[Any], bool]
    prepare: Callable[[Any], Any]
    empty: Any

    def __call__(self, x: Any) -> Any:
        return self.prepare(x) if self.predicate(x) else self.empty


@dataclass(frozen=True)
class _ComposedPrepare:
    parts: Tuple[Tuple[str, Callable[[Any], Any]], ...]

    def __call__(self, x: Any) -> dict:
        return {name: prep(x) for name, prep in self.parts}


@dataclass(frozen=True)
class _ComposedPresent:
    parts: Tuple[Tuple[str, Callable[[Any], Any]], ...]

    def __call__(self, acc: dict) -> dict:
        return {name: pres(acc[name]) for name, pres in self.parts}


def _mean(pair: Tuple[float, int]) -> Optional[float]:
    total, n = pair
    return total / n if n else None


def _none_if_infinite(x: float) -> Optional[float]:
    return None if math.isinf(x) else x


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Aggregator(Generic[T, A, R]):
    prepare: Callable[[T], A]
    monoid: Monoid[A]
    present: Callable[[A], R] = identity  # type: ignore[assignment]

    def aggregate(self, data: Iterable[T]) -> R:
        """Ejecuta la agregación de forma secuencial (útil para pruebas)."""
        return self.present(self.monoid.fold_map(self.prepare, data))

    def map(self, fn: Callable[[R], Any]) -> "Aggregator[T, A, Any]":
        return Aggregator(self.prepare, self.monoid, Compose(self.present, fn))

    def where(self, predicate: Callable[[T], bool]) -> "Aggregator[T, A, R]":
        return Aggregator(
            _WhereCombine(predicate, self.prepare, self.monoid.empty),
            self.monoid,
            self.present,
        )


def combine(**aggs: Aggregator[Any, Any, Any]) -> Aggregator[Any, dict, dict]:
    """Compone varios agregadores en uno solo que hace una única pasada."""
    if not aggs:
        raise ValueError("combine necesita al menos un agregador")
    items = tuple(aggs.items())
    return Aggregator(
        prepare=_ComposedPrepare(tuple((k, a.prepare) for k, a in items)),
        monoid=product_monoid(**{k: a.monoid for k, a in items}),
        present=_ComposedPresent(tuple((k, a.present) for k, a in items)),
    )


# ---------------------------------------------------------------------------
# Agregadores básicos
# ---------------------------------------------------------------------------

def count() -> Aggregator[Any, int, int]:
    return Aggregator(_one, CountMonoid)


def sum_of(field: Any = None) -> Aggregator[Any, float, float]:
    return Aggregator(_ToFloat(getter(field)), SumMonoid)


def mean_of(field: Any = None) -> Aggregator[Any, Tuple[float, int], Optional[float]]:
    return Aggregator(_PairWithOne(getter(field)), AvgMonoid, _mean)


def min_of(field: Any = None) -> Aggregator[Any, float, Optional[float]]:
    return Aggregator(_ToFloat(getter(field)), MinMonoid, _none_if_infinite)


def max_of(field: Any = None) -> Aggregator[Any, float, Optional[float]]:
    return Aggregator(_ToFloat(getter(field)), MaxMonoid, _none_if_infinite)


# ---------------------------------------------------------------------------
# Estadística distribuida: varianza (algoritmo paralelo de Chan et al.)
# ---------------------------------------------------------------------------

Moments = Tuple[int, float, float]  # (n, media, M2)


def _merge_moments(a: Moments, b: Moments) -> Moments:
    """Combina dos resúmenes (n, media, M2). Es asociativo y numéricamente estable."""
    na, ma, m2a = a
    nb, mb, m2b = b
    if na == 0:
        return b
    if nb == 0:
        return a
    n = na + nb
    delta = mb - ma
    mean = ma + delta * nb / n
    m2 = m2a + m2b + delta * delta * na * nb / n
    return (n, mean, m2)


MomentsMonoid: Monoid[Moments] = Monoid(combine=_merge_moments, empty=(0, 0.0, 0.0), name="moments")


@dataclass(frozen=True)
class _ToMoments:
    get: Getter

    def __call__(self, x: Any) -> Moments:
        return (1, float(self.get(x)), 0.0)


@dataclass(frozen=True)
class _PresentMoments:
    what: str        # "variance" | "stddev"
    sample: bool     # True => divide entre n-1

    def __call__(self, m: Moments) -> Optional[float]:
        n, _mean, m2 = m
        denom = n - 1 if self.sample else n
        if denom <= 0:
            return None
        var = m2 / denom
        return math.sqrt(var) if self.what == "stddev" else var


def variance_of(field: Any = None, sample: bool = True) -> Aggregator[Any, Moments, Optional[float]]:
    return Aggregator(_ToMoments(getter(field)), MomentsMonoid, _PresentMoments("variance", sample))


def stddev_of(field: Any = None, sample: bool = True) -> Aggregator[Any, Moments, Optional[float]]:
    return Aggregator(_ToMoments(getter(field)), MomentsMonoid, _PresentMoments("stddev", sample))


# ---------------------------------------------------------------------------
# Percentiles aproximados con histograma (monoid de vectores)
# ---------------------------------------------------------------------------

Sparse = Tuple[Tuple[int, int], ...]  # ((bin, conteo), ...) ordenado por bin


def _merge_sparse(a: Sparse, b: Sparse) -> Sparse:
    """Suma dos histogramas dispersos (merge de listas ordenadas, O(len a + len b))."""
    out: list[Tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        (ka, ca), (kb, cb) = a[i], b[j]
        if ka == kb:
            out.append((ka, ca + cb)); i += 1; j += 1
        elif ka < kb:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return tuple(out)


def _batch_sparse(values: Iterable[Sparse]) -> Sparse:
    totals: dict[int, int] = {}
    for hist in values:
        for k, c in hist:
            totals[k] = totals.get(k, 0) + c
    return tuple(sorted(totals.items()))


HistogramMonoid: Monoid[Sparse] = Monoid(
    combine=_merge_sparse, empty=(), name="histogram", batch=_batch_sparse
)


@dataclass(frozen=True)
class _ToBin:
    get: Getter
    lo: float
    hi: float
    bins: int

    def __call__(self, x: Any) -> Sparse:
        v = float(self.get(x))
        idx = int((v - self.lo) / (self.hi - self.lo) * self.bins)
        idx = min(max(idx, 0), self.bins - 1)  # valores fuera de rango van al extremo
        return ((idx, 1),)


@dataclass(frozen=True)
class _HistogramQuantiles:
    lo: float
    hi: float
    bins: int
    qs: Tuple[float, ...]

    def __call__(self, hist: Sparse) -> dict:
        total = sum(c for _, c in hist)
        if total == 0:
            return {q: None for q in self.qs}
        width = (self.hi - self.lo) / self.bins
        result = {}
        for q in self.qs:
            target = q * total
            seen = 0
            for i, c in hist:
                if seen + c >= target:
                    frac = (target - seen) / c
                    result[q] = self.lo + (i + frac) * width
                    break
                seen += c
        return result


def quantiles_of(
    field: Any,
    qs: Iterable[float] = (0.25, 0.5, 0.75),
    lo: float = 0.0,
    hi: float = 1.0,
    bins: int = 100,
) -> Aggregator[Any, Tuple[int, ...], dict]:
    """Percentiles aproximados en un solo paso distribuido.

    El resultado queda a menos de un bin (``(hi - lo) / bins``) de alguna
    muestra con el rango pedido; con muchos datos eso equivale a un error de
    aproximadamente un bin. Calcular la
    mediana exacta requiere ordenar todos los datos (no es un monoid de tamaño
    fijo); el histograma sí lo es, por eso es la técnica usada en sistemas
    distribuidos.
    """
    if hi <= lo or bins < 1:
        raise ValueError("se requiere hi > lo y bins >= 1")
    qs_t = tuple(qs)
    if any(not 0 <= q <= 1 for q in qs_t):
        raise ValueError("los cuantiles deben estar en [0, 1]")
    return Aggregator(_ToBin(getter(field), lo, hi, bins), HistogramMonoid, _HistogramQuantiles(lo, hi, bins, qs_t))


# ---------------------------------------------------------------------------
# Top-K (monoid acotado)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TopKCombine:
    k: int

    def __call__(self, a: Tuple[Tuple[float, Any], ...], b: Tuple[Tuple[float, Any], ...]) -> Tuple[Tuple[float, Any], ...]:
        return tuple(heapq.nlargest(self.k, a + b, key=operator.itemgetter(0)))


@dataclass(frozen=True)
class _ToScored:
    score: Getter
    label: Getter

    def __call__(self, x: Any) -> Tuple[Tuple[float, Any], ...]:
        return ((float(self.score(x)), self.label(x)),)


def top_k(k: int, score: Any, label: Any = None) -> Aggregator[Any, Tuple[Tuple[float, Any], ...], list]:
    """Los ``k`` elementos con mayor ``score``. Memoria O(k) por worker."""
    if k < 1:
        raise ValueError("k debe ser >= 1")
    monoid = Monoid(combine=_TopKCombine(k), empty=(), name=f"top{k}")
    return Aggregator(_ToScored(getter(score), getter(label)), monoid, list)


# ---------------------------------------------------------------------------
# Resumen estadístico
# ---------------------------------------------------------------------------

def describe(field: Any = None) -> Aggregator[Any, dict, dict]:
    """n, suma, media, desviación estándar, mínimo y máximo en UNA pasada."""
    return combine(
        n=count(),
        total=sum_of(field),
        mean=mean_of(field),
        std=stddev_of(field),
        min=min_of(field),
        max=max_of(field),
    )
