"""
Semigroups y Monoids para agregación distribuida.

Un *Semigroup* es un tipo A con una operación binaria ``combine: A x A -> A``
que es **asociativa**:

    combine(combine(a, b), c) == combine(a, combine(b, c))

Un *Monoid* es un Semigroup que además tiene un **elemento neutro** ``empty``:

    combine(empty, a) == a == combine(a, empty)

¿Por qué importa para Map-Reduce? Porque la asociatividad permite partir los
datos en particiones, reducir cada una en paralelo (en cualquier worker) y
después combinar los resultados parciales: el resultado es el mismo que si se
hubiera reducido todo secuencialmente. El neutro permite manejar particiones
vacías sin casos especiales.

Todas las operaciones de este módulo son funciones puras y las estructuras son
inmutables (``frozen=True``).
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Generic, Iterable, Mapping, Tuple, TypeVar

A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class Semigroup(Generic[A]):
    """Operación binaria asociativa sobre A."""

    combine: Callable[[A, A], A]
    name: str = "semigroup"

    def combine_all(self, values: Iterable[A]) -> A:
        """Reduce una colección NO vacía."""
        iterator = iter(values)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError(f"{self.name}: no se puede combinar una colección vacía") from None
        return reduce(self.combine, iterator, first)


@dataclass(frozen=True)
class Monoid(Semigroup[A]):
    """Semigroup con elemento neutro."""

    empty: Any = None

    def fold(self, values: Iterable[A]) -> A:
        """Reduce cualquier colección (vacía incluida) empezando en ``empty``."""
        return reduce(self.combine, values, self.empty)

    def fold_map(self, fn: Callable[[B], A], values: Iterable[B]) -> A:
        """Mapea cada elemento a A y luego lo pliega con el monoid."""
        return self.fold(map(fn, values))


# ---------------------------------------------------------------------------
# Funciones de combinación con nombre (deben ser de nivel de módulo para que
# multiprocessing pueda serializarlas con pickle; las lambdas no se pueden).
# ---------------------------------------------------------------------------

def _min(a: Any, b: Any) -> Any:
    return a if a <= b else b


def _max(a: Any, b: Any) -> Any:
    return a if a >= b else b


def _pair_add(a: Tuple[float, int], b: Tuple[float, int]) -> Tuple[float, int]:
    return (a[0] + b[0], a[1] + b[1])


def _first(a: Any, _b: Any) -> Any:
    return a


def _last(_a: Any, b: Any) -> Any:
    return b


# ---------------------------------------------------------------------------
# Instancias predefinidas
# ---------------------------------------------------------------------------

SumMonoid: Monoid[float] = Monoid(combine=operator.add, empty=0, name="sum")
ProductMonoid: Monoid[float] = Monoid(combine=operator.mul, empty=1, name="product")
CountMonoid: Monoid[int] = Monoid(combine=operator.add, empty=0, name="count")
MinMonoid: Monoid[float] = Monoid(combine=_min, empty=float("inf"), name="min")
MaxMonoid: Monoid[float] = Monoid(combine=_max, empty=float("-inf"), name="max")
# Promedio como par (suma, cantidad): el par SÍ es un monoid, el promedio no.
AvgMonoid: Monoid[Tuple[float, int]] = Monoid(combine=_pair_add, empty=(0, 0), name="avg")
AllMonoid: Monoid[bool] = Monoid(combine=operator.and_, empty=True, name="all")
AnyMonoid: Monoid[bool] = Monoid(combine=operator.or_, empty=False, name="any")

# Semigroups sin neutro natural
FirstSemigroup: Semigroup[Any] = Semigroup(combine=_first, name="first")
LastSemigroup: Semigroup[Any] = Semigroup(combine=_last, name="last")


# ---------------------------------------------------------------------------
# Composición de monoids
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProductCombine:
    """Combina diccionarios campo por campo con el monoid de cada campo.

    Se implementa como clase (y no como lambda) para que sea serializable.
    """

    monoids: Tuple[Tuple[str, Monoid[Any]], ...]

    def __call__(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
        return {k: m.combine(a[k], b[k]) for k, m in self.monoids}


def product_monoid(**monoids: Monoid[Any]) -> Monoid[dict]:
    """Producto de monoids: el resultado también es un monoid.

    >>> stats = product_monoid(total=SumMonoid, n=CountMonoid)
    >>> stats.fold([{"total": 5, "n": 1}, {"total": 3, "n": 1}])
    {'total': 8, 'n': 2}

    Permite calcular varias agregaciones en UNA sola pasada sobre los datos.
    """
    items = tuple(monoids.items())
    return Monoid(
        combine=_ProductCombine(items),
        empty={k: m.empty for k, m in items},
        name="product(" + ",".join(k for k, _ in items) + ")",
    )


# ---------------------------------------------------------------------------
# Verificación de leyes (útil en tests)
# ---------------------------------------------------------------------------

def check_associativity(sg: Semigroup[A], a: A, b: A, c: A) -> bool:
    return sg.combine(sg.combine(a, b), c) == sg.combine(a, sg.combine(b, c))


def check_identity(m: Monoid[A], a: A) -> bool:
    return m.combine(m.empty, a) == a and m.combine(a, m.empty) == a
