"""
Map-Reduce funcional, inmutable y perezoso (lazy).

Principios de diseño
--------------------
* **Inmutabilidad**: ``MapReduce`` es un ``dataclass(frozen=True)``. Cada
  operación devuelve un objeto NUEVO; el original nunca cambia.
* **Funciones puras**: los mappers / reducers no deben tener efectos laterales.
  Esto es lo que hace seguro ejecutarlos en varios procesos.
* **Evaluación perezosa**: ``map``, ``flat_map``, ``filter``, ``map_values`` y
  ``key_by`` NO se ejecutan al llamarlas; sólo se agregan a un plan (``ops``).
  Cuando se pide un resultado (``collect``, ``reduce_by_key``, ``aggregate``...)
  todo el plan se ejecuta *fusionado* en una sola pasada por partición, así no
  se mueven datos intermedios entre procesos.
* **Corrección por tipos**: ``reduce_by_key`` y ``aggregate`` exigen un
  ``Semigroup`` / ``Monoid`` (no una función cualquiera). Así queda explícito
  que la operación debe ser asociativa, que es la condición para que el
  resultado en paralelo sea igual al secuencial.

Flujo de ``reduce_by_key`` (igual al de Dean & Ghemawat, 2008)
--------------------------------------------------------------
1. *split*:   los datos se dividen en ``workers`` bloques.
2. *map + combine*: cada worker ejecuta el plan y reduce localmente por clave.
3. *shuffle*: los parciales se reparten por hash de la clave.
4. *reduce*:  cada worker reduce las claves de su partición.

Paralelismo
-----------
Se usa ``concurrent.futures.ProcessPoolExecutor``. Si alguna función del plan
no se puede serializar con pickle (por ejemplo una ``lambda``) o hay pocos
datos, se ejecuta secuencialmente con exactamente la misma semántica.
"""
from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import chain, islice
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
)

from .monoid import Monoid, Semigroup
from .partitioners import chunk, hash_partition

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
W = TypeVar("W")

DEFAULT_PARALLEL_THRESHOLD = 5_000

# Una operación pendiente del plan: (tipo, función)
Op = Tuple[str, Callable[..., Any]]


# ---------------------------------------------------------------------------
# Ejecución del plan dentro de cada worker (nivel de módulo => pickle)
# ---------------------------------------------------------------------------

def _map_values(fn: Callable[[Any], Any], pairs: Iterable[Tuple[Any, Any]]) -> Iterator[Tuple[Any, Any]]:
    return ((k, fn(v)) for k, v in pairs)


def _key_by(fn: Callable[[Any], Any], items: Iterable[Any]) -> Iterator[Tuple[Any, Any]]:
    return ((fn(x), x) for x in items)


def _apply_ops(ops: Tuple[Op, ...], items: Iterable[Any]) -> Iterator[Any]:
    """Compone las operaciones pendientes como generadores (una sola pasada)."""
    stream: Iterable[Any] = items
    for kind, fn in ops:
        if kind == "map":
            stream = map(fn, stream)
        elif kind == "flat_map":
            stream = chain.from_iterable(map(fn, stream))
        elif kind == "filter":
            stream = filter(fn, stream)
        elif kind == "map_values":
            stream = _map_values(fn, stream)
        elif kind == "key_by":
            stream = _key_by(fn, stream)
        else:  # pragma: no cover - protegido por la API
            raise ValueError(f"operación desconocida: {kind}")
    return iter(stream)


def _combine_pairs(combine: Callable[[V, V], V], pairs: Iterable[Tuple[K, V]]) -> Tuple[Tuple[K, V], ...]:
    acc: Dict[K, V] = {}
    for k, v in pairs:
        acc[k] = combine(acc[k], v) if k in acc else v
    return tuple(acc.items())


def _task_materialize(args: Tuple[Tuple[Op, ...], Tuple[Any, ...]]) -> Tuple[Any, ...]:
    ops, block = args
    return tuple(_apply_ops(ops, block))


def _task_map_combine(
    args: Tuple[Tuple[Tuple[Op, ...], Callable[[Any, Any], Any]], Tuple[Any, ...]]
) -> Tuple[Tuple[Any, Any], ...]:
    """Fase map + combiner local: ejecuta el plan y reduce por clave."""
    (ops, combine), block = args
    return _combine_pairs(combine, _apply_ops(ops, block))


def _task_reduce(
    args: Tuple[Callable[[Any, Any], Any], Tuple[Tuple[Any, Any], ...]]
) -> Tuple[Tuple[Any, Any], ...]:
    combine, block = args
    return _combine_pairs(combine, block)


def _task_fold(args: Tuple[Tuple[Tuple[Op, ...], Monoid[Any]], Tuple[Any, ...]]) -> Any:
    (ops, monoid), block = args
    return monoid.fold(_apply_ops(ops, block))


def _is_picklable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:  # PicklingError, AttributeError, TypeError...
        return False


# ---------------------------------------------------------------------------
# MapReduce
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapReduce(Generic[T]):
    """Colección inmutable y perezosa con operaciones Map-Reduce.

    >>> MapReduce.of(["a b a"]).flat_map(str.split).count()
    3
    """

    source: Tuple[Any, ...] = ()
    ops: Tuple[Op, ...] = ()
    workers: int = 1
    parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD

    # ----- construcción ----------------------------------------------------

    @staticmethod
    def of(
        data: Iterable[T],
        workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ) -> "MapReduce[T]":
        """Crea un MapReduce. ``workers=None`` usa todos los CPUs."""
        n = workers if workers is not None else (os.cpu_count() or 1)
        if n < 1:
            raise ValueError("workers debe ser >= 1")
        return MapReduce(tuple(data), (), n, parallel_threshold)

    def _then(self, kind: str, fn: Callable[..., Any]) -> "MapReduce[Any]":
        """Agrega una operación al plan (no ejecuta nada)."""
        return replace(self, ops=self.ops + ((kind, fn),))

    def _with(self, data: Iterable[U]) -> "MapReduce[U]":
        """Nuevo MapReduce con datos ya materializados y plan vacío."""
        return replace(self, source=tuple(data), ops=())  # type: ignore[return-value]

    # ----- motor de ejecución ---------------------------------------------

    def _run(self, task: Callable[[Tuple[Any, Any]], W], payload: Any, blocks: Sequence[Tuple[Any, ...]]) -> List[W]:
        """Ejecuta ``task`` sobre cada bloque, en paralelo si es posible."""
        jobs = [(payload, b) for b in blocks]
        size = sum(len(b) for b in blocks)
        if self.workers > 1 and size >= self.parallel_threshold and _is_picklable(payload):
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                return list(pool.map(task, jobs))
        return [task(job) for job in jobs]

    def _blocks(self) -> Tuple[Tuple[Any, ...], ...]:
        return chunk(self.source, self.workers)

    @property
    def data(self) -> Tuple[T, ...]:
        """Ejecuta el plan pendiente y devuelve los elementos."""
        if not self.ops:
            return self.source  # type: ignore[return-value]
        parts = self._run(_task_materialize, self.ops, self._blocks())
        return tuple(chain.from_iterable(parts))

    def cache(self) -> "MapReduce[T]":
        """Materializa el plan para reutilizar el resultado varias veces."""
        return self._with(self.data)

    # ----- transformaciones perezosas --------------------------------------

    def map(self, fn: Callable[[T], U]) -> "MapReduce[U]":
        return self._then("map", fn)

    def flat_map(self, fn: Callable[[T], Iterable[U]]) -> "MapReduce[U]":
        return self._then("flat_map", fn)

    def filter(self, predicate: Callable[[T], bool]) -> "MapReduce[T]":
        return self._then("filter", predicate)

    def map_values(self: "MapReduce[Tuple[K, V]]", fn: Callable[[V], W]) -> "MapReduce[Tuple[K, W]]":
        """Aplica ``fn`` sólo al valor de cada par (clave, valor)."""
        return self._then("map_values", fn)

    def key_by(self, key_fn: Callable[[T], K]) -> "MapReduce[Tuple[K, T]]":
        """Convierte cada elemento x en el par (key_fn(x), x)."""
        return self._then("key_by", key_fn)

    # ----- transformaciones con shuffle (ejecutan el plan) ------------------

    def reduce_by_key(self: "MapReduce[Tuple[K, V]]", semigroup: Semigroup[V]) -> "MapReduce[Tuple[K, V]]":
        """Reduce los valores de cada clave con un Semigroup (operación asociativa).

        Implementa split -> map + combine local -> shuffle -> reduce.
        """
        if not isinstance(semigroup, Semigroup):
            raise TypeError(
                "reduce_by_key requiere un Semigroup/Monoid; "
                "envuelve tu función: Semigroup(combine=fn)"
            )
        if not self.source:
            return self._with(())
        combine = semigroup.combine

        # 1-2. split + plan pendiente + combiner local, en cada worker
        partials = self._run(_task_map_combine, (self.ops, combine), self._blocks())
        # 3. shuffle: misma clave -> misma partición
        shuffled = hash_partition(chain.from_iterable(partials), self.workers)
        # 4. reduce final por partición
        reduced = self._run(_task_reduce, combine, shuffled)
        return self._with(_sorted_if_possible(chain.from_iterable(reduced)))

    def group_by_key(self: "MapReduce[Tuple[K, V]]") -> "MapReduce[Tuple[K, Tuple[V, ...]]]":
        """Agrupa los valores de cada clave en una tupla."""
        groups: Dict[K, List[V]] = {}
        for k, v in self.data:
            groups.setdefault(k, []).append(v)
        return self._with(_sorted_if_possible((k, tuple(vs)) for k, vs in groups.items()))

    def sort_by_key(self, descending: bool = False) -> "MapReduce[T]":
        return self._with(sorted(self.data, key=_key, reverse=descending))

    def sort_by_value(self, descending: bool = False) -> "MapReduce[T]":
        return self._with(sorted(self.data, key=_value, reverse=descending))

    def take(self, n: int) -> "MapReduce[T]":
        return self._with(islice(self.data, n))

    # ----- acciones (devuelven un valor) ------------------------------------

    def aggregate(self, monoid: Monoid[T]) -> T:
        """Reduce TODA la colección con un Monoid, en paralelo por bloques."""
        if not isinstance(monoid, Monoid):
            raise TypeError("aggregate requiere un Monoid")
        partials = self._run(_task_fold, (self.ops, monoid), self._blocks())
        return monoid.fold(partials)

    def count(self) -> int:
        return len(self.data)

    def collect(self) -> List[T]:
        return list(self.data)

    def to_dict(self: "MapReduce[Tuple[K, V]]") -> Dict[K, V]:
        return dict(self.data)


def _key(pair: Tuple[Any, Any]) -> Any:
    return pair[0]


def _value(pair: Tuple[Any, Any]) -> Any:
    return pair[1]


def _sorted_if_possible(pairs: Iterable[Tuple[K, V]]) -> List[Tuple[K, V]]:
    """Ordena por clave para que el resultado sea determinista.

    Si las claves no son comparables entre sí, se devuelven en el orden dado.
    """
    items = list(pairs)
    try:
        return sorted(items, key=_key)
    except TypeError:
        return items
