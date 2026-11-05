"""
Lectura de múltiples fuentes de datos (Semana 3).

Cada fuente (archivo CSV, por ejemplo una sucursal o un servidor) se lee en un
proceso distinto. Las filas se convierten con un *esquema* (funciones puras de
conversión por columna) y se etiquetan con el nombre de su fuente.
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from .mapreduce import DEFAULT_PARALLEL_THRESHOLD, MapReduce

Schema = Mapping[str, Callable[[str], Any]]


class Record(dict):
    """Diccionario inmutable y serializable (pickle) para representar filas."""

    def _blocked(self, *_: Any, **__: Any) -> None:
        raise TypeError("Record es inmutable")

    __setitem__ = __delitem__ = _blocked  # type: ignore[assignment]
    clear = pop = popitem = setdefault = update = _blocked  # type: ignore[assignment]
    __ior__ = _blocked  # type: ignore[assignment]

    def __reduce__(self) -> Any:
        return (Record, (dict(self),))

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted(self.items(), key=lambda kv: kv[0])))

    def with_(self, **changes: Any) -> "Record":
        """Copia con cambios (la forma funcional de 'modificar')."""
        return Record({**self, **changes})


@dataclass(frozen=True)
class ParseError:
    """Fila que no se pudo convertir. Se conserva como dato, no como excepción."""

    source: str
    line: int
    error: str


def _read_one(args: Tuple[str, Tuple[Tuple[str, Callable[[str], Any]], ...]]) -> Tuple[Tuple[Any, ...], Tuple[ParseError, ...]]:
    path, schema_items = args
    schema = dict(schema_items)
    name = Path(path).stem
    rows: list[Any] = []
    errors: list[ParseError] = []
    with open(path, newline="", encoding="utf-8") as f:
        for line_no, raw in enumerate(csv.DictReader(f), start=2):
            try:
                row = {k: schema.get(k, str)(v) for k, v in raw.items()}
            except (ValueError, TypeError) as exc:
                errors.append(ParseError(name, line_no, str(exc)))
                continue
            row["source"] = name
            rows.append(Record(row))
    return tuple(rows), tuple(errors)


@dataclass(frozen=True)
class LoadResult:
    data: MapReduce[Any]
    errors: Tuple[ParseError, ...]


def read_csv_sources(
    paths: Iterable[str | os.PathLike[str]],
    schema: Optional[Schema] = None,
    workers: Optional[int] = None,
    parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
) -> LoadResult:
    """Lee varios CSV en paralelo y los une en un solo ``MapReduce``.

    Las filas son ``Record`` (diccionarios inmutables). Las filas
    inválidas no detienen el proceso: se devuelven en ``errors``.
    """
    files: Sequence[str] = [str(p) for p in paths]
    if not files:
        raise ValueError("se necesita al menos un archivo")
    schema_items = tuple((schema or {}).items())
    jobs = [(p, schema_items) for p in files]
    n = workers if workers is not None else (os.cpu_count() or 1)
    if n > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=min(n, len(files))) as pool:
            results = list(pool.map(_read_one, jobs))
    else:
        results = [_read_one(j) for j in jobs]
    rows = tuple(r for part, _ in results for r in part)
    errors = tuple(e for _, errs in results for e in errs)
    return LoadResult(MapReduce.of(rows, workers=n, parallel_threshold=parallel_threshold), errors)
