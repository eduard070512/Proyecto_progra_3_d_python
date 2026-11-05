"""
Ejemplo de la Semana 1: conteo de palabras con Map-Reduce funcional.

Uso:
    python -m examples.word_count                 # archivo de ejemplo
    python -m examples.word_count ruta.txt        # tu propio archivo
    python -m examples.word_count --bench         # compara 1, 2 y 4 workers
"""
from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

from src import MapReduce, SumMonoid

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_text.txt"
WORD = re.compile(r"\w+", re.UNICODE)


# Funciones puras de nivel de módulo (serializables => paralelizables)
def tokenize(line: str) -> list[str]:
    return WORD.findall(line.lower())


def to_pair(word: str) -> tuple[str, int]:
    return (word, 1)


def is_meaningful(word: str) -> bool:
    return len(word) > 3


def word_count(lines, workers: int = 4, top: int = 10, threshold: int = 5_000):
    return (
        MapReduce.of(lines, workers=workers, parallel_threshold=threshold)
        .flat_map(tokenize)
        .filter(is_meaningful)
        .map(to_pair)
        .reduce_by_key(SumMonoid)
        .sort_by_value(descending=True)
        .take(top)
        .collect()
    )


def synthetic_lines(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    vocab = tokenize(SAMPLE.read_text(encoding="utf-8"))
    return [" ".join(rng.choice(vocab) for _ in range(12)) for _ in range(n)]


def bench() -> None:
    lines = synthetic_lines(200_000)
    print(f"{len(lines):,} líneas")
    baseline = None
    for workers in (1, 2, 4):
        start = time.perf_counter()
        result = word_count(lines, workers=workers, threshold=0)
        elapsed = time.perf_counter() - start
        baseline = baseline or elapsed
        print(f"workers={workers}: {elapsed:6.2f}s  speedup={baseline / elapsed:4.2f}x  top={result[0]}")


def main(argv: list[str]) -> None:
    if "--bench" in argv:
        bench()
        return
    path = Path(argv[0]) if argv else SAMPLE
    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"Top palabras en {path.name}:")
    for word, n in word_count(lines):
        print(f"  {word:<15} {n}")


if __name__ == "__main__":
    main(sys.argv[1:])
