"""
Benchmarks de escalabilidad (Semana 4).

    python -m benchmarks.benchmark            # completo (~1 min)
    python -m benchmarks.benchmark --rapido   # versión corta

Mide 3 cargas de trabajo con 1, 2, 4 (y hasta cpu_count) workers y distintos
tamaños de datos. Guarda los resultados en benchmarks/resultados.md y
benchmarks/resultados.csv. Si matplotlib está instalado también genera
benchmarks/speedup.png.
"""
from __future__ import annotations

import csv
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from src import MapReduce, SumMonoid, combine, count, describe, quantiles_of, sum_of

HERE = Path(__file__).resolve().parent
VOCAB = [f"palabra{i}" for i in range(2_000)]
CATS = [f"cat{i}" for i in range(50)]


# ---------------------------------------------------------------- datos

def lineas(n: int) -> list[str]:
    rng = random.Random(1)
    return [" ".join(rng.choices(VOCAB, k=15)) for _ in range(n)]


def registros(n: int) -> list[dict]:
    rng = random.Random(2)
    return [{"cat": rng.choice(CATS), "monto": rng.uniform(1, 1000)} for _ in range(n)]


# ---------------------------------------------------- funciones puras (pickle)

def tokenizar(linea: str) -> list[str]:
    return linea.split()


def par_uno(p: str) -> tuple[str, int]:
    return (p, 1)


ESTADISTICAS = combine(n=count(), total=sum_of("monto"), resumen=describe("monto"))
PERCENTILES = quantiles_of("monto", qs=(0.5, 0.95), lo=0, hi=1000, bins=1000)


# ---------------------------------------------------------- cargas de trabajo

def wc(data: list, workers: int):
    return (MapReduce.of(data, workers=workers, parallel_threshold=0)
            .flat_map(tokenizar).map(par_uno).reduce_by_key(SumMonoid).collect())


def stats(data: list, workers: int):
    return MapReduce.of(data, workers=workers, parallel_threshold=0).aggregate_by_key("cat", ESTADISTICAS).collect()


def pct(data: list, workers: int):
    return MapReduce.of(data, workers=workers, parallel_threshold=0).aggregate_by_key("cat", PERCENTILES).collect()


def wc_python_puro(data: list) -> list:
    """Referencia: implementación secuencial idiomática con Counter."""
    return sorted(Counter(w for l in data for w in l.split()).items())


def casi_iguales(a: object, b: object, tol: float = 1e-9) -> bool:
    """Igualdad con tolerancia para floats.

    La suma de punto flotante NO es exactamente asociativa: (a+b)+c puede
    diferir de a+(b+c) en el último bit. Al cambiar el número de workers
    cambia el orden de las sumas, así que comparamos con tolerancia relativa.
    """
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(casi_iguales(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(casi_iguales(x, y, tol) for x, y in zip(a, b))
    return a == b


def medir(fn: Callable[[], object], repeticiones: int) -> float:
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        tiempos.append(time.perf_counter() - t0)
    return min(tiempos)  # el mínimo es el menos afectado por ruido del sistema


def main(rapido: bool) -> None:
    cpus = os.cpu_count() or 1
    workers_list = sorted({1, 2, 4, cpus} if cpus <= 8 else {1, 2, 4, 8})
    escala = 1 if rapido else 4
    reps = 1 if rapido else 3
    cargas = [
        ("conteo de palabras", wc, lineas, [25_000 * escala, 50_000 * escala]),
        ("estadísticas por categoría", stats, registros, [50_000 * escala, 100_000 * escala]),
        ("percentiles por categoría", pct, registros, [50_000 * escala, 100_000 * escala]),
    ]

    filas = []
    for nombre, fn, generar, tamanos in cargas:
        for n in tamanos:
            data = generar(n)
            esperado = fn(data, 1)
            base = None
            for w in workers_list:
                assert casi_iguales(fn(data, w), esperado), "¡el resultado paralelo difiere del secuencial!"
                t = medir(lambda: fn(data, w), reps)
                base = base or t
                filas.append({"carga": nombre, "n": n, "workers": w, "segundos": round(t, 3), "speedup": round(base / t, 2)})
                print(f"{nombre:<28} n={n:>8,} workers={w}  {t:7.3f}s  speedup {base / t:4.2f}x")
            if fn is wc:
                t = medir(lambda: wc_python_puro(data), reps)
                filas.append({"carga": "conteo (Counter secuencial)", "n": n, "workers": 1, "segundos": round(t, 3), "speedup": None})
                print(f"{'  referencia Counter':<28} n={n:>8,}            {t:7.3f}s")

    guardar(filas, cpus)


def guardar(filas: list[dict], cpus: int) -> None:
    with open(HERE / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    md = [
        "# Resultados de benchmarks",
        "",
        f"Máquina: {platform.system()} {platform.machine()}, {cpus} CPUs, Python {platform.python_version()}.",
        "Cada tiempo es el mejor de varias repeticiones. En todos los casos se verificó",
        "que el resultado paralelo es igual al secuencial (con tolerancia 1e-9 en floats,",
        "porque la suma de punto flotante no es exactamente asociativa).",
        "",
        "| Carga | n | Workers | Tiempo (s) | Speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    md += [f"| {r['carga']} | {r['n']:,} | {r['workers']} | {r['segundos']} | {r['speedup'] or '-'} |" for r in filas]
    (HERE / "resultados.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nGuardado en {HERE / 'resultados.md'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(instala matplotlib para generar la gráfica speedup.png)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series: dict[tuple, list] = {}
    for r in filas:
        if r["speedup"] is not None:
            series.setdefault((r["carga"], r["n"]), []).append((r["workers"], r["speedup"]))
    for (carga, n), pts in series.items():
        ax.plot(*zip(*pts), marker="o", label=f"{carga} (n={n:,})")
    ws = sorted({r["workers"] for r in filas})
    ax.plot(ws, ws, "k--", alpha=0.4, label="ideal (lineal)")
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup")
    ax.set_xticks(ws)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Escalabilidad del Map-Reduce funcional")
    fig.tight_layout()
    fig.savefig(HERE / "speedup.png", dpi=150)
    print(f"Gráfica en {HERE / 'speedup.png'}")


if __name__ == "__main__":
    main(rapido="--rapido" in sys.argv)
