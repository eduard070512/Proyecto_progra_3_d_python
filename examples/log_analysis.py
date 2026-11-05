"""
Caso de uso: monitoreo de servidores web a partir de sus logs.

Une los logs de 2 servidores y calcula tasa de errores, latencias p50/p95 por
endpoint (percentiles distribuidos) y tráfico por ventanas de 15 minutos.

    python data/generate_samples.py         # sólo la primera vez
    python -m examples.log_analysis
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from src import Record, combine, count, max_of, mean_of, quantiles_of, read_csv_sources, sum_of

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
SCHEMA = {"status": int, "latencia_ms": float}
VENTANA_SEG = 15 * 60


def epoch(r: Record) -> float:
    return datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()


def es_error(r: Record) -> bool:
    return r["status"] >= 500


def es_error_num(r: Record) -> int:
    return 1 if r["status"] >= 500 else 0


def ventana(r: Record) -> float:
    """Inicio de la ventana de tiempo (tumbling window) a la que pertenece r."""
    t = epoch(r)
    return t - t % VENTANA_SEG


def tasa(d: dict) -> dict:
    return {**d, "tasa_error": d["errores"] / d["peticiones"] if d["peticiones"] else 0.0}


def main() -> None:
    archivos = sorted(SAMPLES.glob("logs_*.csv"))
    if not archivos:
        sys.exit("No hay datos. Ejecuta primero: python data/generate_samples.py")

    logs = read_csv_sources(archivos, SCHEMA, parallel_threshold=1_000).data.cache()

    salud = combine(
        peticiones=count(),
        errores=sum_of(es_error_num),
        latencia_media=mean_of("latencia_ms"),
        latencia_max=max_of("latencia_ms"),
    ).map(tasa)

    print("Salud por servidor:")
    for srv, v in logs.aggregate_by_key("source", salud).collect():
        print(f"  {srv:<16} {v['peticiones']:>5} peticiones  errores {v['tasa_error']:6.2%}  "
              f"latencia media {v['latencia_media']:6.1f} ms  máx {v['latencia_max']:7.1f} ms")

    print("\nLatencia por endpoint (percentiles distribuidos):")
    percentiles = quantiles_of("latencia_ms", qs=(0.5, 0.95), lo=0, hi=1_000, bins=2_000)
    for ep, q in logs.aggregate_by_key("endpoint", percentiles).collect():
        alerta = "  <-- lento" if q[0.95] > 200 else ""
        print(f"  {ep:<12} p50 {q[0.5]:6.1f} ms   p95 {q[0.95]:6.1f} ms{alerta}")

    print(f"\nPeticiones y errores por ventana de {VENTANA_SEG // 60} minutos:")
    trafico = combine(peticiones=count(), errores=sum_of(es_error_num))
    for inicio, v in logs.aggregate_by_key(ventana, trafico).collect():
        hora = datetime.fromtimestamp(inicio).strftime("%H:%M")
        errores = int(v["errores"])
        print(f"  {hora}  {v['peticiones']:>4} peticiones  {'#' * errores} {errores} errores")

    print("\nIPs con más errores 5xx:")
    top_ips = (
        logs.filter(es_error)
        .aggregate_by_key("ip", count())
        .sort_by_value(descending=True)
        .take(5)
        .collect()
    )
    for ip, n in top_ips:
        print(f"  {ip:<14} {n}")


if __name__ == "__main__":
    main()
