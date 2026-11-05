"""
Caso de uso: inteligencia de negocios para una cadena de tiendas.

Agrega las ventas de 3 sucursales (3 fuentes distintas), las cruza con el
catálogo de productos (join distribuido) y genera un dashboard HTML.

    python data/generate_samples.py         # sólo la primera vez
    python -m examples.sales_analytics
"""
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

from src import (
    MapReduce,
    Record,
    combine,
    count,
    describe,
    mean_of,
    quantiles_of,
    read_csv_sources,
    sum_of,
    top_k,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
OUTPUT = ROOT / "data" / "output"

SCHEMA_VENTAS = {"cantidad": int, "precio_unitario": float}
SCHEMA_CATALOGO = {"costo": float}


# ---- funciones puras de nivel de módulo (paralelizables) --------------------

def con_importe(r: Record) -> Record:
    return r.with_(importe=round(r["cantidad"] * r["precio_unitario"], 2))


def mes(r: Record) -> str:
    return r["fecha"][:7]


def producto_y_venta(r: Record) -> tuple[str, Record]:
    return (r["producto"], r)


def producto_y_catalogo(r: Record) -> tuple[str, Record]:
    return (r["producto"], r)


def con_margen(par: tuple[str, tuple[Record, Record]]) -> Record:
    _prod, (venta, cat) = par
    costo = venta["cantidad"] * cat["costo"]
    return venta.with_(categoria=cat["categoria"], margen=round(venta["importe"] - costo, 2))


def etiqueta_venta(r: Record) -> str:
    return f'{r["id"]} ({r["source"]}, {r["producto"]} x{r["cantidad"]})'


# ---- análisis -------------------------------------------------------------

def analizar(workers: int | None = None) -> dict[str, Any]:
    archivos = sorted(SAMPLES.glob("ventas_*.csv"))
    if not archivos:
        sys.exit("No hay datos. Ejecuta primero: python data/generate_samples.py")

    cargado = read_csv_sources(archivos, SCHEMA_VENTAS, workers=workers, parallel_threshold=1_000)
    ventas = cargado.data.map(con_importe).cache()

    catalogo = read_csv_sources([SAMPLES / "catalogo.csv"], SCHEMA_CATALOGO, workers=1).data

    por_sucursal = combine(
        ventas=count(),
        ingreso=sum_of("importe"),
        ticket_promedio=mean_of("importe"),
    )

    enriquecidas = (
        ventas.map(producto_y_venta)
        .join(catalogo.map(producto_y_catalogo))
        .map(con_margen)
        .cache()
    )

    return {
        "errores": cargado.errors,
        "resumen": ventas.aggregate_with(describe("importe")),
        "cuantiles": ventas.aggregate_with(
            quantiles_of("importe", qs=(0.5, 0.9, 0.99), lo=0, hi=80_000, bins=8_000)
        ),
        "sucursales": ventas.aggregate_by_key("source", por_sucursal).collect(),
        "categorias": enriquecidas.aggregate_by_key(
            "categoria", combine(ingreso=sum_of("importe"), margen=sum_of("margen"))
        ).collect(),
        "meses": ventas.aggregate_by_key(mes, sum_of("importe")).collect(),
        "top": ventas.aggregate_with(top_k(5, "importe", etiqueta_venta)),
    }


# ---- presentación -----------------------------------------------------------

def dinero(x: float | None) -> str:
    return "-" if x is None else f"${x:,.2f}"


def imprimir(r: dict[str, Any]) -> None:
    s = r["resumen"]
    print(f"Ventas: {s['n']:,}   Ingreso: {dinero(s['total'])}   "
          f"Ticket medio: {dinero(s['mean'])}  (σ = {dinero(s['std'])})")
    q = r["cuantiles"]
    print(f"Mediana: {dinero(q[0.5])}   p90: {dinero(q[0.9])}   p99: {dinero(q[0.99])}")
    print(f"Filas con error ignoradas: {len(r['errores'])}")
    for e in r["errores"]:
        print(f"   {e.source} línea {e.line}: {e.error}")

    print("\nPor sucursal:")
    for nombre, v in r["sucursales"]:
        print(f"  {nombre:<15} {v['ventas']:>6,} ventas  {dinero(v['ingreso']):>16}  ticket {dinero(v['ticket_promedio'])}")

    print("\nPor categoría (join con catálogo):")
    for cat, v in sorted(r["categorias"], key=lambda kv: -kv[1]["ingreso"]):
        pct = 100 * v["margen"] / v["ingreso"]
        print(f"  {cat:<12} ingreso {dinero(v['ingreso']):>16}  margen {dinero(v['margen']):>14} ({pct:4.1f}%)")

    print("\nPor mes:")
    for m, total in r["meses"]:
        print(f"  {m}  {dinero(total):>16}")

    print("\nTop 5 ventas:")
    for importe, etiqueta in r["top"]:
        print(f"  {dinero(importe):>12}  {etiqueta}")


def barras(filas: list[tuple[str, float]]) -> str:
    maximo = max((v for _, v in filas), default=1) or 1
    return "".join(
        f'<div class="row"><span class="lbl">{html.escape(str(k))}</span>'
        f'<span class="bar" style="width:{70 * v / maximo:.1f}%"></span>'
        f'<span class="val">{dinero(v)}</span></div>'
        for k, v in filas
    )


def dashboard(r: dict[str, Any]) -> Path:
    s = r["resumen"]
    kpis = [
        ("Ventas", f"{s['n']:,}"),
        ("Ingreso total", dinero(s["total"])),
        ("Ticket promedio", dinero(s["mean"])),
        ("Mediana", dinero(r["cuantiles"][0.5])),
    ]
    categorias = sorted(r["categorias"], key=lambda kv: -kv[1]["ingreso"])
    doc = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard de ventas</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:0;padding:24px;background:#f6f7f9;color:#1d2433}}
 h1{{margin:0 0 4px}} p.sub{{color:#5b6475;margin:0 0 20px}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}}
 .kpi,.card{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 3px #0001}}
 .kpi b{{display:block;font-size:1.4em;margin-top:4px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
 .row{{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:.9em}}
 .lbl{{width:90px;flex:none}} .bar{{height:14px;background:#3b82f6;border-radius:3px}}
 .val{{color:#5b6475;white-space:nowrap}}
 table{{width:100%;border-collapse:collapse;font-size:.9em}} td,th{{padding:6px;border-bottom:1px solid #eee;text-align:left}}
</style></head><body>
<h1>Dashboard de ventas</h1>
<p class="sub">{len(r['sucursales'])} sucursales · generado con Map-Reduce funcional</p>
<div class="kpis">{''.join(f'<div class="kpi">{k}<b>{v}</b></div>' for k, v in kpis)}</div>
<div class="grid">
 <div class="card"><h3>Ingreso por sucursal</h3>{barras([(k, v['ingreso']) for k, v in r['sucursales']])}</div>
 <div class="card"><h3>Ingreso por mes</h3>{barras(r['meses'])}</div>
 <div class="card"><h3>Margen por categoría</h3>{barras([(k, v['margen']) for k, v in categorias])}</div>
 <div class="card"><h3>Top 5 ventas</h3><table>
  {''.join(f'<tr><td>{dinero(i)}</td><td>{html.escape(e)}</td></tr>' for i, e in r['top'])}
 </table></div>
</div></body></html>"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destino = OUTPUT / "dashboard_ventas.html"
    destino.write_text(doc, encoding="utf-8")
    return destino


if __name__ == "__main__":
    resultado = analizar()
    imprimir(resultado)
    print(f"\nDashboard: {dashboard(resultado)}")
