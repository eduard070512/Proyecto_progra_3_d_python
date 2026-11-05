"""
Genera datos de ejemplo reproducibles (semilla fija).

    python data/generate_samples.py

Crea en data/samples/:
  ventas_centro.csv, ventas_norte.csv, ventas_sur.csv   (3 sucursales)
  catalogo.csv                                          (productos y costos)
  logs_servidor1.csv, logs_servidor2.csv                (logs web)
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent / "samples"

PRODUCTOS = {
    # producto: (categoria, precio, costo)
    "laptop": ("electronica", 15000, 11500),
    "mouse": ("electronica", 350, 180),
    "teclado": ("electronica", 800, 450),
    "monitor": ("electronica", 4200, 3100),
    "cuaderno": ("papeleria", 45, 20),
    "pluma": ("papeleria", 15, 5),
    "mochila": ("accesorios", 650, 380),
    "termo": ("accesorios", 280, 120),
    "cafe": ("alimentos", 120, 70),
    "galletas": ("alimentos", 35, 18),
}
SUCURSALES = {"centro": 1.3, "norte": 1.0, "sur": 0.7}  # factor de demanda
ENDPOINTS = ["/", "/productos", "/carrito", "/pago", "/login", "/api/stock"]


def ventas(rng: random.Random) -> None:
    inicio = datetime(2025, 8, 1)
    nombres = list(PRODUCTOS)
    pesos = [6, 20, 12, 4, 30, 40, 8, 10, 25, 30]
    for sucursal, factor in SUCURSALES.items():
        filas = []
        for i in range(int(1500 * factor)):
            prod = rng.choices(nombres, weights=pesos)[0]
            _cat, precio, _costo = PRODUCTOS[prod]
            fecha = inicio + timedelta(minutes=rng.randint(0, 60 * 24 * 120))
            filas.append({
                "id": f"{sucursal[:1].upper()}{i:05d}",
                "fecha": fecha.strftime("%Y-%m-%d %H:%M"),
                "producto": prod,
                "cantidad": rng.choices([1, 2, 3, 5], weights=[70, 20, 7, 3])[0],
                "precio_unitario": round(precio * rng.uniform(0.9, 1.05), 2),
            })
        filas.sort(key=lambda f: f["fecha"])
        # algunas filas dañadas para demostrar el manejo de errores
        filas[10]["cantidad"] = "N/A"
        escribir(f"ventas_{sucursal}.csv", filas)


def catalogo() -> None:
    escribir("catalogo.csv", [
        {"producto": p, "categoria": c, "costo": costo}
        for p, (c, _precio, costo) in PRODUCTOS.items()
    ])


def logs(rng: random.Random) -> None:
    inicio = datetime(2025, 11, 1, 9, 0)
    for servidor, error_rate in (("servidor1", 0.02), ("servidor2", 0.08)):
        filas = []
        for _ in range(3000):
            ep = rng.choice(ENDPOINTS)
            lento = ep in ("/pago", "/api/stock")
            latencia = rng.lognormvariate(4.6 if lento else 3.8, 0.5)
            status = 500 if rng.random() < error_rate else rng.choices([200, 304, 404], weights=[85, 10, 5])[0]
            ts = inicio + timedelta(seconds=rng.randint(0, 3 * 3600))
            filas.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "ip": f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 30)}",
                "endpoint": ep,
                "status": status,
                "latencia_ms": round(latencia, 1),
            })
        filas.sort(key=lambda f: f["timestamp"])
        escribir(f"logs_{servidor}.csv", filas)


def escribir(nombre: str, filas: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / nombre, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"  {nombre}: {len(filas)} filas")


if __name__ == "__main__":
    rng = random.Random(2025)
    ventas(rng)
    catalogo()
    logs(rng)
