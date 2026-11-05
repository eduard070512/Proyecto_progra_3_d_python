# Resultados de benchmarks

Máquina: Linux x86_64, 4 CPUs, Python 3.13.18.
Cada tiempo es el mejor de varias repeticiones. En todos los casos se verificó
que el resultado paralelo es igual al secuencial (con tolerancia 1e-9 en floats,
porque la suma de punto flotante no es exactamente asociativa).

| Carga | n | Workers | Tiempo (s) | Speedup |
|---|---:|---:|---:|---:|
| conteo de palabras | 100,000 | 1 | 0.439 | 1.0 |
| conteo de palabras | 100,000 | 2 | 0.321 | 1.37 |
| conteo de palabras | 100,000 | 4 | 0.224 | 1.96 |
| conteo (Counter secuencial) | 100,000 | 1 | 0.264 | - |
| conteo de palabras | 200,000 | 1 | 0.692 | 1.0 |
| conteo de palabras | 200,000 | 2 | 0.644 | 1.07 |
| conteo de palabras | 200,000 | 4 | 0.403 | 1.72 |
| conteo (Counter secuencial) | 200,000 | 1 | 0.454 | - |
| estadísticas por categoría | 200,000 | 1 | 0.589 | 1.0 |
| estadísticas por categoría | 200,000 | 2 | 0.491 | 1.2 |
| estadísticas por categoría | 200,000 | 4 | 0.407 | 1.45 |
| estadísticas por categoría | 400,000 | 1 | 1.175 | 1.0 |
| estadísticas por categoría | 400,000 | 2 | 0.987 | 1.19 |
| estadísticas por categoría | 400,000 | 4 | 0.606 | 1.94 |
| percentiles por categoría | 200,000 | 1 | 0.415 | 1.0 |
| percentiles por categoría | 200,000 | 2 | 0.503 | 0.83 |
| percentiles por categoría | 200,000 | 4 | 0.382 | 1.09 |
| percentiles por categoría | 400,000 | 1 | 0.785 | 1.0 |
| percentiles por categoría | 400,000 | 2 | 0.908 | 0.86 |
| percentiles por categoría | 400,000 | 4 | 0.707 | 1.11 |
