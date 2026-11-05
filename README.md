# Proyecto 7: Sistema de Agregación y Análisis de Datos Distribuidos

Sistema funcional en Python que agrega y analiza datos usando el patrón
**Map-Reduce** en paradigma funcional puro: datos inmutables, funciones puras,
evaluación perezosa y agregación basada en **Monoids y Semigroups**.

Materia: Programación Funcional, Universidad de Colima.

## Estado

| Semana | Entregable | Estado |
|---|---|---|
| 1 | Framework básico de Map-Reduce funcional | ✅ |
| 2 | Operadores de agregación composables | ⏳ |
| 3 | Sistema de procesamiento paralelo | ⏳ (base ya implementada) |
| 4 | Casos de uso y benchmarks | ⏳ |

## Instalación y uso

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

pytest -v                                  # pruebas
python -m examples.word_count              # conteo de palabras
python -m examples.word_count --bench      # comparación 1, 2 y 4 workers
```

Requiere Python 3.10 o superior. El núcleo usa solo la biblioteca estándar.

```python
from src import MapReduce, SumMonoid

def tokenize(line): return line.lower().split()
def to_pair(word):  return (word, 1)

conteo = (
    MapReduce.of(lineas, workers=4)
    .flat_map(tokenize)          # perezoso: aún no se ejecuta
    .map(to_pair)                # perezoso
    .reduce_by_key(SumMonoid)    # aquí se ejecuta todo, en paralelo
    .sort_by_value(descending=True)
    .take(10)
    .collect()
)
```

## Estructura

```
src/
  monoid.py        Semigroup, Monoid, instancias (Sum, Count, Min, Max, Avg...) y product_monoid
  partitioners.py  chunk, hash_partition (shuffle), range_partition
  mapreduce.py     MapReduce inmutable y perezoso con ejecución paralela
examples/
  word_count.py    Ejemplo de la semana 1 y benchmark
tests/             45 pruebas (leyes de monoid, inmutabilidad, laziness, paralelo == secuencial)
data/samples/      Datos de ejemplo
```

## Diseño

### 1. Monoids y Semigroups

Un **Semigroup** es una operación `combine` **asociativa**. Un **Monoid** añade
un **elemento neutro** `empty`.

La asociatividad es la propiedad que permite el procesamiento distribuido:

```
combine(combine(a, b), c) == combine(a, combine(b, c))
```

Gracias a ella, cada worker puede reducir su partición por separado y luego
combinar los resultados parciales; el resultado es idéntico al secuencial. El
neutro resuelve las particiones vacías.

El promedio **no** es un monoid, pero el par `(suma, cantidad)` sí lo es
(`AvgMonoid`). El promedio se calcula al final con `suma / cantidad`.

`product_monoid(total=SumMonoid, top=MaxMonoid)` combina varios monoids en uno
y calcula varias agregaciones en una sola pasada.

### 2. Corrección por tipos

`reduce_by_key` y `aggregate` **solo aceptan** `Semigroup`/`Monoid`. Una función
binaria cualquiera lanza `TypeError`. Así, quien la usa declara de forma
explícita que su operación es asociativa. Las pruebas verifican las leyes de
cada instancia (`check_associativity`, `check_identity`).

### 3. Inmutabilidad

`MapReduce` y `Monoid` son `@dataclass(frozen=True)` y almacenan tuplas. Cada
operación devuelve un objeto nuevo y la entrada nunca se modifica. Sin estado
compartido mutable, los procesos no pueden tener condiciones de carrera.

### 4. Evaluación perezosa y fusión

`map`, `flat_map`, `filter`, `map_values` y `key_by` solo agregan un paso al
plan (`ops`). Al pedir un resultado, el plan completo se ejecuta como una
cadena de generadores **en una sola pasada por partición**. Los datos
intermedios no viajan entre procesos.

### 5. Flujo de `reduce_by_key`

```
datos ─ split ─► [worker 1: plan + combiner local] ─┐
              └► [worker 2: plan + combiner local] ─┤ shuffle por hash(clave)
              └► [worker N: plan + combiner local] ─┘
                    ─► [reduce partición 1] ... [reduce partición N] ─► resultado
```

### 6. Paralelismo

Se usa `concurrent.futures.ProcessPoolExecutor`. Las funciones deben estar
definidas a nivel de módulo para que `pickle` las pueda enviar a otros
procesos. Si el plan contiene una `lambda` o hay pocos datos (menos de
`parallel_threshold`), se ejecuta secuencialmente con el mismo resultado.

Benchmark de conteo de palabras con 200,000 líneas en una máquina de 4 núcleos
(`python -m examples.word_count --bench`):

| workers | tiempo | speedup |
|---|---|---|
| 1 | 1.00 s | 1.00x |
| 2 | 0.75 s | 1.35x |
| 4 | 0.44 s | 2.26x |

## Componente de emprendimiento

Servicio de análisis de datos para PyMEs: dashboards de ventas e inteligencia
de negocio a partir de varias fuentes (tiendas, e-commerce, inventario),
procesados en paralelo en hardware propio sin pagar plataformas cloud de big
data.

## Referencias

1. Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified data processing on
   large clusters. *Communications of the ACM, 51*(1), 107–113.
   https://doi.org/10.1145/1327452.1327492
2. Yu, Y., Gunda, P. K., & Isard, M. (2009). Distributed aggregation for
   data-parallel computing: Interfaces and implementations. *Proceedings of the
   ACM SIGOPS 22nd Symposium on Operating Systems Principles (SOSP '09)*,
   247–260. https://doi.org/10.1145/1629575.1629600
3. Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data
   processing. *Communications of the ACM, 59*(11), 56–65.
   https://doi.org/10.1145/2934664
