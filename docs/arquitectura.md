# Arquitectura y decisiones de diseño

## Vista general

```
             fuentes (CSV de sucursales / servidores)
                         │  read_csv_sources: 1 proceso por archivo
                         ▼
     ┌──────────────── MapReduce (inmutable, perezoso) ────────────────┐
     │ source: tupla de Records    ops: plan pendiente (map/filter...) │
     └─────────────────────────────────────────────────────────────────┘
          │ acción: collect / aggregate / reduce_by_key / join
          ▼
   split en N bloques ──► worker i: ejecuta el plan fusionado + combiner local
                                   │
                         shuffle por hash(clave)
                                   ▼
                        worker j: reduce final con el Monoid
                                   ▼
                       resultado (tupla ordenada por clave)
```

| Módulo | Responsabilidad |
|---|---|
| `src/monoid.py` | Abstracciones `Semigroup` y `Monoid`, instancias, `product_monoid`, verificación de leyes |
| `src/partitioners.py` | `chunk`, `hash_partition` (shuffle determinista con CRC32), `range_partition` |
| `src/mapreduce.py` | Motor: plan perezoso, ejecución paralela, `reduce_by_key`, `join`, `window`, `aggregate_by_key` |
| `src/aggregators.py` | Agregadores composables: count, sum, mean, min, max, varianza, desviación, percentiles, top-k, describe |
| `src/sources.py` | Lectura paralela de varias fuentes CSV, `Record` inmutable, errores como datos |

## 1. Por qué Monoids y Semigroups

Map-Reduce reparte los datos entre workers y luego junta los resultados
parciales. Eso solo es correcto si la operación de combinación es
**asociativa** (Semigroup):

```
combine(combine(a, b), c) == combine(a, combine(b, c))
```

Si además existe un **neutro** (Monoid), las particiones vacías se manejan sin
casos especiales.

| Monoid | Representación | combine | neutro |
|---|---|---|---|
| Sum / Count | número | `+` | 0 |
| Min / Max | número | `min` / `max` | +∞ / −∞ |
| Avg | `(suma, n)` | suma por componente | `(0, 0)` |
| Moments (varianza) | `(n, media, M2)` | fórmula paralela de Chan et al. | `(0, 0, 0)` |
| Histogram (percentiles) | `((bin, conteo), ...)` disperso | merge de listas ordenadas | `()` |
| Top-K | `k` mejores | `nlargest(k, a + b)` | `()` |
| Producto | diccionario de monoids | campo por campo | diccionario de neutros |

Idea clave: **promedio, varianza y percentiles NO son asociativos**, pero sí lo
es un *resumen* intermedio de ellos. El `Aggregator` separa la agregación en
tres pasos: `prepare` (registro → resumen), `monoid` (combinar resúmenes) y
`present` (resumen → resultado final).

## 2. Corrección por tipos

* `reduce_by_key` acepta solo `Semigroup` y `aggregate` acepta solo `Monoid`.
  Pasar una función cualquiera produce `TypeError`. Quien la usa declara así,
  de forma explícita, que su operación cumple las leyes.
* Las anotaciones genéricas (`MapReduce[T]`, `Monoid[A]`,
  `Aggregator[T, A, R]`) documentan qué tipo entra y sale de cada etapa.
* Las leyes de cada instancia se verifican en `tests/test_monoid.py` y
  `tests/test_aggregators.py`.

## 3. Inmutabilidad

* `MapReduce`, `Monoid`, `Aggregator` y todos los auxiliares son
  `@dataclass(frozen=True)`.
* Los datos se guardan como tuplas y las filas como `Record`, un diccionario
  que rechaza cualquier modificación. Para "modificar" se usa `with_`, que
  devuelve una copia.
* Sin estado compartido mutable no puede haber condiciones de carrera entre
  procesos.
* Excepción controlada: algunas funciones usan un diccionario local para
  acumular. Por ejemplo `Monoid.batch` y el combiner de `reduce_by_key`. Esa
  mutación nunca sale de la función, por lo que la función sigue siendo pura
  (misma entrada, misma salida, sin efectos observables).

## 4. Evaluación perezosa y fusión de etapas

`map`, `flat_map`, `filter`, `map_values` y `key_by` solo agregan un paso a
`ops`. Al ejecutar una acción, el plan se compone como una cadena de
generadores dentro de cada worker.

Resultado medido: en la primera versión cada etapa se ejecutaba por separado y
enviaba sus resultados de vuelta al proceso principal. Con 4 workers el conteo
de palabras era **3 veces más lento** que con 1. Al fusionar las etapas, el
mismo caso alcanzó **2.26x más rápido**.

## 5. Paralelismo

* `ProcessPoolExecutor` usa procesos, no hilos, porque el GIL de Python impide
  que los hilos ejecuten código Python en paralelo.
* Las funciones deben poder serializarse con `pickle`. Por eso los auxiliares
  son funciones de nivel de módulo o clases `frozen`, no lambdas.
* Si el plan contiene algo no serializable, o hay menos datos que
  `parallel_threshold`, se ejecuta secuencialmente con la misma semántica.
* El shuffle usa CRC32 en vez de `hash()`, porque `hash()` de un `str` cambia
  entre procesos (`PYTHONHASHSEED`).

## 6. Resultados y análisis de rendimiento

Ver [`benchmarks/resultados.md`](../benchmarks/resultados.md) y
[`benchmarks/speedup.png`](../benchmarks/speedup.png).

Observaciones:

1. **El speedup es sublineal** (1.5x a 2x con 4 workers). Cada registro se
   serializa para enviarlo a otro proceso, y ese costo es secuencial (ley de
   Amdahl). Las cargas con más cómputo por registro, como el conteo de
   palabras, escalan mejor.
2. **Los percentiles casi no escalan**: el cálculo por registro es muy barato
   comparado con el costo de serializar el diccionario. En un clúster real se
   mitiga leyendo cada partición directamente en el worker. `read_csv_sources`
   ya lo hace por archivo.
3. **La suma de punto flotante no es exactamente asociativa.** Al cambiar el
   número de workers cambia el orden de las sumas y el resultado difiere en
   ~1e-12. Por eso los benchmarks comparan con tolerancia relativa de 1e-9.
   Es una limitación real que también tienen Spark y Hadoop.
4. Para datos pequeños, `parallel_threshold` (5,000 por defecto) evita pagar
   el costo de crear procesos cuando no conviene.

## 7. Limitaciones y trabajo futuro

* Todo corre en una sola máquina. Para un clúster bastaría reemplazar
  `ProcessPoolExecutor` por un ejecutor distribuido (Dask, Ray) sin cambiar la
  API funcional.
* Los percentiles son aproximados y requieren conocer el rango `[lo, hi]`. Un
  t-digest o KLL sketch eliminaría ese requisito.
* `group_by_key`, `sort_*` y `take` se ejecutan en el proceso principal.
