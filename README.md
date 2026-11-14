# Predicción del precio de coches usados

Modelo de machine learning para estimar el precio de un coche usado a partir de sus características, comparando regresión lineal, Ridge y LASSO.

<p align="center">
  <img src="https://media.giphy.com/media/2FHNqsKHq3aHjHdy0a/giphy.gif" alt="Toyota Yaris de rally saltando" width="450">
</p>

## Objetivo

A partir de las características de 750 coches usados (marca, modelo, año, kilometraje, motor, estado, costes...), predecir su precio de venta y entender **qué factores determinan ese precio**.

## Dataset

Se carga con la función `load_car_sales` de la librería [`apafib`](https://pypi.org/project/apafib/):

- **750 coches** de 5 marcas (Toyota, Ford, VW, BMW, Porsche) y 15 modelos.
- **16 variables**: continuas (año, kilometraje, tamaño del motor, valoraciones de 0 a 10, costes) y categóricas (marca, modelo, combustible, seguro).
- **Variable objetivo:** `Price`.
- Hay valores perdidos (menos del 8%) en casi todas las variables continuas.

## Metodología

**Criterio de evaluación (el mismo en todo el análisis):** MAE en euros.
- El error de generalización de cada modelo es el de **validación cruzada** (10 folds sobre train). Es el único que se usa para ajustar hiperparámetros y elegir modelos.
- El error de **test** solo sirve para detectar sobreajuste: debe caer dentro de la distribución de errores de validación cruzada.
- El error sobre train no se usa.
- Los modelos sobre `log(Price)` deshacen la transformación (`TransformedTargetRegressor`) y se evalúan también en euros.

El análisis se divide en cuatro fases:

**a) Preprocesado y PCA**
- División train/test (70% / 30%) antes de cualquier transformación, para evitar fuga de información. Todo el preprocesado se ajusta con train y se aplica al test.
- Se elimina `Inspection rating` (varianza casi nula).
- `Insurance cost` se trata como binaria y `Fuel type` con one-hot.
- Se elimina `Manufacturer`: la marca se deduce del modelo, y mantener las dos produce colinealidad perfecta (demostrada con el rango de la matriz).
- Outliers eliminados con la regla de 3σ (solo en train).
- Estandarización y después imputación con `KNNImputer` (número de vecinos elegido por validación cruzada).
- PCA sobre todas las variables estandarizadas para visualizar los datos y el precio en dos dimensiones.

**b) Regresión lineal**
- Modelo base sobre `Price`: residuos en forma de U y precios negativos.
- Transformación logarítmica del precio, porque los efectos son multiplicativos. Se elige porque reduce el MAE de validación cruzada de unos 5.000 € a unos 980 €.
- Interpretación de los pesos como efectos porcentuales y diagnóstico de residuos (con predicciones de validación cruzada).

**c) Ridge y LASSO**
- Selección de `alpha` por validación cruzada (`GridSearchCV`, MAE en euros).
- Camino de regularización de LASSO y regla de 1 error estándar para ver qué variables se pueden eliminar.

**d) Interacciones**
- `PolynomialFeatures` de grado 2 y nuevo ajuste de Ridge y LASSO, sobre `Price` y sobre `log(Price)`.

## Resultados

Todos los modelos se evalúan con el mismo criterio, el **MAE en euros**: el error de generalización es la media ± desviación del MAE en validación cruzada de 10 folds sobre train, y el test solo se usa para comprobar que no hay sobreajuste (que su MAE cae dentro de la distribución de CV).

| Modelo | Objetivo | MAE CV | MAE test |
|---|---|---|---|
| Regresión lineal | `Price` | 5.028 ± 1.047 € | 5.207 € |
| LASSO grado 2 | `Price` | 2.004 ± 675 € | 1.972 € |
| **Regresión lineal** | **`log(Price)`** | **978 ± 262 €** | **886 €** |
| LASSO grado 2 | `log(Price)` | 993 ± 404 € | 964 € |

## Conclusiones

- **El mejor modelo es la regresión lineal sobre `log(Price)`**: tiene el menor error de validación cruzada (~980 €) y su error de test (886 €) cae dentro de esa distribución, sin sobreajuste. LASSO de grado 2 sobre `log(Price)` empata estadísticamente, pero es más complejo y menos estable, así que se prefiere el modelo simple.
- El precio se explica sobre todo por la **antigüedad** (~6% menos por cada año), el **kilometraje** (~42% menos cada 64.000 km), el **modelo** y el **tamaño del motor**.
- Las **valoraciones subjetivas** (reseñas, equipo de sonido, confort...) apenas influyen en el precio. LASSO las elimina.
- Las interacciones de grado 2 corrigen los residuos del modelo sobre `Price`, pero no aportan nada sobre `log(Price)`: el logaritmo ya capta de forma exacta los efectos multiplicativos.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook individual.ipynb
```

Versiones utilizadas: Python 3.13, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2, seaborn 0.13.2, scipy 1.18.1, apafib 0.5.2.
