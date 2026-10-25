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

El análisis se divide en cuatro fases:

**a) Preprocesado y PCA**
- División train/test (70% / 30%) antes de cualquier transformación, para evitar fuga de información.
- Se elimina `Inspection rating` (varianza casi nula).
- `Insurance cost` se trata como binaria y `Fuel type` con one-hot.
- Se elimina `Manufacturer`: la marca se deduce del modelo, y mantener las dos produce colinealidad perfecta (demostrada con el rango de la matriz).
- Outliers eliminados con la regla de 3σ (solo en train).
- Estandarización y después imputación con `KNNImputer`.
- PCA para visualizar los datos y el precio en dos dimensiones.

**b) Regresión lineal**
- Modelo base sobre `Price`: residuos en forma de U y precios negativos.
- Transformación logarítmica del precio, porque los efectos son multiplicativos.
- Interpretación de los pesos como efectos porcentuales y diagnóstico de residuos.

**c) Ridge y LASSO**
- Selección de `alpha` por validación cruzada (`RidgeCV`, `LassoCV`).
- Camino de regularización de LASSO y regla de 1 error estándar para ver qué variables se pueden eliminar.

**d) Interacciones**
- `PolynomialFeatures` de grado 2 y nuevo ajuste de Ridge y LASSO, sobre `Price` y sobre `log(Price)`.

## Resultados

| Modelo | Objetivo | R² test | MAE test | MAPE test |
|---|---|---|---|---|
| Regresión lineal | `Price` | 0.726 | 5.121 € | 305% |
| LASSO grado 2 | `Price` | 0.922 | 2.049 € | 191% |
| **Regresión lineal** | **`log(Price)`** | **0.987** | **878 €** | **7.3%** |
| LASSO grado 2 | `log(Price)` | 0.977 | 888 € | 6.9% |

## Conclusiones

- **El mejor modelo es la regresión lineal sobre `log(Price)`**: precisa (error medio del ~7%), simple y fácil de interpretar.
- El precio se explica sobre todo por la **antigüedad** (~6% menos por cada año), el **kilometraje** (~41% menos cada 68.000 km), el **modelo** y el **tamaño del motor**.
- Las **valoraciones subjetivas** (reseñas, equipo de sonido, confort...) apenas influyen en el precio. LASSO las elimina.
- Las interacciones de grado 2 corrigen los residuos del modelo sobre `Price`, pero no aportan nada sobre `log(Price)`: el logaritmo ya capta de forma exacta los efectos multiplicativos.

## Cómo ejecutarlo

```bash
pip install apafib scikit-learn pandas numpy matplotlib seaborn scipy jupyter
```

```bash
jupyter notebook individual.ipynb
```

Versiones utilizadas: Python 3.13, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2, seaborn 0.13.2, scipy 1.18.1, apafib 0.5.2.
