
# DSLR — Clasificación de casas de Hogwarts

Proyecto de clasificación mediante regresión logística multiclase (one-vs-all).
Analiza un dataset de estudiantes de Hogwarts, visualiza relaciones entre características,
entrena un modelo con gradient descent y predice la casa de cada estudiante.

## Estructura del repositorio

| Archivo / Directorio | Descripción |
|---|---|
| `logreg_train.py` | Entrena el modelo de regresión logística y genera `model.json` |
| `logreg_predict.py` | Predice las casas usando el modelo entrenado, genera `houses.csv` |
| `main.py` | Pipeline completo: describe, histogramas, scatter plots, pair plot y entrenamiento |
| `describe.py` | Estadísticas descriptivas del dataset (count, mean, std, min, max, percentiles) |
| `histogram.py` | Histogramas por casa con test de Kruskal-Wallis para homogeneidad |
| `scatter_plot.py` | Gráficos de dispersión para correlaciones de Pearson (|r| >= 0.85) |
| `pair_plot.py` | Pair plots con eliminación de features correlacionadas (|r| >= 0.8) |
| `gradient_descent.py` | Implementación de gradient descent para regresión logística multiclase |
| `datasets/` | Datasets de entrenamiento y test |
| `prediction/accuracity.py` | Divide el dataset en train/validation |
| `validation/validation.py` | Calcula la accuracy del modelo |

## Requisitos

- Python 3.8 o superior
- Dependencias: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

Instalación:

```bash
python -m venv venv
venv\Scripts\activate        # Windows PowerShell
source venv/bin/activate     # Linux / macOS
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Uso

Todos los comandos se ejecutan desde la raíz del proyecto `DSLR`.

### 1. Explorar el dataset (visualización completa)

```bash
python main.py datasets/dataset_train.csv
```

Genera estadísticas descriptivas, histogramas, scatter plots de correlaciones y pair plots.

### 2. Entrenar el modelo

```bash
python logreg_train.py datasets/dataset_train.csv
```

Genera `model.json` con los pesos, bias, medias y desviaciones estándar.

### 3. Predecir casas

```bash
python logreg_predict.py datasets/dataset_test.csv
```

Genera `houses.csv` con las predicciones. Opciones:
- `-v` / `--verbose`: muestra detalles de las predicciones
- `--model <ruta>`: usar un modelo alternativo (default: `model.json`)

### 4. Validar accuracy

```bash
python prediction/accuracity.py
python validation/validation.py
```

### Scripts individuales

```bash
python describe.py datasets/dataset_train.csv
python histogram.py          # (se usa desde main.py)
python scatter_plot.py       # (se usa desde main.py)
python pair_plot.py          # (se usa desde main.py)
```

## Archivos generados

| Archivo | Descripción | Generado por |
|---|---|---|
| `model.json` | Pesos, bias, estadísticas del modelo | `logreg_train.py` |
| `houses.csv` | Predicciones de casas | `logreg_predict.py` |
| `weights.csv` | Pesos del modelo en CSV | `logreg_predict.py` |

## Datos

Los datasets están en `datasets/`. Se espera una columna `Hogwarts House` con valores:
`Gryffindor`, `Slytherin`, `Ravenclaw`, `Hufflepuff`.

