
# DSLR — Análisis y modelos (Visualización y ML)

Proyecto de ejemplo para análisis de datos, visualización y modelos de aprendizaje automático.
Contiene scripts para explorar datasets, visualizar relaciones entre características, entrenar modelos
mediante descenso de gradiente y realizar predicciones con modelos lineales y de regresión logística.

## Estructura del repositorio

- `describe.py` — funciones auxiliares para describir datasets.
- `gradient_descent.py` — implementación de descenso por gradiente para regresión lineal.
- `histogram.py` — generación de histogramas para variables del dataset.
- `logreg_predict.py` — predicción con un modelo de regresión logística (inferencia).
- `main.py` — script principal / ejemplo de ejecución (puede orquestar otros módulos).
- `pair_plot.py` — gráficos de pares para explorar correlaciones entre variables.
- `scatter_plot.py` — gráficos de dispersión personalizados.
- `datasets/` — carpeta con los datasets de entrenamiento y test (`dataset_train.csv`, `dataset_test.csv`).
- `prediction/accuracity.py` — métricas de evaluación (precisión, exactitud, etc.).
- `validation/validation.py` — rutinas de validación y evaluación de modelos.

## Requisitos

- Python 3.8 o superior
- Paquetes recomendados: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

Instalación rápida (recomendado dentro de un virtualenv o entorno conda):

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate     # Windows PowerShell
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Uso

Ejemplos de ejecución desde la raíz del proyecto `DSLR`:

- Ejecutar el script principal de ejemplo:

```bash
python main.py
```

- Generar un pair-plot para explorar el dataset:

```bash
python pair_plot.py
```

- Entrenar o probar la implementación de descenso por gradiente:

```bash
python gradient_descent.py
```

- Ejecutar la predicción con el modelo de regresión logística:

```bash
python logreg_predict.py --input datasets/dataset_test.csv
```

Cada script tiene comentarios al inicio explicando parámetros y opciones disponibles.

## Datos

Los datasets de ejemplo están en la carpeta `datasets/`. Revisa `dataset_train.csv` y `dataset_test.csv`.
Si quieres usar tus propios datos, coloca el archivo CSV en `datasets/` y adapta los nombres de columnas
en los scripts según sea necesario.

## Contribuciones

Si quieres mejorar este repositorio:

1. Crea una rama nueva `feature/tu-cambio`.
2. Añade tests o ejemplos de uso si modificas la lógica.
3. Abre un pull request explicando los cambios.

## Licencia

Este proyecto no incluye una licencia explícita. Añade una licencia (ej. MIT) si planeas compartirlo públicamente.

## Contacto

Para dudas o mejoras, puedes abrir un issue en el repositorio o contactarme directamente.

