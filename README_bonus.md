# Bonus - DSLR

Este documento explica las características adicionales implementadas en el proyecto DSLR.

## 1. Campos estadísticos extendidos en `describe.py`

### ¿Qué se añadió?

Además de las estadísticas básicas requeridas (Count, Mean, Std, Min, 25%, 50%, 75%, Max), se implementaron 6 campos adicionales:

| Campo | Descripción | Fórmula |
|---|---|---|
| **Mode** | Valor más frecuente en la distribución | Valor con mayor frecuencia |
| **Variance** | Dispersión de los datos respecto a la media | σ² = Σ(x - μ)² / (n-1) |
| **Range** | Diferencia entre el máximo y el mínimo | max - min |
| **Skewness** | Asimetría de la distribución | γ = Σ((x - μ)/σ)³ / n |
| **Kurtosis** | "Puntiagudez" de la distribución (exceso) | κ = Σ((x - μ)/σ)⁴ / n - 3 |
| **Sum** | Suma total de todos los valores | Σx |

### Implementación

Todas las funciones están implementadas manualmente sin usar funciones estadísticas de librerías:

```python
# describe.py
def my_variance(dataset, column):
    # Calcula la varianza muestral (n-1)
    
def my_mode(dataset, column):
    # Encuentra el valor más frecuente
    
def my_range(dataset, column):
    # max - min
    
def my_skewness(dataset, column):
    # Mide la asimetría de la distribución
    
def my_kurtosis(dataset, column):
    # Mide el exceso de curtosis
```

### Uso

```bash
python describe.py datasets/dataset_train.csv
```

Output extendido con las 14 filas de estadísticas.

---

## 2. Algoritmos de optimización

### ¿Qué se añadió?

Se implementaron 4 algoritmos de optimización para el entrenamiento del modelo:

### 2.1 Batch Gradient Descent (default)

**Descripción:** Calcula el gradiente usando todo el dataset en cada iteración.

**Características:**
- Estable y converge de forma suave
- Lento por iteración (procesamiento completo del dataset)
- Requiere menos iteraciones para converger

**Uso:**
```bash
python logreg_train.py datasets/dataset_train.csv -a batch
# o simplemente
python logreg_train.py datasets/dataset_train.csv
```

### 2.2 Stochastic Gradient Descent (SGD)

**Descripción:** Actualiza los pesos usando un solo ejemplo aleatorio por iteración.

**Características:**
- Muy rápido por iteración
- Convergencia ruidosa (oscila alrededor del mínimo)
- Requiere más iteraciones para converger
- Puede escapar de mínimos locales

**Uso:**
```bash
python logreg_train.py datasets/dataset_train.csv -a sgd
```

### 2.3 Mini-batch Gradient Descent

**Descripción:** Compromiso entre Batch y SGD. Usa batches de 32 ejemplos por iteración.

**Características:**
- Balance entre estabilidad y velocidad
- Menos ruidoso que SGD
- Aprovecha vectorización de numpy
- batch_size = 32 (configurable)

**Uso:**
```bash
python logreg_train.py datasets/dataset_train.csv -a mini_batch
```

### 2.4 Adam Optimizer

**Descripción:** Adaptive Moment Estimation. Combina momentum y learning rate adaptativo.

**Características:**
- Learning rate adaptativo por parámetro
- Momentum: acumula gradientes anteriores (β1 = 0.9)
- RMSprop: ajusta learning rate por varianza de gradientes (β2 = 0.999)
- Convergencia más rápida en la mayoría de casos
- Robusto a la elección de hiperparámetros

**Fórmulas:**
```
m_t = β1 * m_{t-1} + (1 - β1) * g_t     (primer momento)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²    (segundo momento)
m̂_t = m_t / (1 - β1^t)                   (corrección de bias)
v̂_t = v_t / (1 - β2^t)                   (corrección de bias)
θ = θ - lr * m̂_t / (√v̂_t + ε)
```

**Uso:**
```bash
python logreg_train.py datasets/dataset_train.csv -a adam
```

---

## 3. Características comunes

Todos los algoritmos incluyen:

- **Normalización z-score:** `(x - μ) / σ`
- **Detección de divergencia:** Si el loss aumenta >10% o es NaN
- **Early stopping:** Con patience = 100 iteraciones sin mejora
- **Max iteraciones:** 5000
- **Learning rate:** 0.05 (para batch, sgd, mini_batch)
- **One-vs-all:** Clasificación multiclase

---

## 4. Comparación de algoritmos

| Algoritmo | Velocidad/iter | Estabilidad | Iteraciones necesarias | Recomendación |
|---|---|---|---|---|
| **batch** | Lenta | Alta | Menos | Dataset pequeño/mediano |
| **sgd** | Muy rápida | Baja | Más | Dataset muy grande |
| **mini_batch** | Rápida | Media | Media | Balance general |
| **adam** | Rápida | Alta | Menos | Mejor opción general |

---

## 5. Ejemplos completos

### Entrenar con Adam y predecir

```bash
# Entrenar con Adam
python logreg_train.py datasets/dataset_train.csv -a adam

# Predecir
python logreg_predict.py datasets/dataset_test.csv

# Validar accuracy
python prediction/accuracity.py
python validation/validation.py
```

### Pipeline completo con mini_batch

```bash
python main.py datasets/dataset_train.csv -a mini_batch
```

### Comparar algoritmos

```bash
# Batch (default)
python logreg_train.py datasets/dataset_train.csv
python logreg_predict.py datasets/dataset_test.csv
# Accuracy: ~99%

# Adam
python logreg_train.py datasets/dataset_train.csv -a adam
python logreg_predict.py datasets/dataset_test.csv
# Accuracy: ~99%

# SGD
python logreg_train.py datasets/dataset_train.csv -a sgd
python logreg_predict.py datasets/dataset_test.csv
# Accuracy: ~98-99% (puede variar por la aleatoriedad)
```

---

## 6. Archivos modificados

| Archivo | Cambios |
|---|---|
| `describe.py` | Añadidas funciones: `my_variance`, `my_mode`, `my_range`, `my_skewness`, `my_kurtosis`, `my_sum` |
| `gradient_descent.py` | Refactorizado con funciones separadas: `_batch_gradient_descent`, `_stochastic_gradient_descent`, `_mini_batch_gradient_descent`, `_adam_optimizer` |
| `logreg_train.py` | Añadido argumento `-a` / `--algorithm` |
| `main.py` | Añadido argumento `-a` / `--algorithm` |

---

## 7. Notas técnicas

- **SGD y Mini-batch** usan `numpy` internamente para mejorar el rendimiento
- **Adam** mantiene estado de momentos (m, v) que se actualizan en cada iteración
- Todos los algoritmos guardan el modelo en el mismo formato (`model.json`)
- El parámetro `batch_size` en mini_batch es fijo (32) pero puede modificarse en el código
