# Proyecto 4 — Aprendizaje de Máquina
### Clasificación de Conducta Suicida en Bogotá

**Pontificia Universidad Javeriana — Ingeniería de Sistemas**

---

## Problema

¿Es posible predecir si una persona en Bogotá tendrá un **intento de suicidio** versus solo **ideación suicida**, basándose en sus factores de riesgo sociodemográficos y psicosociales?

Este problema de clasificación binaria tiene relevancia clínica directa: un modelo predictivo puede apoyar a profesionales de salud mental como herramienta de screening temprano, permitiendo priorizar intervenciones en casos de mayor riesgo.

---

## Dataset

- **Fuente:** [Datos Abiertos Bogotá — Secretaría Distrital de Salud](https://datosabiertos.bogota.gov.co/dataset/tasa-de-suicidio-en-bogota-d-c)
- **Registros:** ~254,000 casos individuales (2012–2026)
- **Target:** `clasificaciondelaconducta` → *Ideación suicida* / *Intento de Suicidio*
- **Features principales:** edad, sexo, ciclo vital, localidad, nivel educativo, y factores de riesgo binarios (maltrato sexual, conflicto de pareja, problemas económicos, problemas laborales, etc.)

---

## Técnicas aplicadas

| Técnica | Implementación | Notas |
|---------|---------------|-------|
| Random Forest (RF) | `sklearn.ensemble.RandomForestClassifier` | Entrenado sobre dataset completo |
| Support Vector Machine (SVM) | `sklearn.svm.SVC` | Entrenado sobre muestra estratificada de 15,000 registros (RBF escala O(n²)) |
| Red Neuronal (MLP) | `sklearn.neural_network.MLPClassifier` | Entrenado sobre dataset completo |

---

## Estructura del proyecto

```
ProyectoIA4/
│
├── data/
│   ├── raw/
│   │   └── conducta_suicida_bogota.csv       # Dataset original sin modificar
│   └── processed/
│       └── datos_limpios.csv                 # Dataset tras limpieza y estandarización
│
├── notebooks/
│   ├── 01_eda_limpieza.ipynb                 # Análisis exploratorio y limpieza
│   ├── 02_modelos.ipynb                      # Entrenamiento RF, SVM, MLP
│   └── 03_experimento.ipynb                  # Experimento comparativo de hiperparámetros
│
├── src/
│   ├── preprocessing.py                      # Carga, limpieza, encoding, split, escalado
│   ├── evaluation.py                         # Métricas y matriz de confusión
│   └── plots.py                              # Todas las funciones de visualización
│
├── results/
│   ├── eda/                                  # Gráficas del análisis exploratorio
│   ├── confusion_matrices/                   # Matrices de confusión por modelo
│   ├── training/                             # Curva de pérdida del MLP
│   ├── comparacion_modelos.png               # Barplot comparativo de métricas
│   └── experiment_table.csv                  # Tabla consolidada del experimento
│
└── requirements.txt
```

---

## Instalación y ejecución (desde cero)

### 0. Requisito previo — Python
Necesitas tener Python instalado. Si no lo tienes:
- Descárgalo desde **https://www.python.org/downloads/**
- Durante la instalación marca **"Add Python to PATH"**
- Cualquier versión 3.10+ funciona

### 1. Clonar el repositorio
```powershell
git clone <url-del-repo>
cd ProyectoIA4
```

### 2. Ejecutar el script de setup (una sola vez)
```powershell
.\setup.ps1
```
Esto crea el entorno virtual `.venv`, instala automáticamente todo lo que está en `requirements.txt` y registra el kernel de Jupyter para VS Code.

### 3. Abrir en VS Code
```powershell
code .
```
- Abre cualquier notebook en `notebooks/`
- Clic en el selector de kernel (arriba a la derecha) → **"Python (ProyectoIA4)"**
- Ejecutar en orden — el `01` genera el CSV limpio que usan los demás:
  1. `notebooks/01_eda_limpieza.ipynb`
  2. `notebooks/02_modelos.ipynb`
  3. `notebooks/03_experimento.ipynb`

---

## Diseño experimental

Para cada técnica se seleccionaron **3 hiperparámetros** con **2 valores** cada uno (8 combinaciones por técnica, 24 runs totales):

| Técnica | Hiperparámetro 1 | Hiperparámetro 2 | Hiperparámetro 3 |
|---------|-----------------|-----------------|------------------|
| RF | `n_estimators` [100, 300] | `max_depth` [5, 15] | `min_samples_split` [2, 10] |
| SVM | `C` [0.1, 10] | `kernel` [rbf, poly] | `gamma` [scale, 0.01] |
| MLP | `hidden_layer_sizes` [(50,), (100,100)] | `learning_rate_init` [0.001, 0.01] | `max_iter` [200, 500] |

---

## Resultados (resumen)

Los resultados completos se encuentran en `results/experiment_table.csv` y en las matrices de confusión de `results/confusion_matrices/`.

---

## Preprocesamiento

1. Normalización de nombres de columnas (minúsculas, sin espacios)
2. Eliminación de filas sin target
3. Imputación de valores faltantes con mediana
4. Codificación de variables categóricas con `LabelEncoder`
5. Escalado con `StandardScaler` (requerido por SVM y MLP)
6. Split 80/20 estratificado por clase

---

## Dependencias

```
pandas · numpy · scikit-learn · matplotlib · seaborn · jupyter
```
