# Proyecto 4 — Aprendizaje de Máquina
### Clasificación de Conducta Suicida en Bogotá

**Pontificia Universidad Javeriana — Ingeniería de Sistemas**

---

## Problema

¿Es posible predecir si una persona en Bogotá tendrá un **intento de suicidio** versus solo **ideación suicida**, basándose en sus factores de riesgo sociodemográficos y psicosociales?

Este problema de clasificación binaria tiene relevancia clínica directa: un modelo predictivo puede apoyar a profesionales de salud mental como herramienta de screening temprano.

---

## Dataset

- **Fuente:** [Datos Abiertos Bogotá — Secretaría Distrital de Salud](https://datosabiertos.bogota.gov.co/dataset/tasa-de-suicidio-en-bogota-d-c)
- **Tipo:** repositorio público de datos abiertos (equivalente a un benchmark de salud pública; el enunciado sugiere Kaggle como ejemplo, no como requisito exclusivo).
- **Registros:** ~254,000 casos (2012–2026)
- **Target:** `clasificaciondelaconducta` → *Ideación suicida* / *Intento de Suicidio*
- **Features:** edad, sexo, ciclo vital, localidad, nivel educativo, factores de riesgo binarios, etc.

---

## Técnicas aplicadas

| Técnica | Implementación | Entrenamiento |
|---------|---------------|---------------|
| Random Forest (RF) | `sklearn.ensemble.RandomForestClassifier` | Train completo (~203k) |
| Support Vector Machine (SVM) | `sklearn.svm.SVC` | Muestra estratificada 15,000 (costo O(n²)) |
| Red Neuronal (MLP) | `sklearn.neural_network.MLPClassifier` | Train completo |

---

## Estructura del proyecto

```
ProyectoIA4/
├── data/raw/                    # CSV original
├── data/processed/              # datos_limpios.csv
├── notebooks/
│   ├── 01_eda_limpieza.ipynb    # EDA, verificación y limpieza
│   ├── 03_experimento.ipynb     # Experimento 2³ + mejores modelos + análisis
│   └── 02_modelos.ipynb         # Entrenamiento con mejores HP del experimento
├── src/                         # preprocessing, plots, experiment_analysis, …
├── scripts/
│   ├── run_experiment.py        # Ejecuta experimento completo desde terminal
│   └── generate_ppt.py          # Genera plantilla PPT editable
├── results/                     # Tablas, gráficas, matrices (generados al ejecutar)
├── presentacion/                # Proyecto 4 - GX.pptx
└── requirements.txt
```

---

## Instalación (automática)

> **`git pull` no trae `.venv`** — en cada PC nuevo ejecuta **un solo comando** de abajo.  
> Requiere **Python 3.10+** instalado (no sirve 3.7).

| Sistema | Comando (elige uno) |
|---------|---------------------|
| **Windows** | Doble clic en `setup.bat` **o** `.\setup.ps1` **o** `py -3.11 scripts\bootstrap.py` |
| **macOS / Linux** | `./setup.sh` **o** `python3 scripts/bootstrap.py` |
| **Cursor / VS Code** | `Cmd/Ctrl+Shift+P` → **Tasks: Run Task** → **ProyectoIA4: Setup** |

Eso crea `.venv`, instala librerías, registra el kernel **Python (ProyectoIA4)** y las carpetas `results/`.

Luego en el notebook: kernel → **Python Environments** → **`.venv`** o **Python (ProyectoIA4)**.  
Si no aparece: **Enter interpreter path** → `ProyectoIA4/.venv/Scripts/python.exe` (Windows) o `.venv/bin/python` (Mac).

---

## Ejecución (orden recomendado)

1. **`notebooks/01_eda_limpieza.ipynb`** — genera `data/processed/datos_limpios.csv`
2. **`notebooks/03_experimento.ipynb`** — experimento factorial, tabla, análisis, matrices del **mejor modelo**
3. **`notebooks/02_modelos.ipynb`** — (opcional) reentrena con `results/best_configs.json`

**Alternativa rápida por terminal** (pasos 2 y 3 automatizados):
```bash
.venv/bin/python scripts/run_experiment.py
```

---

## Diseño experimental

Para cada técnica: **3 hiperparámetros × 2 valores** → 8 configuraciones × 3 técnicas = **24 runs**.

| Técnica | Hiperparámetro 1 | Hiperparámetro 2 | Hiperparámetro 3 |
|---------|-----------------|-----------------|------------------|
| RF | `n_estimators` [100, 200] | `max_depth` [10, 20] | `min_samples_split` [5, 10] |
| SVM | `C` [1, 10] | `kernel` [rbf, poly] | `gamma` [scale, auto] |
| MLP | `hidden_layer_sizes` [(100,50), (200,100)] | `activation` [relu, tanh] | `learning_rate_init` [0.001, 0.01] |

Criterio de mejor modelo: **F1 ponderado** (adecuado para desbalance 73% / 27%).

Salidas en `results/`:
- `experiment_table.csv`
- `analisis_comparativo.md`
- `best_configs.json`
- `confusion_matrices/cm_*_mejores.png`
- `experimento_accuracy.png`

---

## Presentación (PPT)

Generar plantilla editable (cambiar `--grupo` y nombres):

```bash
.venv/bin/python scripts/generate_ppt.py --grupo 4 --estudiantes "Ana Pérez\nLuis Gómez"
```

Archivo: `presentacion/Proyecto 4 - G4.pptx`  
Completar con capturas de notebooks y gráficas de `results/`.  
Entregar al buzón como **Proyecto 4 - GX** con comentario: Grupo X + nombres.

---

## Preprocesamiento

1. Normalización de nombres de columnas
2. Eliminación de duplicados y filas sin target
3. Verificación de valores de target y outliers en edad
4. Estandarización de nivel educativo e imputación de nulos
5. Codificación `LabelEncoder` + imputación mediana
6. `StandardScaler` para SVM/MLP
7. Split 80/20 estratificado

---

## Dependencias

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `python-pptx`
