#!/usr/bin/env bash
# setup.sh — ejecutar una sola vez después de git clone (macOS / Linux)
set -e

echo ""
echo "=== Setup ProyectoIA4 ==="

if ! command -v python3 &>/dev/null; then
  echo "[ERROR] python3 no encontrado. Instálalo desde https://www.python.org/downloads/"
  exit 1
fi

echo "Python detectado: $(python3 --version)"

if [ ! -d ".venv" ]; then
  echo ""
  echo "[1/3] Creando entorno virtual..."
  python3 -m venv .venv
else
  echo ""
  echo "[1/3] Entorno virtual ya existe."
fi

echo ""
echo "[2/3] Instalando dependencias..."
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "[3/3] Registrando kernel de Jupyter..."
.venv/bin/python -m ipykernel install --user --name proyectoIA4 --display-name "Python (ProyectoIA4)"

mkdir -p results/eda results/confusion_matrices results/training results/experiment

echo ""
echo "=== Listo ==="
echo "Ejecuta los notebooks en orden:"
echo "  1. notebooks/01_eda_limpieza.ipynb"
echo "  2. notebooks/03_experimento.ipynb"
echo "  3. notebooks/02_modelos.ipynb  (opcional; usa configs del experimento)"
echo ""
echo "O ejecuta el experimento completo desde terminal:"
echo "  .venv/bin/python scripts/run_experiment.py"
